"""
observation_encoder.py
======================
Encodes a *variable-length* set of within-day MODIS swath observations into a
single fixed-length daily embedding.

Design rationale
----------------
MODIS (Terra / Aqua) makes multiple swath passes over any given location every
day.  The number of usable, cloud-free observations varies from 0 to ~6.  A
simple mean-pool loses the fine-grained angular / temporal sampling diversity;
an RNN assumes an ordering that does not naturally exist.

We instead treat each day's observations as an *unordered set* and apply a
Perceiver / Set-Attention style cross-attention module:

  1.  Each observation vector o_i is projected to a common hidden dimension.
  2.  A small set of learned query vectors (``num_latents``) attend over the
      projected observations through multi-head cross-attention.
  3.  The resulting latent matrix is flattened + projected to ``embed_dim``.

Missing-band handling (MYD / Aqua nighttime)
--------------------------------------------
Aqua night-time overpasses lack MODIS bands 1-7 (visible / NIR).  We handle
this with a *band-mask* token approach:

  * For every observation a boolean mask vector ``band_mask`` of length
    ``num_bands`` is provided.  Masked-out bands are set to 0.0 before the
    input linear projection.
  * A learnable *missing-band embedding* of shape ``(num_bands, hidden_dim)``
    is **added** to the projected features element-wise for each masked band.
    This allows the model to distinguish "band present but near-zero" from
    "band absent".

Usage
-----
    enc = DailyObservationEncoder(num_bands=36)
    # obs: (B, T_obs, num_bands)  — T_obs varies per day
    # mask: (B, T_obs, num_bands) — True where band is MISSING
    daily_emb = enc(obs, band_mask=mask)  # → (B, embed_dim)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DailyObservationEncoder(nn.Module):
    """Set-attention encoder for within-day MODIS observations.

    Parameters
    ----------
    num_bands : int
        Number of MODIS spectral bands in each observation vector.
        For MOD/MYD02 TOA reflectance products this is typically 36.
    hidden_dim : int
        Internal projection dimension.
    embed_dim : int
        Output embedding dimension passed downstream to the temporal model.
    num_latents : int
        Number of learned query (latent) vectors.  Controls information
        compression; 4-8 is usually sufficient for ~36 input bands.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate applied inside the cross-attention and feed-forward layers.
    """

    def __init__(
        self,
        num_bands: int = 36,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        num_latents: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # --- input projection ---
        self.input_proj = nn.Linear(num_bands, hidden_dim)

        # --- missing-band embedding: one learnable vector per band ---
        # Added to the projected features at positions where that band is absent.
        self.missing_band_embed = nn.Parameter(
            torch.zeros(num_bands, hidden_dim)
        )
        nn.init.normal_(self.missing_band_embed, std=0.02)

        # --- learned latent queries ---
        self.latent_queries = nn.Parameter(
            torch.empty(num_latents, hidden_dim)
        )
        nn.init.normal_(self.latent_queries, std=0.02)

        # --- cross-attention: latents attend over observations ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # --- feed-forward refinement ---
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)

        # --- output projection: (num_latents * hidden_dim) → embed_dim ---
        self.output_proj = nn.Linear(num_latents * hidden_dim, embed_dim)

        # --- fallback for days with zero valid observations ---
        self.no_obs_embed = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.no_obs_embed, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        observations: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None,
        obs_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        observations : Tensor, shape (B, T_obs, num_bands)
            Swath-level TOA reflectance / BT values.  Absent bands should
            already be set to 0.0.
        band_mask : BoolTensor, shape (B, T_obs, num_bands), optional
            True where a band value is **missing** (e.g. visible bands for
            night-time MYD passes).
        obs_padding_mask : BoolTensor, shape (B, T_obs), optional
            True for *padded* (non-existent) observation slots so that days
            with fewer passes can be batched to a common T_obs length.

        Returns
        -------
        Tensor, shape (B, embed_dim)
            One embedding per day per batch element.
        """
        B, T_obs, _ = observations.shape

        # 1.  Zero-out missing bands (defensive; caller should already do this)
        if band_mask is not None:
            observations = observations.masked_fill(band_mask, 0.0)

        # 2.  Project all observations to hidden_dim: (B, T_obs, hidden_dim)
        x = self.input_proj(observations)

        # 3.  Add missing-band correction signal
        if band_mask is not None:
            # band_mask: (B, T_obs, num_bands) → float (B, T_obs, num_bands, 1)
            # missing_band_embed: (num_bands, hidden_dim)
            # contribution: sum over bands dimension of (mask * embed)
            correction = torch.einsum(
                "btn,nh->bth",
                band_mask.float(),
                self.missing_band_embed,
            )
            x = x + correction

        # 4.  Expand latent queries to batch: (B, num_latents, hidden_dim)
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)

        # 5.  Cross-attention (latents → observations)
        attn_out, _ = self.cross_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=obs_padding_mask,
        )
        latents = self.cross_attn_norm(queries + attn_out)

        # 6.  Feed-forward refinement
        latents = self.ff_norm(latents + self.ff(latents))

        # 7.  Handle days with *all* observations padded (no valid data)
        if obs_padding_mask is not None:
            all_padded = obs_padding_mask.all(dim=1)  # (B,)
        else:
            all_padded = None

        # 8.  Flatten latents and project to embed_dim
        flat = latents.reshape(B, -1)          # (B, num_latents * hidden_dim)
        daily_emb = self.output_proj(flat)     # (B, embed_dim)

        # 9.  Replace embedding with the no-observation fallback where needed
        if all_padded is not None and all_padded.any():
            fallback = self.no_obs_embed.unsqueeze(0).expand(B, -1)
            daily_emb = torch.where(
                all_padded.unsqueeze(1), fallback, daily_emb
            )

        return daily_emb
