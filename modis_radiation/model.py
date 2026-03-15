"""
model.py
========
Full temporal model for sea-surface net-radiation inversion.

Architecture
------------
Input (per pixel / grid cell, per day)
  └─ DailyObservationEncoder           → daily embedding  (embed_dim,)
        ↓  (sequence of W days)
  └─ Positional encoding
        ↓
  └─ Bidirectional LSTM                → contextual features
        ↓
  └─ Transformer encoder (causal mask)
        ↓
  └─ MLP regression head               → predicted net radiation (W/m²)

The causal mask on the Transformer guarantees that when predicting the
net radiation at day t only days ≤ t are visible, which is required for
real-time / operational use.  During training we predict every day in the
window simultaneously (teacher-forcing style) so that each GPU batch covers
W predictions.

Window size
-----------
W = 3 or 5 days is recommended.  Larger windows capture cloud-cover cycles
and synoptic weather patterns; smaller windows reduce label latency.
Empirically W = 5 strikes a good balance for ocean radiation.

Model parameters (defaults)
---------------------------
embed_dim  : 256   (output of DailyObservationEncoder)
lstm_hidden: 256
lstm_layers: 2
tf_heads   : 4
tf_layers  : 2
mlp_hidden : 128
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .observation_encoder import DailyObservationEncoder


# ---------------------------------------------------------------------------
# Positional encoding (sinusoidal, standard)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Adds sinusoidal positional encodings to a sequence tensor."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)           # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)                           # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class NetRadiationModel(nn.Module):
    """MODIS TOA → sea-surface net radiation (W/m²) time-series model.

    Parameters
    ----------
    num_bands : int
        MODIS bands per observation (default 36 for MOD/MYD02).
    obs_hidden_dim : int
        Internal dimension of DailyObservationEncoder.
    embed_dim : int
        Daily embedding dimension (output of encoder, input of LSTM).
    num_latents : int
        Number of set-attention latent vectors in the daily encoder.
    obs_heads : int
        Attention heads in the daily encoder cross-attention.
    lstm_hidden : int
        Hidden size of each LSTM direction.
    lstm_layers : int
        Number of stacked LSTM layers.
    tf_hidden : int
        Transformer model dimension (projected from 2*lstm_hidden if bidir).
    tf_heads : int
        Transformer encoder attention heads.
    tf_layers : int
        Number of Transformer encoder layers.
    tf_ff_dim : int
        Transformer feed-forward intermediate size.
    mlp_hidden : int
        MLP regression head hidden size.
    window : int
        Time-series window length W (days).
    dropout : float
        Dropout probability applied throughout.
    """

    def __init__(
        self,
        num_bands: int = 36,
        obs_hidden_dim: int = 128,
        embed_dim: int = 256,
        num_latents: int = 4,
        obs_heads: int = 4,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        tf_hidden: int = 256,
        tf_heads: int = 4,
        tf_layers: int = 2,
        tf_ff_dim: int = 512,
        mlp_hidden: int = 128,
        window: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window = window
        self.lstm_hidden = lstm_hidden

        # ---- daily encoder ----
        self.daily_encoder = DailyObservationEncoder(
            num_bands=num_bands,
            hidden_dim=obs_hidden_dim,
            embed_dim=embed_dim,
            num_latents=num_latents,
            num_heads=obs_heads,
            dropout=dropout,
        )

        # ---- optional auxiliary features projection ----
        # Auxiliary scalar features per day (e.g. solar zenith angle, DOY sine/cos,
        # land/sea mask, DEM) can be concatenated before the LSTM.
        # Set aux_dim > 0 to enable; default disabled (aux_dim = 0).
        self.aux_dim = 0  # updated by set_aux_dim()
        self._embed_to_lstm_in = embed_dim  # updated if aux added

        # ---- LSTM ----
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # ---- project LSTM output to tf_hidden ----
        self.lstm_proj = nn.Linear(lstm_out_dim, tf_hidden)

        # ---- positional encoding ----
        self.pos_enc = SinusoidalPositionalEncoding(tf_hidden, dropout=dropout)

        # ---- Transformer encoder (causal) ----
        tf_layer = nn.TransformerEncoderLayer(
            d_model=tf_hidden,
            nhead=tf_heads,
            dim_feedforward=tf_ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(tf_layer, num_layers=tf_layers)

        # ---- MLP regression head ----
        self.head = nn.Sequential(
            nn.Linear(tf_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    # ------------------------------------------------------------------
    def set_aux_dim(self, aux_dim: int) -> None:
        """Enable auxiliary features.  Must be called before first forward."""
        self.aux_dim = aux_dim
        old_in = self._embed_to_lstm_in
        self._embed_to_lstm_in = old_in + aux_dim
        self.lstm = nn.LSTM(
            input_size=self._embed_to_lstm_in,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.lstm.dropout,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask for Transformer (True = ignore)."""
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    # ------------------------------------------------------------------
    def encode_daily(
        self,
        obs: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None,
        obs_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode per-day observation sets.

        Parameters
        ----------
        obs : Tensor, shape (B, W, T_obs, num_bands)
        band_mask : BoolTensor, shape (B, W, T_obs, num_bands), optional
        obs_padding_mask : BoolTensor, shape (B, W, T_obs), optional

        Returns
        -------
        Tensor, shape (B, W, embed_dim)
        """
        B, W, T_obs, C = obs.shape
        # Flatten the W dimension into batch for efficient encoder call
        obs_flat = obs.reshape(B * W, T_obs, C)
        bm_flat = band_mask.reshape(B * W, T_obs, C) if band_mask is not None else None
        pm_flat = obs_padding_mask.reshape(B * W, T_obs) if obs_padding_mask is not None else None

        emb_flat = self.daily_encoder(obs_flat, bm_flat, pm_flat)  # (B*W, embed_dim)
        return emb_flat.reshape(B, W, -1)                          # (B, W, embed_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None,
        obs_padding_mask: Optional[torch.Tensor] = None,
        aux_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor, shape (B, W, T_obs, num_bands)
            TOA values for each day in the window and each swath observation.
        band_mask : BoolTensor, shape (B, W, T_obs, num_bands), optional
            True where a band is **absent** (nighttime MYD visible bands, etc.).
        obs_padding_mask : BoolTensor, shape (B, W, T_obs), optional
            True for padded (non-existent) observation slots.
        aux_features : Tensor, shape (B, W, aux_dim), optional
            Per-day auxiliary scalar features (solar zenith, DOY, etc.).

        Returns
        -------
        Tensor, shape (B, W)
            Predicted net radiation (W/m²) for each day in the window.
        """
        # 1. Encode daily observations
        daily_emb = self.encode_daily(obs, band_mask, obs_padding_mask)  # (B, W, embed_dim)

        # 2. Optionally append auxiliary features
        if aux_features is not None:
            daily_emb = torch.cat([daily_emb, aux_features], dim=-1)     # (B, W, embed_dim+aux)

        # 3. LSTM
        lstm_out, _ = self.lstm(daily_emb)                               # (B, W, 2*lstm_hidden)

        # 4. Project + positional encoding
        x = self.lstm_proj(lstm_out)                                     # (B, W, tf_hidden)
        x = self.pos_enc(x)

        # 5. Causal Transformer encoder
        causal_mask = self._causal_mask(x.size(1), x.device)
        x = self.transformer(x, mask=causal_mask)                        # (B, W, tf_hidden)

        # 6. Regression head
        out = self.head(x).squeeze(-1)                                   # (B, W)
        return out
