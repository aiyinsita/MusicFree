"""
tests/test_modis_radiation.py
==============================
Unit and integration tests for the modis_radiation module.

Run with:
    pytest tests/test_modis_radiation.py -v
"""

import math

import numpy as np
import pytest
import torch

from modis_radiation.observation_encoder import DailyObservationEncoder
from modis_radiation.model import NetRadiationModel, SinusoidalPositionalEncoding
from modis_radiation.dataset import (
    ModisRadiationDataset,
    DataConfig,
    build_band_mask_for_night,
    compute_normalisation_stats,
    years_to_slice,
    DEFAULT_SPLIT,
    MODIS_VISIBLE_BANDS,
)
from modis_radiation.train import compute_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_BANDS = 36
WINDOW = 5
MAX_OBS = 4
BATCH = 2
HIDDEN = 64
EMBED = 128


@pytest.fixture()
def encoder():
    return DailyObservationEncoder(
        num_bands=NUM_BANDS,
        hidden_dim=HIDDEN,
        embed_dim=EMBED,
        num_latents=4,
        num_heads=4,
        dropout=0.0,
    )


@pytest.fixture()
def model():
    return NetRadiationModel(
        num_bands=NUM_BANDS,
        obs_hidden_dim=HIDDEN,
        embed_dim=EMBED,
        num_latents=4,
        obs_heads=4,
        lstm_hidden=HIDDEN,
        lstm_layers=1,
        tf_hidden=EMBED,
        tf_heads=4,
        tf_layers=1,
        tf_ff_dim=HIDDEN * 2,
        mlp_hidden=HIDDEN,
        window=WINDOW,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# DailyObservationEncoder tests
# ---------------------------------------------------------------------------

class TestDailyObservationEncoder:

    def test_output_shape_basic(self, encoder):
        """Encoder returns (B, embed_dim) given a clean observation tensor."""
        obs = torch.randn(BATCH, MAX_OBS, NUM_BANDS)
        out = encoder(obs)
        assert out.shape == (BATCH, EMBED)

    def test_output_shape_with_band_mask(self, encoder):
        """Encoder handles missing bands without shape change."""
        obs = torch.randn(BATCH, MAX_OBS, NUM_BANDS)
        # mask first 7 bands (visible) for all observations in batch
        band_mask = torch.zeros(BATCH, MAX_OBS, NUM_BANDS, dtype=torch.bool)
        band_mask[:, :, :7] = True
        obs_masked = obs.clone()
        obs_masked[band_mask] = 0.0
        out = encoder(obs_masked, band_mask=band_mask)
        assert out.shape == (BATCH, EMBED)

    def test_output_shape_with_padding_mask(self, encoder):
        """Encoder handles padding (variable observation count per day)."""
        obs = torch.randn(BATCH, MAX_OBS, NUM_BANDS)
        # second sample has only 2 valid observations
        obs_pad = torch.zeros(BATCH, MAX_OBS, dtype=torch.bool)
        obs_pad[1, 2:] = True
        out = encoder(obs, obs_padding_mask=obs_pad)
        assert out.shape == (BATCH, EMBED)

    def test_all_padded_uses_fallback(self, encoder):
        """When all observation slots are padded the fallback embedding is used."""
        obs = torch.zeros(1, MAX_OBS, NUM_BANDS)
        obs_pad = torch.ones(1, MAX_OBS, dtype=torch.bool)  # all padded
        out = encoder(obs, obs_padding_mask=obs_pad)
        # Should equal the no_obs_embed parameter
        expected = encoder.no_obs_embed.unsqueeze(0)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_no_nan_in_output(self, encoder):
        """Encoder output should not contain NaN for normal inputs."""
        obs = torch.randn(4, MAX_OBS, NUM_BANDS)
        out = encoder(obs)
        assert not torch.isnan(out).any()

    def test_night_band_masking_changes_output(self, encoder):
        """Masking visible bands should change the encoder output."""
        torch.manual_seed(0)
        obs = torch.randn(BATCH, MAX_OBS, NUM_BANDS)
        out_day = encoder(obs)

        band_mask = torch.zeros(BATCH, MAX_OBS, NUM_BANDS, dtype=torch.bool)
        band_mask[:, :, :7] = True
        obs_night = obs.clone()
        obs_night[band_mask] = 0.0
        out_night = encoder(obs_night, band_mask=band_mask)

        assert not torch.allclose(out_day, out_night)


# ---------------------------------------------------------------------------
# NetRadiationModel tests
# ---------------------------------------------------------------------------

class TestNetRadiationModel:

    def test_forward_output_shape(self, model):
        """Model returns (B, W) predictions."""
        obs = torch.randn(BATCH, WINDOW, MAX_OBS, NUM_BANDS)
        out = model(obs)
        assert out.shape == (BATCH, WINDOW)

    def test_forward_with_masks(self, model):
        """Model runs correctly with band_mask and obs_padding_mask."""
        obs = torch.randn(BATCH, WINDOW, MAX_OBS, NUM_BANDS)
        band_mask = torch.zeros(BATCH, WINDOW, MAX_OBS, NUM_BANDS, dtype=torch.bool)
        band_mask[:, :, :, :7] = True
        obs_pad = torch.zeros(BATCH, WINDOW, MAX_OBS, dtype=torch.bool)
        obs_pad[:, :, 3:] = True

        out = model(obs, band_mask, obs_pad)
        assert out.shape == (BATCH, WINDOW)
        assert not torch.isnan(out).any()

    def test_forward_with_aux_features(self, model):
        """Model accepts auxiliary features."""
        obs = torch.randn(BATCH, WINDOW, MAX_OBS, NUM_BANDS)
        aux = torch.randn(BATCH, WINDOW, 5)
        model.set_aux_dim(5)
        out = model(obs, aux_features=aux)
        assert out.shape == (BATCH, WINDOW)

    def test_no_nan_output(self, model):
        """No NaN in predictions for random inputs."""
        obs = torch.randn(BATCH, WINDOW, MAX_OBS, NUM_BANDS)
        out = model(obs)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self, model):
        """Gradients should flow to all parameters (including fallback embeddings)."""
        obs = torch.randn(BATCH, WINDOW, MAX_OBS, NUM_BANDS)
        tgt = torch.randn(BATCH, WINDOW)

        # Provide a band_mask so that missing_band_embed participates.
        band_mask = torch.zeros(BATCH, WINDOW, MAX_OBS, NUM_BANDS, dtype=torch.bool)
        band_mask[:, :, :, :7] = True

        # Pad ALL observations in one sample so that no_obs_embed participates.
        obs_pad = torch.zeros(BATCH, WINDOW, MAX_OBS, dtype=torch.bool)
        obs_pad[0, :, :] = True  # all slots padded for sample 0

        loss = ((model(obs, band_mask=band_mask, obs_padding_mask=obs_pad) - tgt) ** 2).mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDatasetHelpers:

    def _make_arrays(self, n_days=100, n_lat=4, n_lon=4, max_obs=MAX_OBS):
        obs_data  = np.random.randn(n_days, n_lat, n_lon, max_obs, NUM_BANDS).astype(np.float32)
        obs_mask  = np.zeros((n_days, n_lat, n_lon, max_obs), dtype=bool)
        band_mask = np.zeros((n_days, n_lat, n_lon, max_obs, NUM_BANDS), dtype=bool)
        target    = np.random.uniform(50, 350, (n_days, n_lat, n_lon)).astype(np.float32)
        day_index = np.arange("2003-01-01", "2003-04-11", dtype="datetime64[D]")[:n_days]
        return obs_data, obs_mask, band_mask, target, day_index

    def test_band_mask_night(self):
        """Night observations should have visible bands masked."""
        is_night = np.zeros((10, 2, 2, 3), dtype=bool)
        is_night[0, 0, 0, 1] = True   # slot 1 is night

        bm = build_band_mask_for_night(is_night, num_bands=NUM_BANDS)
        assert bm.shape == (10, 2, 2, 3, NUM_BANDS)
        # Night slot: visible bands masked
        assert bm[0, 0, 0, 1, :7].all()
        assert not bm[0, 0, 0, 1, 7:].any()
        # Daytime slot: nothing masked
        assert not bm[0, 0, 0, 0].any()

    def test_years_to_slice_basic(self):
        """years_to_slice returns correct indices."""
        days = np.arange("2003-01-01", "2006-01-01", dtype="datetime64[D]")
        sl = years_to_slice(days, (2004, 2004))
        # 2004 is a leap year: 366 days
        selected_years = days[sl].astype("datetime64[Y]").astype(int) + 1970
        assert set(selected_years.tolist()) == {2004}

    def test_dataset_length(self):
        """Dataset length is correct for the given window and step."""
        obs_data, obs_mask, band_mask, target, day_index = self._make_arrays(n_days=50)
        cfg = DataConfig(window=WINDOW, step=1, spatial_stride=1)
        ds = ModisRadiationDataset(
            config=cfg, split="train",
            obs_data=obs_data, obs_mask=obs_mask, band_mask=band_mask,
            target=target, day_index=day_index,
        )
        # Max possible: (50 - 5 + 1) * 4 * 4 = 736 (minus any NaN windows)
        assert len(ds) <= (50 - WINDOW + 1) * 4 * 4
        assert len(ds) > 0

    def test_dataset_item_shapes(self):
        """Dataset items have the expected shapes."""
        obs_data, obs_mask, band_mask, target, day_index = self._make_arrays()
        cfg = DataConfig(window=WINDOW, step=1, spatial_stride=1)
        ds = ModisRadiationDataset(
            config=cfg, split="train",
            obs_data=obs_data, obs_mask=obs_mask, band_mask=band_mask,
            target=target, day_index=day_index,
        )
        item = ds[0]
        assert item["obs"].shape             == (WINDOW, MAX_OBS, NUM_BANDS)
        assert item["obs_padding_mask"].shape == (WINDOW, MAX_OBS)
        assert item["band_mask"].shape        == (WINDOW, MAX_OBS, NUM_BANDS)
        assert item["target"].shape           == (WINDOW,)

    def test_normalisation_stats_shape(self):
        """Normalisation stats have the correct shapes."""
        obs_data, _, _, target, _ = self._make_arrays()
        sl = slice(0, 70)
        stats = compute_normalisation_stats(obs_data, target, sl)
        assert stats["obs_mean"].shape == (NUM_BANDS,)
        assert stats["obs_std"].shape  == (NUM_BANDS,)
        assert stats["obs_std"].min() > 0


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_perfect_predictions(self):
        arr = np.random.randn(100)
        m = compute_metrics(arr, arr)
        assert m["RMSE"] == pytest.approx(0.0, abs=1e-6)
        assert m["R2"]   == pytest.approx(1.0, abs=1e-5)

    def test_mean_prediction(self):
        """Predicting the mean gives R² = 0."""
        tgt = np.random.randn(100)
        pred = np.full_like(tgt, tgt.mean())
        m = compute_metrics(pred, tgt)
        assert m["R2"] == pytest.approx(0.0, abs=1e-4)

    def test_metric_keys(self):
        pred = np.zeros(10)
        tgt  = np.ones(10)
        m = compute_metrics(pred, tgt)
        assert set(m.keys()) == {"RMSE", "MAE", "R2"}

    def test_rmse_mae_positive(self):
        pred = np.array([1.0, 2.0, 3.0])
        tgt  = np.array([2.0, 2.0, 2.0])
        m = compute_metrics(pred, tgt)
        assert m["RMSE"] > 0
        assert m["MAE"]  > 0


# ---------------------------------------------------------------------------
# SinusoidalPositionalEncoding tests
# ---------------------------------------------------------------------------

class TestSinusoidalPE:

    def test_output_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=128)
        x = torch.zeros(2, 10, 64)
        out = pe(x)
        assert out.shape == (2, 10, 64)

    def test_different_positions_differ(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=128, dropout=0.0)
        x = torch.zeros(1, 5, 64)
        out = pe(x)
        # Different time steps should have different encodings
        for i in range(5):
            for j in range(i + 1, 5):
                assert not torch.allclose(out[0, i], out[0, j])
