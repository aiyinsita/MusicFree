"""
dataset.py
==========
Dataset construction and time-based train / validation / test splitting for the
MODIS sea-surface net-radiation model.

Data layout assumptions
-----------------------
The expected on-disk layout is a directory tree with one NetCDF / HDF file per
day (or a pre-processed NumPy archive):

    data_root/
      modis/
        YYYY/DOY/          ← per-swath observation files
      gldas_target/
        YYYY/              ← GLDAS/GHOSE net radiation daily composites
      aux/
        YYYY/              ← auxiliary static / daily fields

For large-scale use we recommend pre-processing the raw swath data into a
single NumPy memmap or Zarr array indexed by (year, doy, lat_idx, lon_idx,
obs_slot, band).

Split strategy
--------------
Sea-surface radiation has strong annual cycles AND inter-annual variability
(ENSO, PDO…).  A purely random split would leak future information into
training and inflate validation scores.

We use a **time-ordered split**:

  ┌─────────────┬──────────────────────────────────────────────────┐
  │  Split      │  Years (20 yr archive: 2003–2022)                │
  ├─────────────┼──────────────────────────────────────────────────┤
  │  Train      │  2003–2016  (14 years, ~70 %)                    │
  │  Validation │  2017–2018  ( 2 years, ~10 %)                    │
  │  Test       │  2019–2022  ( 4 years, ~20 %)                    │
  └─────────────┴──────────────────────────────────────────────────┘

Rationale
---------
* Training ends at 2016 so that the validation set (2017-2018) is temporally
  *after* training data.  This mimics operational deployment.
* Test set covers 4 recent years including the stronger ENSO event of 2019.
* If your archive starts at 2000 (full Terra record), shift all years by -3:
  Train 2000-2013, Val 2014-2015, Test 2016-2019.

Usage
-----
    from modis_radiation.dataset import build_dataloaders

    train_dl, val_dl, test_dl = build_dataloaders(
        data_root="path/to/data",
        window=5,          # days per sample
        spatial_stride=4,  # sub-sample grid to reduce memory
        batch_size=32,
    )
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Default temporal split
# ---------------------------------------------------------------------------

_NORM_EPS: float = 1e-6
"""Small constant added to standard deviations to prevent division by zero."""

DEFAULT_SPLIT: Dict[str, Tuple[int, int]] = {
    "train": (2003, 2016),   # inclusive on both ends
    "val":   (2017, 2018),
    "test":  (2019, 2022),
}

# MODIS bands 1-7 are visible / NIR – absent in MYD nighttime passes.
MODIS_VISIBLE_BANDS: List[int] = list(range(7))  # indices 0-6 (bands 1-7)


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for dataset construction."""

    data_root: str = "data"
    """Root directory of the pre-processed data archive."""

    num_bands: int = 36
    """Number of MODIS spectral bands per observation."""

    max_obs_per_day: int = 6
    """Maximum number of within-day swath observations (pad to this length)."""

    window: int = 5
    """Number of consecutive days per training sample."""

    step: int = 1
    """Slide step (days) between consecutive windows in the training set.
    Use step=window for non-overlapping windows."""

    spatial_stride: int = 1
    """Sub-sample grid cells to reduce dataset size.  stride=4 keeps every
    4th lat/lon cell."""

    split: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: dict(DEFAULT_SPLIT)
    )
    """Year ranges (inclusive) for each split."""

    include_myd: bool = True
    """Whether to include Aqua (MYD) observations alongside Terra (MOD)."""

    aux_features: List[str] = field(default_factory=lambda: [
        "sza_mean",    # mean solar zenith angle of the day
        "doy_sin",     # sin(2π * DOY / 365)
        "doy_cos",     # cos(2π * DOY / 365)
        "lat",         # latitude (normalised)
        "lon",         # longitude (normalised)
    ])
    """Per-day auxiliary scalar features appended after the daily embedding."""


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ModisRadiationDataset(Dataset):
    """PyTorch Dataset for MODIS TOA → net-radiation sequences.

    Each sample is a sliding window of ``config.window`` consecutive days for
    a single grid cell.

    Parameters
    ----------
    config : DataConfig
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    obs_data : np.ndarray, shape (n_days, n_lat, n_lon, max_obs, num_bands)
        Pre-loaded or memory-mapped TOA observation array.  Padded slots should
        be filled with 0.0 and marked in ``obs_mask``.
    obs_mask : np.ndarray of bool, shape (n_days, n_lat, n_lon, max_obs)
        True for *padded* (non-existent) observation slots.
    band_mask : np.ndarray of bool, shape (n_days, n_lat, n_lon, max_obs, num_bands)
        True where a band value is absent (e.g. visible bands at night).
    target : np.ndarray, shape (n_days, n_lat, n_lon)
        Daily net radiation target (W/m²) from GLDAS/GHOSE.
    day_index : np.ndarray of int, shape (n_days,)
        Absolute day index (e.g. days since 2000-01-01) for each time step.
    aux : np.ndarray, shape (n_days, n_lat, n_lon, n_aux), optional
        Per-day auxiliary features.
    """

    def __init__(
        self,
        config: DataConfig,
        split: str,
        obs_data: np.ndarray,
        obs_mask: np.ndarray,
        band_mask: np.ndarray,
        target: np.ndarray,
        day_index: np.ndarray,
        aux: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.obs_data = obs_data
        self.obs_mask = obs_mask
        self.band_mask = band_mask
        self.target = target
        self.day_index = day_index
        self.aux = aux

        n_days, n_lat, n_lon = target.shape
        W = config.window
        stride = config.spatial_stride
        step = config.step

        # ---- build list of (start_day_idx, lat_idx, lon_idx) tuples ----
        self.samples: List[Tuple[int, int, int]] = []
        for t in range(0, n_days - W + 1, step):
            for i in range(0, n_lat, stride):
                for j in range(0, n_lon, stride):
                    # Drop windows that contain any NaN target
                    if np.any(np.isnan(target[t: t + W, i, j])):
                        continue
                    self.samples.append((t, i, j))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t, i, j = self.samples[idx]
        W = self.config.window

        # ---- observations (W, max_obs, num_bands) ----
        obs = torch.from_numpy(
            self.obs_data[t: t + W, i, j].astype(np.float32)
        )
        obs_pad = torch.from_numpy(
            self.obs_mask[t: t + W, i, j].astype(bool)
        )
        band_m = torch.from_numpy(
            self.band_mask[t: t + W, i, j].astype(bool)
        )

        # ---- target (W,) ----
        tgt = torch.from_numpy(
            self.target[t: t + W, i, j].astype(np.float32)
        )

        item: Dict[str, torch.Tensor] = {
            "obs": obs,               # (W, max_obs, num_bands)
            "obs_padding_mask": obs_pad,  # (W, max_obs)  – True = padded slot
            "band_mask": band_m,      # (W, max_obs, num_bands)
            "target": tgt,            # (W,)
        }

        # ---- auxiliary features ----
        if self.aux is not None:
            item["aux"] = torch.from_numpy(
                self.aux[t: t + W, i, j].astype(np.float32)
            )  # (W, n_aux)

        return item


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------

def compute_normalisation_stats(
    obs_data: np.ndarray,
    target: np.ndarray,
    train_time_slice: slice,
) -> Dict[str, np.ndarray]:
    """Compute channel-wise mean and std from the training split only.

    Parameters
    ----------
    obs_data : (n_days, n_lat, n_lon, max_obs, num_bands)
    target   : (n_days, n_lat, n_lon)
    train_time_slice : slice covering training days

    Returns
    -------
    dict with keys "obs_mean", "obs_std", "tgt_mean", "tgt_std"
    """
    train_obs = obs_data[train_time_slice]          # (T_train, …, num_bands)
    valid = train_obs[train_obs != 0.0]             # rough but sufficient
    obs_mean = np.nanmean(train_obs, axis=(0, 1, 2, 3))  # (num_bands,)
    obs_std = np.nanstd(train_obs, axis=(0, 1, 2, 3)) + _NORM_EPS

    train_tgt = target[train_time_slice]
    tgt_mean = float(np.nanmean(train_tgt))
    tgt_std = float(np.nanstd(train_tgt)) + _NORM_EPS

    return {
        "obs_mean": obs_mean.astype(np.float32),
        "obs_std": obs_std.astype(np.float32),
        "tgt_mean": np.float32(tgt_mean),
        "tgt_std": np.float32(tgt_std),
    }


# ---------------------------------------------------------------------------
# Band mask construction
# ---------------------------------------------------------------------------

def build_band_mask_for_night(
    is_night: np.ndarray,
    num_bands: int = 36,
    visible_band_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Create a band_mask array marking visible bands absent for night passes.

    Parameters
    ----------
    is_night : bool array, shape (n_days, n_lat, n_lon, max_obs)
        True for Aqua/MYD night-time observation slots.
    num_bands : int
    visible_band_indices : list[int], optional
        Indices of visible/NIR bands to mask at night.
        Defaults to MODIS_VISIBLE_BANDS (bands 1-7, indices 0-6).

    Returns
    -------
    np.ndarray of bool, shape (n_days, n_lat, n_lon, max_obs, num_bands)
    """
    if visible_band_indices is None:
        visible_band_indices = MODIS_VISIBLE_BANDS

    # Start with all False (no bands masked)
    mask = np.zeros((*is_night.shape, num_bands), dtype=bool)
    # For night observations, mask visible bands
    mask[..., visible_band_indices] = is_night[..., np.newaxis]
    return mask


# ---------------------------------------------------------------------------
# Helper: year-range → time slice
# ---------------------------------------------------------------------------

def years_to_slice(
    day_index: np.ndarray,
    year_range: Tuple[int, int],
) -> slice:
    """Convert an inclusive year range to a slice into the day_index array.

    Parameters
    ----------
    day_index : np.ndarray of datetime64[D] or int (days-since-epoch)
        One entry per time step in the dataset.
    year_range : (start_year, end_year)  — both inclusive

    Returns
    -------
    slice
    """
    years = day_index.astype("datetime64[Y]").astype(int) + 1970
    start = int(np.searchsorted(years, year_range[0], side="left"))
    end = int(np.searchsorted(years, year_range[1], side="right"))
    return slice(start, end)


# ---------------------------------------------------------------------------
# High-level factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: str,
    window: int = 5,
    spatial_stride: int = 4,
    batch_size: int = 32,
    num_workers: int = 4,
    split: Optional[Dict[str, Tuple[int, int]]] = None,
    include_myd: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load pre-processed data and return DataLoaders for all three splits.

    This factory function expects the following files under *data_root*:

    ``obs_data.npy``        – float32 (n_days, n_lat, n_lon, max_obs, num_bands)
    ``obs_mask.npy``        – bool    (n_days, n_lat, n_lon, max_obs)
    ``is_night.npy``        – bool    (n_days, n_lat, n_lon, max_obs)
    ``target.npy``          – float32 (n_days, n_lat, n_lon)  [W/m²]
    ``day_index.npy``       – datetime64[D] (n_days,)
    ``aux.npy``             – float32 (n_days, n_lat, n_lon, n_aux)  [optional]

    Parameters
    ----------
    data_root : str
    window : int
    spatial_stride : int
    batch_size : int
    num_workers : int
    split : dict, optional
        Custom year ranges.  Keys: "train", "val", "test".
    include_myd : bool
    pin_memory : bool

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    if split is None:
        split = DEFAULT_SPLIT

    cfg = DataConfig(
        data_root=data_root,
        window=window,
        spatial_stride=spatial_stride,
        split=split,
        include_myd=include_myd,
    )

    # ---- load arrays ----
    obs_data  = np.load(os.path.join(data_root, "obs_data.npy"),  mmap_mode="r")
    obs_mask  = np.load(os.path.join(data_root, "obs_mask.npy"),  mmap_mode="r")
    is_night  = np.load(os.path.join(data_root, "is_night.npy"),  mmap_mode="r")
    target    = np.load(os.path.join(data_root, "target.npy"),    mmap_mode="r")
    day_index = np.load(os.path.join(data_root, "day_index.npy"))

    aux_path = os.path.join(data_root, "aux.npy")
    aux = np.load(aux_path, mmap_mode="r") if os.path.exists(aux_path) else None

    # ---- build band mask ----
    num_bands = obs_data.shape[-1]
    band_mask = build_band_mask_for_night(is_night, num_bands=num_bands)

    # ---- normalisation (fit on training data only) ----
    train_slice = years_to_slice(day_index, split["train"])
    stats = compute_normalisation_stats(obs_data, target, train_slice)

    # Apply z-score normalisation in-place on a writable copy
    obs_norm = (obs_data.astype(np.float32) - stats["obs_mean"]) / stats["obs_std"]
    tgt_norm  = (target.astype(np.float32)  - stats["tgt_mean"]) / stats["tgt_std"]

    # ---- build datasets ----
    def _make_ds(split_key: str, is_train: bool) -> ModisRadiationDataset:
        sl = years_to_slice(day_index, split[split_key])
        step = cfg.step if is_train else window
        ds_cfg = DataConfig(
            data_root=data_root,
            num_bands=num_bands,
            max_obs_per_day=obs_data.shape[3],
            window=window,
            step=step,
            spatial_stride=spatial_stride if is_train else 1,
            split=split,
            include_myd=include_myd,
        )
        return ModisRadiationDataset(
            config=ds_cfg,
            split=split_key,
            obs_data=obs_norm[sl],
            obs_mask=obs_mask[sl],
            band_mask=band_mask[sl],
            target=tgt_norm[sl],
            day_index=day_index[sl],
            aux=aux[sl] if aux is not None else None,
        )

    train_ds = _make_ds("train", is_train=True)
    val_ds   = _make_ds("val",   is_train=False)
    test_ds  = _make_ds("test",  is_train=False)

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle,
        )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)
