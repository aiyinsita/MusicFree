"""
train.py
========
End-to-end training and evaluation script for the MODIS net-radiation model.

Quick start
-----------
    python -m modis_radiation.train \\
        --data_root /path/to/processed \\
        --window 5 \\
        --epochs 100 \\
        --batch_size 32 \\
        --lr 1e-4 \\
        --output_dir ./checkpoints

The script will:
  1. Build DataLoaders for train / val / test splits.
  2. Instantiate NetRadiationModel and DailyObservationEncoder.
  3. Train with AdamW + cosine LR decay + early stopping.
  4. Evaluate on the test set and report RMSE / MAE / R².
  5. Save the best checkpoint and normalisation statistics.
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import build_dataloaders
from .model import NetRadiationModel

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and R² given flat arrays."""
    preds = preds.ravel()
    targets = targets.ravel()
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae  = float(np.mean(np.abs(preds - targets)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model: NetRadiationModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run one training or evaluation epoch.

    Returns
    -------
    mean_loss, all_preds (flat), all_targets (flat)
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0
    all_preds: list = []
    all_targets: list = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in loader:
            obs           = batch["obs"].to(device)               # (B, W, max_obs, C)
            obs_pad       = batch["obs_padding_mask"].to(device)  # (B, W, max_obs)
            band_m        = batch["band_mask"].to(device)         # (B, W, max_obs, C)
            tgt           = batch["target"].to(device)            # (B, W)
            aux           = batch.get("aux")
            if aux is not None:
                aux = aux.to(device)

            preds = model(obs, band_m, obs_pad, aux)              # (B, W)
            loss  = criterion(preds, tgt)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(tgt.detach().cpu().numpy())

    mean_loss = total_loss / max(n_batches, 1)
    return (
        mean_loss,
        np.concatenate(all_preds,   axis=0),
        np.concatenate(all_targets, axis=0),
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    log.info("Using device: %s", device)

    # ---- data ----
    log.info("Building DataLoaders from: %s", args.data_root)
    train_dl, val_dl, test_dl = build_dataloaders(
        data_root=args.data_root,
        window=args.window,
        spatial_stride=args.spatial_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    log.info(
        "Dataset sizes – train: %d  val: %d  test: %d",
        len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset),
    )

    # ---- infer aux_dim from first batch ----
    sample_batch = next(iter(train_dl))
    aux_dim = sample_batch["aux"].shape[-1] if "aux" in sample_batch else 0

    # ---- model ----
    model = NetRadiationModel(
        num_bands=args.num_bands,
        obs_hidden_dim=args.obs_hidden_dim,
        embed_dim=args.embed_dim,
        num_latents=args.num_latents,
        obs_heads=args.obs_heads,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        tf_hidden=args.tf_hidden,
        tf_heads=args.tf_heads,
        tf_layers=args.tf_layers,
        tf_ff_dim=args.tf_ff_dim,
        mlp_hidden=args.mlp_hidden,
        window=args.window,
        dropout=args.dropout,
    )
    if aux_dim > 0:
        model.set_aux_dim(aux_dim)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{n_params:,}")

    # ---- optimiser ----
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    criterion = nn.MSELoss()

    # ---- training ----
    best_val_rmse = math.inf
    patience_counter = 0
    history: list = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_preds, train_tgts = _run_epoch(
            model, train_dl, optimizer, device, criterion
        )
        val_loss, val_preds, val_tgts = _run_epoch(
            model, val_dl, None, device, criterion
        )
        scheduler.step()

        val_metrics = compute_metrics(val_preds, val_tgts)
        elapsed = time.time() - t0

        log.info(
            "Epoch %3d/%d | train_loss %.4f | val_loss %.4f | "
            "val_RMSE %.4f | val_R2 %.4f | %.1fs",
            epoch, args.epochs,
            train_loss, val_loss,
            val_metrics["RMSE"], val_metrics["R2"],
            elapsed,
        )

        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        # ---- checkpoint ----
        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            patience_counter = 0
            ckpt_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                ckpt_path,
            )
            log.info("  ✓ Saved best model (val_RMSE=%.4f)", best_val_rmse)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info("Early stopping at epoch %d.", epoch)
                break

    # ---- save training history ----
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- final test evaluation ----
    log.info("Loading best checkpoint for test evaluation …")
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_preds, test_tgts = _run_epoch(model, test_dl, None, device, criterion)
    test_metrics = compute_metrics(test_preds, test_tgts)
    log.info("Test metrics: %s", test_metrics)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MODIS net-radiation model")

    # data
    p.add_argument("--data_root",      type=str, required=True)
    p.add_argument("--window",         type=int, default=5,
                   help="Time-series window in days (3 or 5 recommended)")
    p.add_argument("--spatial_stride", type=int, default=4)
    p.add_argument("--num_workers",    type=int, default=4)

    # model architecture
    p.add_argument("--num_bands",       type=int, default=36)
    p.add_argument("--obs_hidden_dim",  type=int, default=128)
    p.add_argument("--embed_dim",       type=int, default=256)
    p.add_argument("--num_latents",     type=int, default=4)
    p.add_argument("--obs_heads",       type=int, default=4)
    p.add_argument("--lstm_hidden",     type=int, default=256)
    p.add_argument("--lstm_layers",     type=int, default=2)
    p.add_argument("--tf_hidden",       type=int, default=256)
    p.add_argument("--tf_heads",        type=int, default=4)
    p.add_argument("--tf_layers",       type=int, default=2)
    p.add_argument("--tf_ff_dim",       type=int, default=512)
    p.add_argument("--mlp_hidden",      type=int, default=128)
    p.add_argument("--dropout",         type=float, default=0.1)

    # training
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--lr_min",        type=float, default=1e-6,
                   help="Minimum learning rate for cosine annealing (eta_min)")
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--output_dir",    type=str,   default="./checkpoints")

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
