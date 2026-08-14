"""Train the DDPM model on MNIST — the single entry point for training.

Usage::

    uv run python scripts/train.py                       # defaults (configs/default.yaml)
    uv run python scripts/train.py optim.lr=5e-4          # override a key
    uv run python scripts/train.py --config configs/x.yaml train.epochs=50

The script:
  1. Loads & merges config (default file + CLI overrides).
  2. Builds the data module, model, DDPM core, Lightning module, callbacks.
  3. Saves the resolved config into the run's log dir for reproducibility.
  4. Runs ``Trainer.fit`` (auto device selection, TensorBoard logging, checkpoints).

Outputs (all gitignored):
  * ``runs/<name>/``    — TensorBoard logs
  * ``checkpoints/``    — best checkpoint (monitor val/loss) + last
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Make the package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from ddpm_lab.callbacks import SamplingCallback
from ddpm_lab.config import load_config, save_config, to_yaml
from ddpm_lab.core import DiffusionCore
from ddpm_lab.data import MNISTDataModule
from ddpm_lab.lightning_module import DDPMLightningModule
from ddpm_lab.models import build_model


def main() -> None:
    cfg = load_config()

    # Reproducibility.
    seed = int(cfg.train.get("seed", 42))
    L.seed_everything(seed, workers=True)

    # --- Paths --------------------------------------------------------------
    run_name = f"ddpm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = Path(cfg.paths.log_dir)
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save the resolved config next to the logs *and* next to the checkpoints:
    # scripts/sample.py looks for it in the checkpoint dir to rebuild the exact
    # model that was trained (crucial for non-default model configs).
    save_config(cfg, log_dir / f"{run_name}-config.yaml")
    save_config(cfg, ckpt_dir / f"{run_name}-config.yaml")
    print("Resolved config:\n", to_yaml(cfg))

    # --- Data ---------------------------------------------------------------
    dm = MNISTDataModule.from_config(cfg)
    dm.prepare_data()  # download now (single process)
    dm.setup()

    # Build a small real-image pool for coverage/mode-collapse on the CPU.
    real_imgs = torch.stack([dm.mnist_test[i][0] for i in range(int(cfg.metrics.coverage_num_real))])
    real_labels = torch.tensor([dm.mnist_test[i][1] for i in range(int(cfg.metrics.coverage_num_real))])

    # --- Model + core + Lightning module -----------------------------------
    # Cast the schedule to float32 for training (the float64 buffers were only for
    # precision of the cumprod chain; at train time float32 is fine and faster).
    core = DiffusionCore(
        num_timesteps=int(cfg.diffusion.num_timesteps),
        beta_start=float(cfg.diffusion.beta_start),
        beta_end=float(cfg.diffusion.beta_end),
        schedule=str(cfg.diffusion.schedule),
    ).float()
    model = build_model(cfg)
    pl_module = DDPMLightningModule(model, core, cfg)

    # --- Callbacks ----------------------------------------------------------
    sampling_cb = SamplingCallback(
        core=core,
        cfg=cfg,
        image_shape=dm.shape,
        real_pool=(real_imgs, real_labels),
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{run_name}-best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # --- Logger -------------------------------------------------------------
    logger = TensorBoardLogger(save_dir=str(log_dir), name=run_name, version="")

    # --- Trainer ------------------------------------------------------------
    # Optional smoke-test knobs: train.limit_train_batches / limit_val_batches.
    # Defaults: 1.0 = use the full dataset. Set to a small int for quick checks.
    limit_train = cfg.train.get("limit_train_batches", 1.0)
    limit_val = cfg.train.get("limit_val_batches", 1.0)

    precision = str(cfg.train.get("precision", "16-mixed"))
    if not torch.cuda.is_available() and precision != "32-true":
        # Mixed precision on CPU is emulated and painfully slow — auto-fallback.
        print(f"[info] no CUDA detected: switching precision {precision} -> 32-true")
        precision = "32-true"

    trainer = L.Trainer(
        max_epochs=int(cfg.train.epochs),
        max_steps=int(cfg.train.get("max_steps", -1)),
        accelerator="auto",
        devices="auto",
        precision=precision,
        gradient_clip_val=float(cfg.train.get("gradient_clip_val", 0.0)) or None,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        logger=logger,
        callbacks=[sampling_cb, checkpoint_cb],
        log_every_n_steps=50,
    )

    print(f"\nRun name: {run_name}")
    print(f"Logs:     {log_dir}/{run_name}")
    print(f"Best ckpt will be at: {ckpt_dir}/{run_name}-best.ckpt")
    print("Start TensorBoard with:  tensorboard --logdir", log_dir, "\n")

    trainer.fit(pl_module, datamodule=dm)

    print(f"\nDone. Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val/loss: {checkpoint_cb.best_model_score:.4f}" if checkpoint_cb.best_model_score else "")


if __name__ == "__main__":
    main()
