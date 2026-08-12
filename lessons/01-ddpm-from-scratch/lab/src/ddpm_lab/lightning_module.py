"""PyTorch Lightning module wrapping the model, the DDPM core, and the loss.

This is where the training algorithm of theory.md §16 lands in code::

    training_step:
        t   <- U{1..T}              (one per sample)
        eps <- N(0, I)
        x_t <- q_sample(x0, t, eps)  # theory.md §4, the "jump"
        eps_pred <- model(x_t, t)
        loss <- mean( (eps_pred - eps)^2 )   # theory.md §15, unweighted

Validation additionally computes:
    * val/loss                       — scalar
    * val/loss_bucket_{i}            — loss bucketed by t (metrics.loss_by_t_bucket)
    * val/coverage, val/nn_class_*   — only when samples are generated this epoch
                                       (the SamplingCallback attaches a real-image pool)
"""

from __future__ import annotations

from typing import Any

import torch
from lightning import LightningModule

from .core import DiffusionCore
from .metrics import loss_by_t_bucket


class DDPMLightningModule(LightningModule):
    """Holds model + core; implements training/val steps per theory.md §16."""

    def __init__(self, model: torch.nn.Module, core: DiffusionCore, cfg: Any) -> None:
        super().__init__()
        self.model = model
        self.core = core
        self.cfg = cfg
        self.num_t_buckets = int(getattr(cfg.metrics, "num_t_buckets", 10))
        # Don't try to save state of the whole module dict; Lightning saves
        # self.model via the automatic optimization checkpointing.
        self.automatic_optimization = True

    # ------------------------------------------------------------- training
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x0, _ = batch  # labels are unused (unconditional generation)
        b = x0.shape[0]

        # theory.md §16 steps 2-4: sample t, eps, compute x_t.
        t = torch.randint(0, self.core.num_timesteps, (b,), device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.core.q_sample(x0, t, eps)

        # Step 5: predict the noise.
        eps_pred = self.model(xt, t)

        # Step 6: unweighted MSE (theory.md §15).
        loss = self.core.compute_loss(eps_pred, eps)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=b)
        return loss

    # ----------------------------------------------------------- validation
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x0, _ = batch
        b = x0.shape[0]
        t = torch.randint(0, self.core.num_timesteps, (b,), device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.core.q_sample(x0, t, eps)
        eps_pred = self.model(xt, t)
        loss = self.core.compute_loss(eps_pred, eps)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=b)

        # Bucketed loss — shows where the model struggles across noise levels.
        buckets = loss_by_t_bucket(eps_pred, eps, t, self.core.num_timesteps, self.num_t_buckets)
        for i, v in enumerate(buckets.tolist()):
            if v == v:  # skip NaN (empty bucket)
                self.log(f"val/loss_bucket_{i}", v, on_epoch=True, batch_size=b)
        return loss

    # --------------------------------------------------------------- optim
    def configure_optimizers(self):
        oc = self.cfg.optim
        if oc.type != "adam":
            raise NotImplementedError(f"optimizer type='{oc.type}' not supported; use 'adam'.")
        betas = tuple(oc.get("betas", [0.9, 0.999]))
        return torch.optim.Adam(
            self.model.parameters(),
            lr=float(oc.lr),
            weight_decay=float(oc.get("weight_decay", 0.0)),
            betas=betas,
        )
