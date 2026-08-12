"""Lightning callbacks: fixed-noise sampling + TensorBoard image logging.

The key idea — **fixed noise across epochs** (theory.md §17/§18):
    We keep one starting noise ``x_T`` and reuse it every epoch. As training
    progresses, the *same* seed becomes a clearer digit. This makes progress
    visible at a glance in TensorBoard.

We also optionally log a DDPM-vs-DDIM comparison from the same ``x_T`` so the
student can see the difference between stochastic and deterministic sampling
(theory.md §17 vs §18).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .core import DiffusionCore
from .metrics import coverage_and_mode_collapse
from .samplers import build_sampler


def _denormalize_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Map normalized (zero-mean/unit-var) MNIST images back to [0, 255] uint8.

    The model trains on ``Normalize((0.1307,), (0.3081,))`` data, so we invert it
    for visualization. Clips to [0, 1] before scaling to uint8.
    """
    mean = torch.tensor(0.1307, device=x.device, dtype=x.dtype)
    std = torch.tensor(0.3081, device=x.device, dtype=x.dtype)
    x = x * std + mean
    return x.clamp(0.0, 1.0)


def _grid(x: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Build a single-image grid (C=1 -> replicated to 3 channels for TB)."""
    g = make_grid(_denormalize_to_uint8(x), nrow=nrow)
    if g.shape[0] == 1:
        g = g.repeat(3, 1, 1)  # TensorBoard wants 3-channel images
    return g


class SamplingCallback(Callback):
    """Generates samples with a fixed seed each ``sample_freq`` epochs."""

    def __init__(
        self,
        core: DiffusionCore,
        cfg: Any,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        real_pool: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.core = core
        self.cfg = cfg
        self.image_shape = image_shape
        self.num_samples = int(cfg.callbacks.num_samples)
        self.sample_freq = int(cfg.callbacks.sample_freq)
        self.sampler_name = cfg.callbacks.sampler
        self.num_sample_steps = int(cfg.callbacks.num_sample_steps)
        self.eta = float(cfg.callbacks.eta)
        self.compare_samplers = bool(cfg.callbacks.compare_samplers)
        # A pool of real images + labels for coverage/mode-collapse. Set by train.py.
        self.real_pool = real_pool  # (images, labels) on CPU
        # We make sampling deterministic across epochs by seeding the sampler's
        # generator with a fixed value, so the *same* x_T is reused every time.
        self._fixed_seed = 12345
        self._tb_writer: SummaryWriter | None = None

    def _get_writer(self, trainer: Trainer) -> SummaryWriter | None:
        # Lightning's TensorBoard logger exposes a SummaryWriter.
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return None
        try:
            return logger.experiment  # SummaryLogger.experiment is a SummaryWriter
        except Exception:
            return None

    @torch.no_grad()
    def _sample(
        self, pl_module: LightningModule, sampler_name: str
    ) -> torch.Tensor:
        """Run a sampler with a fixed seed (same x_T every epoch → visible progress)."""
        device = pl_module.device
        model = pl_module.model.to(device).eval()
        sampler = build_sampler(sampler_name)
        gen = torch.Generator(device="cpu").manual_seed(self._fixed_seed)
        kwargs = dict(generator=gen, device=device)
        if sampler_name == "ddim":
            kwargs["num_steps"] = self.num_sample_steps
            kwargs["eta"] = self.eta
        out = sampler(model, self.core, (self.num_samples, *self.image_shape), **kwargs)
        return out

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        if epoch % self.sample_freq != 0:
            return
        writer = self._get_writer(trainer)
        if writer is None:
            return

        try:
            # Sample in float32 to avoid dtype friction with the model.
            samples = self._sample(pl_module, self.sampler_name).float()
        except Exception as e:  # don't let a sampling bug kill training
            print(f"[SamplingCallback] sampling failed at epoch {epoch}: {e}")
            return

        step = trainer.global_step
        grid = _grid(samples, nrow=int(math.isqrt(samples.shape[0]) or 8))
        writer.add_image(f"samples/{self.sampler_name}", grid, step)

        # Optional DDPM-vs-DDIM comparison from the same starting seed.
        # (Both samplers re-seed internally with self._fixed_seed, so DDIM's x_T is
        # identical across runs; DDPM's x_T is also identical, but its per-step
        # noise makes the final images differ — exactly the contrast we want to
        # show, theory.md §17 vs §18.)
        if self.compare_samplers:
            try:
                ddim_s = self._sample(pl_module, "ddim").float()
                ddpm_s = self._sample(pl_module, "ddpm").float()
                writer.add_image("compare/ddim", _grid(ddim_s, nrow=int(math.isqrt(ddim_s.shape[0]) or 8)), step)
                writer.add_image("compare/ddpm", _grid(ddpm_s, nrow=int(math.isqrt(ddpm_s.shape[0]) or 8)), step)
            except Exception as e:
                print(f"[SamplingCallback] comparison sampling failed: {e}")

        # Coverage / mode-collapse against the real pool (if provided).
        if self.real_pool is not None:
            real_imgs, real_labels = self.real_pool
            real_imgs = real_imgs[: int(self.cfg.metrics.coverage_num_real)].to(samples.device)
            real_labels = real_labels[: real_imgs.shape[0]].to(samples.device)
            try:
                cov, hist = coverage_and_mode_collapse(samples, real_imgs, real_labels)
                writer.add_scalar("val/coverage", cov, step)
                writer.add_histogram("val/nn_class_distribution", hist.float(), step)
            except Exception as e:
                print(f"[SamplingCallback] metrics failed: {e}")
