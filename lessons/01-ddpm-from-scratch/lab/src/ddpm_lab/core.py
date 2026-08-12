"""DDPM core: the math of the forward process and the training loss.

This module is **deliberately independent** of the neural network used to predict
the noise ``eps``. It owns the noise schedule and three operations:

* :meth:`DiffusionCore.q_sample` — the closed-form forward step ("прыжок через все
  шаги сразу", theory.md §4).
* :meth:`DiffusionCore.predict_start_from_noise` — recover ``x0`` from ``x_t`` and
  the noise (theory.md §14, "репараметризация через шум").
* :meth:`DiffusionCore.compute_loss` — the unweighted MSE loss
  ``E[||eps - eps_theta||^2]`` (theory.md §15, "Финальная функция потерь").

Theory references in comments point at ``theory.md`` (the lesson's theory file).

Why float64 buffers?
    The cumulative product ``alphas_cumprod`` is a long chain of multiplications of
    numbers close to 1. In float32 this loses precision badly for large ``t`` (the
    product drifts). Computing the schedule in float64 and only casting to the
    working dtype at *use* time keeps the variance-preserving property accurate.
    This is a standard DDPM implementation detail.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .schedules import linear_beta_schedule


class DiffusionCore(nn.Module):
    """Owns the noise schedule and the closed-form forward/loss operations.

    All schedule tensors are registered as buffers, so they automatically follow
    the module's device (``.to("cuda")`` moves them too) and are *not* trainable
    parameters.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ) -> None:
        super().__init__()
        if schedule != "linear":
            # The lesson's theory only covers the linear schedule (theory.md §3).
            # Other schedules (cosine, etc.) are out of scope for lesson 1.
            raise NotImplementedError(
                f"schedule='{schedule}' is not implemented; lesson 1 only supports 'linear' "
                "(see theory.md §3)."
            )

        self.num_timesteps = num_timesteps

        # --- The raw schedule: beta_t for t = 1..T (theory.md §3) ---------------
        betas = linear_beta_schedule(beta_start, beta_end, num_timesteps)  # float64, [T]

        # --- alpha_t = 1 - beta_t --------------------------------------------
        alphas = 1.0 - betas

        # --- alpha_bar_t = prod_{s=1..t} alpha_s  (cumulative product) --------
        # theory.md §4 "Трюк №1": this is what lets us jump straight from x0 to x_t.
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # --- Convenience coefficients used by q_sample and the samplers -------
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Register everything as buffers (non-trainable, device-aware).
        # We keep the high-precision (float64) versions and expose a helper to cast
        # at use time, so the cumprod chain stays numerically accurate.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    # ------------------------------------------------------------------ helpers
    def _gather(self, vals: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Index ``vals`` at positions ``t`` and reshape to broadcast over ``x``.

        ``vals`` has shape ``[T]``; ``t`` has shape ``[B]``; we return ``[B, 1, 1, ...]``
        matching ``x``'s number of dimensions so it broadcasts cleanly.
        """
        # Clamp t into [0, T-1] for indexing safety (t is 1-based in the theory,
        # but we store arrays 0-based; callers pass 0-based indices here).
        b = x.shape[0]
        out = vals.gather(-1, t.long())
        return out.reshape(b, *([1] * (x.ndim - 1))).to(x.dtype)

    # ------------------------------------------------------------- q_sample
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward process in closed form — "прыжок через все шаги" (theory.md §4).

        ``x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps``

        Parameters
        ----------
        x0 : Tensor[B, C, H, W]
            Clean images.
        t : Tensor[B]
            Integer timestep index in ``[0, T-1]`` (0-based). Each sample in the
            batch gets its own ``t`` — this matches theory.md §16 (training
            algorithm, step 2: "t <- случайное число из {1,...,T}").
        noise : Tensor, optional
            Optional pre-sampled ``eps ~ N(0, I)``. If ``None``, we sample it.
            Passing it explicitly makes the call deterministic (useful for tests
            and for fixed-noise visualization).

        Returns
        -------
        Tensor[B, C, H, W]
            Noisy images ``x_t``.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # theory.md §4: x_t = sqrt(ᾱ_t) · x0 + sqrt(1-ᾱ_t) · ε
        sqrt_ac = self._gather(self.sqrt_alphas_cumprod, t, x0)
        sqrt_omac = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0)
        return sqrt_ac * x0 + sqrt_omac * noise

    # ----------------------------------------------- predict_start_from_noise
    def predict_start_from_noise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x0 from x_t and the noise — theory.md §14 "репараметризация".

        ``x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)``

        This is the inverse of :meth:`q_sample` and is used by the samplers to
        turn the network's predicted noise into an estimate of the clean image at
        each denoising step.
        """
        sqrt_ac = self._gather(self.sqrt_alphas_cumprod, t, xt)
        sqrt_omac = self._gather(self.sqrt_one_minus_alphas_cumprod, t, xt)
        return (xt - sqrt_omac * noise) / sqrt_ac.clamp_min(1e-8)

    # --------------------------------------------------------- compute_loss
    @staticmethod
    def compute_loss(eps_pred: torch.Tensor, eps_target: torch.Tensor) -> torch.Tensor:
        """Unweighted MSE loss on the predicted noise (theory.md §15).

        ``L = mean( (eps_pred - eps_target)^2 )``

        theory.md §15 "Отбрасывание веса": the KL term has a per-``t`` weight, but
        DDPM empirically sets it to 1 for all ``t`` (and it works *better*). So the
        loss is just plain MSE between predicted and true noise — no per-timestep
        weighting, no reduction over spatial dims other than mean.
        """
        # mean over all elements (batch, channels, height, width)
        return torch.mean((eps_pred - eps_target) ** 2)
