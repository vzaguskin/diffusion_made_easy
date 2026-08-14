"""Stochastic DDPM sampler — theory.md §17 «Алгоритм генерации (inference)».

This is the *Markov* reverse process: at every step we add fresh noise ``z ~ N(0,I)``,
except on the last step (``t = 1``) where we return the mean ``mu`` directly to
avoid a grainy final image (theory.md §17, "Зачем на последнем шаге добавляется
случайный шум?").

Algorithm (theory.md §17):
    1. x_T ~ N(0, I)
    2. for t = T, T-1, ..., 1:
         a. eps_pred = eps_theta(x_t, t)
         b. mu = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_pred)
         c. if t > 1: x_{t-1} = mu + sigma_t * z,  z ~ N(0, I)
            else:     x_0 = mu
    3. return x_0

The variance ``sigma_t`` is taken from the schedule (we use ``beta_t`` here, the
common "fixed small" choice; theory.md §15 says the variance is fixed, not
predicted by the network).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..core import DiffusionCore
from ._util import randn_with_generator


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    core: DiffusionCore,
    shape: tuple[int, ...],
    num_steps: Optional[int] = None,
    *,
    sigma: str = "posterior",
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Generate images with the stochastic DDPM sampler (theory.md §17).

    Parameters
    ----------
    model : nn.Module
        ε_θ network with signature ``model(x, t) -> eps``.
    core : DiffusionCore
        Owns the schedule. ``num_steps`` defaults to ``core.num_timesteps`` (the
        sampler walks all T steps, as in theory.md §17).
    shape : tuple
        Batch shape, e.g. ``(64, 1, 28, 28)``.
    sigma : {"posterior", "beta"}
        Choice of the fixed reverse-step variance σ_t (theory.md §15):
        ``"posterior"`` (default) uses β̃_t from the §13 derivation and gives
        better samples; ``"beta"`` is the simpler "small" choice.
    generator : torch.Generator, optional
        For reproducible sampling (fixes the starting noise and per-step noise).
    device : optional
        Defaults to ``core``'s device.

    Returns
    -------
    Tensor
        Generated images of shape ``shape``, in the same range the model was
        trained on (normalized MNIST: zero-mean, unit-variance).
    """
    if num_steps is None:
        num_steps = core.num_timesteps
    if num_steps != core.num_timesteps:
        raise ValueError(
            "The DDPM sampler is Markov and must walk all T steps. For fewer steps "
            f"use the DDIM sampler. (got num_steps={num_steps}, T={core.num_timesteps})"
        )
    device = device or core.betas.device

    model.eval()
    # Work in the model's dtype (float32) — the schedule is stored in float64 for
    # precision but the network runs in float32.
    work_dtype = next(model.parameters()).dtype

    # Step 1: x_T ~ N(0, I)
    x = randn_with_generator(shape, generator, device, work_dtype)

    # --- Reverse-step variance sigma_t ---------------------------------------
    # theory.md §15 fixes the variance (it is not learned); §13 derives the
    # *posterior* variance of q(x_{t-1} | x_t, x_0):
    #     beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    # The DDPM paper compares two choices:
    #   * sigma = "beta"     (the "small" choice)  — simpler, slightly worse;
    #   * sigma = "posterior" (the "large" choice) — matches the derivation of
    #     §13 and empirically gives better samples. This is our default.
    alphas = core.alphas.to(device=device, dtype=work_dtype)
    alphas_cumprod = core.alphas_cumprod.to(device=device, dtype=work_dtype)
    if sigma == "posterior":
        # alpha_bar_{t-1} for each t, with the convention alpha_bar_{-1} = 1.
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device, dtype=work_dtype), alphas_cumprod[:-1]]
        )
        posterior_variance = core.betas.to(device=device, dtype=work_dtype) * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod).clamp_min(1e-20)
        # At t = 0 the posterior variance is 0 — consistent with the last step
        # adding no noise anyway.
        sigmas = torch.sqrt(posterior_variance.clamp_min(0.0))
    elif sigma == "beta":
        sigmas = torch.sqrt(core.betas.clamp_min(0.0)).to(device=device, dtype=work_dtype)
    else:
        raise ValueError(f"sigma must be 'posterior' or 'beta', got {sigma!r}")

    # Step 2: walk t = T-1, T-2, ..., 0 (0-based indices; theory is 1-based).
    for t in reversed(range(core.num_timesteps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        # 2a: predict the noise
        eps_pred = model(x, t_batch)
        # 2b: mean of the reverse step (theory.md §17, eq. for mu)
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
        )
        # 2c: add noise unless this is the last step (t == 0 -> produces x_0)
        if t > 0:
            z = randn_with_generator(shape, generator, device, x.dtype)
            x = mean + sigmas[t] * z
        else:
            x = mean
    return x
