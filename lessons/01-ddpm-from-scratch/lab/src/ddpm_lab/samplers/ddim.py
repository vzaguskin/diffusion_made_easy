"""Deterministic DDIM sampler — theory.md §18 «DDIM: ускоряем генерацию в 10–50 раз».

DDIM builds a *different* reverse process with the same marginals as DDPM, so the
same trained ε_θ works without retraining (theory.md §18 "Почему одну и ту же
модель можно использовать?"). Crucially, when ``eta = 0`` the process is fully
deterministic, which lets us take big steps and skip timesteps — a 25–50 step
DDIM sample is ~20–40× faster than a 1000-step DDPM sample with similar quality.

Per-step recipe (theory.md §18 "Как DDIM делает шаг без шума"):
    1. Predict x0 from the current x_{tau_i}:
           x0_hat = (x_{tau_i} - sqrt(1-ᾱ_{tau_i}) * eps_theta) / sqrt(ᾱ_{tau_i})
    2. Step toward less noise using x0_hat and the *predicted* eps:
           x_{tau_{i-1}} = sqrt(ᾱ_{tau_{i-1}}) * x0_hat
                         + sqrt(1 - ᾱ_{tau_{i-1}}) * eps_theta

The general η > 0 form adds a stochastic term whose variance is controlled by η
(η = 0 reproduces the deterministic equations above; η = 1 approaches DDPM).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..core import DiffusionCore
from ._util import randn_with_generator


def _build_timestep_subset(num_timesteps: int, num_steps: int) -> torch.Tensor:
    """Pick ``num_steps`` evenly spaced timesteps for the strided DDIM walk.

    Standard schedule: ``linspace(0, T-1, num_steps)`` — i.e. it *includes*
    ``t = T-1`` (so the starting noise ``x_T ~ N(0, I)`` matches the first
    network call) and ends at ``t = 0`` (the final step produces ``x_0``).
    Returned in decreasing order, deduplicated.
    """
    if num_steps < 1 or num_steps > num_timesteps:
        raise ValueError(f"num_steps must be in [1, {num_timesteps}], got {num_steps}")
    ts = torch.linspace(0, num_timesteps - 1, num_steps).round().long().flip(0)
    ts = torch.unique(ts, sorted=False)  # dedupe (possible for tiny num_steps)
    return ts.sort(descending=True).values


@torch.no_grad()
def sample(
    model: torch.nn.Module,
    core: DiffusionCore,
    shape: tuple[int, ...],
    num_steps: Optional[int] = None,
    *,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Generate images with the DDIM sampler (theory.md §18).

    Parameters
    ----------
    model : nn.Module
        ε_θ network with signature ``model(x, t) -> eps``.
    core : DiffusionCore
        Owns the schedule.
    shape : tuple
        Batch shape, e.g. ``(64, 1, 28, 28)``.
    num_steps : int, optional
        Number of DDIM steps. Defaults to ``core.num_timesteps``. Smaller values
        (e.g. 25–50) give faster sampling with modest quality loss.
    eta : float
        Stochasticity. ``0.0`` (default) = fully deterministic; ``1.0`` ≈ DDPM.
    generator : torch.Generator, optional
        For reproducible starting noise (and stochastic part when eta > 0).
    device : optional
        Defaults to ``core``'s device.

    Returns
    -------
    Tensor
        Generated images of shape ``shape``.
    """
    num_steps = num_steps or core.num_timesteps
    device = device or core.betas.device
    model.eval()
    # Work in the model's dtype (float32) — the schedule is float64 for precision.
    work_dtype = next(model.parameters()).dtype

    # Build the strided timestep subset τ_1 > τ_2 > ... > τ_S.
    taus = _build_timestep_subset(core.num_timesteps, num_steps).to(device)

    # x_T ~ N(0, I) — the only source of randomness when η = 0.
    x = randn_with_generator(shape, generator, device, work_dtype)

    sqrt_ac = core.sqrt_alphas_cumprod.to(device=device, dtype=work_dtype)
    sqrt_omac = core.sqrt_one_minus_alphas_cumprod.to(device=device, dtype=work_dtype)
    alpha_bar = core.alphas_cumprod.to(device=device, dtype=work_dtype)

    for i, tau in enumerate(taus):
        tau_prev = int(taus[i + 1]) if i + 1 < len(taus) else 0  # last step goes to x_0
        t_batch = torch.full((shape[0],), int(tau), device=device, dtype=torch.long)

        # 1) Predict noise and x0 (theory.md §18 step 1).
        eps_pred = model(x, t_batch)
        x0_hat = (x - sqrt_omac[tau] * eps_pred) / sqrt_ac[tau].clamp_min(1e-8)

        # 2) Direction toward x_{tau_prev} using the *predicted* eps (theory.md §18 step 2).
        ab_tau = alpha_bar[tau]
        ab_prev = alpha_bar[tau_prev] if tau_prev >= 0 else torch.tensor(1.0, device=device)
        # Standard DDIM direction term.
        dir_xt = torch.sqrt(torch.clamp(1.0 - ab_prev, min=0.0)) * eps_pred
        prev_mean = sqrt_ac[tau_prev] * x0_hat + dir_xt

        if tau_prev == 0:
            # Final step: x_0 with no added noise.
            x = prev_mean
            continue

        if eta > 0.0:
            # General η > 0 stochastic term. Variance of the reverse step:
            #   σ² = η² · (1 - ᾱ_prev)/(1 - ᾱ_tau) · (1 - ᾱ_tau/ᾱ_prev)
            variance = (
                eta ** 2
                * (1.0 - ab_prev) / (1.0 - ab_tau).clamp_min(1e-8)
                * (1.0 - ab_tau / ab_prev.clamp_min(1e-8))
            )
            sigma = torch.sqrt(torch.clamp(variance, min=0.0))
            z = randn_with_generator(shape, generator, device, x.dtype)
            x = prev_mean + sigma * z
        else:
            # Deterministic DDIM (the default; theory.md §18 equations).
            x = prev_mean

    return x
