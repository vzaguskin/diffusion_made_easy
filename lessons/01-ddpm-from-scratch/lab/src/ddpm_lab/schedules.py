"""Noise schedules for the forward diffusion process.

Theory reference: ``theory.md`` §3 «Прямой процесс (forward): как мы портим картинку».

A *noise schedule* is the sequence of small variances ``beta_t`` that say how much
Gaussian noise is added at each forward step ``t = 1..T``. The forward process is
variance-preserving (VP): ``x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * eps``
(see theory.md §3, "Почему именно такие коэффициенты?").

DDPM (Ho et al. 2020) uses a *linear* schedule: ``beta_t`` grows linearly from
``beta_start`` to ``beta_end`` over ``T`` steps, with the canonical values
``beta_start = 1e-4``, ``beta_end = 0.02``, ``T = 1000``.
"""

from __future__ import annotations

import torch


def linear_beta_schedule(
    beta_start: float,
    beta_end: float,
    num_timesteps: int,
) -> torch.Tensor:
    """Return a linear noise schedule ``[beta_start, ..., beta_end]``.

    Parameters
    ----------
    beta_start : float
        Variance added at the first step ``t = 1`` (theory.md §3: ``1e-4``).
    beta_end : float
        Variance added at the last step ``t = T`` (theory.md §3: ``0.02``).
    num_timesteps : int
        Number of diffusion steps ``T`` (theory.md §3: ``1000``).

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``[num_timesteps]`` with values linearly spaced
        between ``beta_start`` and ``beta_end``.
    """
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
