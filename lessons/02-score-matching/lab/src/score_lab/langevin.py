"""Langevin dynamics and annealed Langevin dynamics.

Theory references:
* §4 «Сэмплирование через Langevin dynamics» — the discrete update
  ``x_{k+1} = x_k + (ε/2)·∇log p(x_k) + √ε·z``. Follow the score with inertia
  from the noise; run long enough and you sample from p.
* §8 «Многоуровневый шум: почему одного σ недостаточно» — with a *single*
  small σ the walk gets stuck near wherever it starts (poor mode coverage);
  the fix is a *ladder*: start at large σ (smooth landscape, easy travel),
  anneal down (refine into the modes), carrying the state across levels.
"""

from __future__ import annotations

from typing import Callable

import torch

# A score function of (x, sigma) -> score.
ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def langevin_sample(
    score_fn: ScoreFn,
    x0: torch.Tensor,
    n_steps: int,
    eps: float,
    sigma: float,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
):
    """Discrete Langevin dynamics at a single noise level (theory.md §4).

    Parameters
    ----------
    score_fn : callable(x, sigma) -> score
        The (learned or analytic) score.
    x0 : Tensor[N, 2]
        Starting points.
    n_steps, eps, sigma
        Steps, step size, noise level.
    return_trajectory
        If True, also return all intermediate states [n_steps+1, N, 2].

    Returns
    -------
    x (Tensor[N, 2]) and optionally the trajectory.
    """
    x = x0.clone()
    traj = [x.clone()] if return_trajectory else None
    sigma_t = torch.full((x.shape[0], 1), float(sigma), device=x.device, dtype=x.dtype)
    for _ in range(n_steps):
        s = score_fn(x, sigma_t)
        z = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        x = x + 0.5 * eps * s + torch.sqrt(torch.tensor(eps)) * z
        if traj is not None:
            traj.append(x.clone())
    if return_trajectory:
        return x, torch.stack(traj)
    return x


def annealed_langevin(
    score_fn: ScoreFn,
    x0: torch.Tensor,
    sigmas: torch.Tensor,
    steps_per_level: int,
    step_scale: float = 0.1,
    generator: torch.Generator | None = None,
    return_snapshots: bool = True,
    snapshot_stride: int = 1,
):
    """Annealed Langevin over the σ ladder (theory.md §8).

    Walk the levels from σ_max (first element) down to σ_min (last). The state
    carries over between levels (the cloud is NOT re-randomized) — that is the
    whole point of annealing: travel far on the smooth landscape, then refine.

    Step size per level: ``eps_i = step_scale · σ_i²``. Why this scaling?
    Near a mode of the σ-smoothed density the score is ≈ −(x−μ)/σ̃² (σ̃ the
    smoothed width), so the update is an OU-like contraction
    ``x ← x·(1 − c) + noise`` with stationary spread ≈ σ̃ — i.e. *scale
    invariant*: every level jitters by exactly one "mode width" of its own
    smoothing. A fixed ε (independent of σ) would either explode at small σ
    or freeze at large σ; dividing by σ_min² (as NCSN does with tiny α)
    explodes unless α is minuscule. ``step_scale`` is the knob: smaller =
    more careful/longer mixing, larger = faster but less stable.
    """
    x = x0.clone()
    snaps: dict[int, list[torch.Tensor]] = {}
    for lvl, sigma in enumerate(sigmas):
        s = float(sigma)
        eps = step_scale * s * s
        sigma_t = torch.full((x.shape[0], 1), s, device=x.device, dtype=x.dtype)
        snaps[lvl] = [x.clone()] if return_snapshots else []
        for k in range(steps_per_level):
            g = score_fn(x, sigma_t)
            z = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x = x + 0.5 * eps * g + (eps ** 0.5) * z
            if return_snapshots and (k + 1) % snapshot_stride == 0:
                snaps[lvl].append(x.clone())
    if return_snapshots:
        return x, snaps
    return x


def mode_coverage(samples: torch.Tensor, means: torch.Tensor, radius: float) -> float:
    """Fraction of modes covered by at least one sample within ``radius``."""
    d = torch.cdist(samples, means)                 # [N, K]
    hit = (d.min(dim=1).values < radius)
    covered = (d < radius).any(dim=0)               # [K]
    return covered.float().mean().item()
