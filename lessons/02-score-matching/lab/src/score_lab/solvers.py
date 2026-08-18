"""Numerical solvers for the reverse SDE and the probability-flow ODE — §15.

All solvers share one interface: a callable ``drift(x, t) -> dx/dt``, a
starting cloud ``x``, a *decreasing* time grid (sampling runs t: 1 → 0) and,
for the stochastic one, a ``torch.Generator``. Each run returns a
``SolveResult`` with the final cloud, the *actual* number of drift
evaluations (NFE) and the wall-clock time — the two currencies of §15's
solver comparison. The NFE budget (not the step count) is the primary knob:
``nfe_to_grid`` turns "I can afford N model calls" into a time grid per
method (Euler-family: 1 NFE/step, Heun: 2, RK4: 4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

NFE_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4}


@dataclass
class SolveResult:
    x: torch.Tensor          # final cloud
    nfe: int                 # actual drift evaluations
    seconds: float           # wall-clock of the integration loop
    stochastic: bool


class CountingDrift:
    """Wrap a drift callable to count evaluations (and warm up timing)."""

    def __init__(self, drift) -> None:
        self.drift = drift
        self.calls = 0

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.drift(x, t)


def _time_grid(n_steps: int, device) -> torch.Tensor:
    """Decreasing t: 1 → 0 in ``n_steps`` intervals (n_steps+1 points)."""
    return torch.linspace(1.0, 0.0, n_steps + 1, device=device)


def nfe_to_grid(nfe_budget: int, method: str, device) -> torch.Tensor:
    """Time grid for an NFE budget: steps = budget / (NFE per step)."""
    per = NFE_PER_STEP[method]
    n_steps = max(nfe_budget // per, 1)
    return _time_grid(n_steps, device)


@torch.no_grad()
def euler_maruyama(drift, diffusion, x, t_grid, generator=None) -> SolveResult:
    """Euler-Maruyama for the reverse SDE: x ← x + drift·Δt + g·√|Δt|·z (§15).

    ``diffusion(t, batch, device)`` returns the per-sample g(t); the grid is
    decreasing so Δt < 0 and the noise scale uses |Δt|.
    """
    counting = CountingDrift(drift)
    t0 = time.perf_counter()
    for i in range(len(t_grid) - 1):
        t = t_grid[i]
        dt = t_grid[i + 1] - t
        g = diffusion(t, x.shape[0], x.device)
        z = torch.randn(x.shape, generator=generator, device=x.device)
        x = x + counting(x, t) * dt + g * abs(float(dt)) ** 0.5 * z
    return SolveResult(x=x, nfe=counting.calls,
                       seconds=time.perf_counter() - t0, stochastic=True)


# Evaluation times are clamped away from t = 0: the VP noise std √(1−ᾱ) → 0
# there, so a drift evaluation exactly at t = 0 explodes (score = −ε/std).
# Multi-stage methods (Heun/RK4) probe t + dt = 0 on the last step; clamping
# to this floor keeps them finite with no visible effect on the solution.
_T_FLOOR = 1e-3


def _step_euler(f, x, t, dt):
    return x + f(x, t.clamp_min(_T_FLOOR)) * dt


def _step_heun(f, x, t, dt):
    k1 = f(x, t.clamp_min(_T_FLOOR))
    k2 = f(x + k1 * dt, (t + dt).clamp_min(_T_FLOOR))
    return x + 0.5 * (k1 + k2) * dt


def _step_rk4(f, x, t, dt):
    t0 = t.clamp_min(_T_FLOOR)
    tm = (t + dt / 2).clamp_min(_T_FLOOR)
    k1 = f(x, t0)
    k2 = f(x + k1 * dt / 2, tm)
    k3 = f(x + k2 * dt / 2, tm)
    k4 = f(x + k3 * dt, (t + dt).clamp_min(_T_FLOOR))
    return x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


_STEPS = {"euler": _step_euler, "heun": _step_heun, "rk4": _step_rk4}


@torch.no_grad()
def ode_solver(drift, x, t_grid, method: str = "euler") -> SolveResult:
    """Deterministic ODE solver (Euler / Heun / RK4) for the PF-ODE.

    Ignores randomness entirely: same start + same grid → bit-identical end.
    """
    step = _STEPS[method]
    counting = CountingDrift(drift)
    t0 = time.perf_counter()
    for i in range(len(t_grid) - 1):
        dt = t_grid[i + 1] - t_grid[i]
        x = step(counting, x, t_grid[i], dt)
    return SolveResult(x=x, nfe=counting.calls,
                       seconds=time.perf_counter() - t0, stochastic=False)
