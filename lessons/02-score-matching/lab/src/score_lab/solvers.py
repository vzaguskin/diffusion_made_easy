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


_T_SAFETY = 1e-3  # keep drift evaluations off t = 0 (VE sigma_min / VP std = 0)


@torch.no_grad()
def pc_langevin_ve(branch, model, x, m_levels: int, euler_sub: int = 1,
                   corrector_steps: int = 1, corrector_scale: float = 0.15,
                   generator=None) -> SolveResult:
    """NCSN-style predictor-corrector on a VE branch, over ``m_levels`` levels.

    Unlike ``mnist_ve_vp.sample_ve`` (bound to the training ladder), the level
    count here is free: the time grid ``t = linspace(1, 0, m_levels+1)`` maps
    onto the model's conditioning indices via the continuous wrapper, so the
    NFE budget ``m_levels · (euler_sub + corrector_steps)`` can be any number.

    Predictor = Euler step on the PF-ODE (its drift is exactly ``σ'(t)·ε̂``, so
    one step removes ``Δσ·ε̂``); corrector = annealed Langevin at the new level.
    A heavy corrector amplifies the model's eps-error into speckle — with this
    lab's rough VE model, ``corrector_steps = 1`` gives the cleanest digits.
    """
    device = x.device
    grid = torch.linspace(1.0, 0.0, m_levels + 1, device=device)
    counting = CountingDrift(lambda xt, tt: branch.pf_ode_drift(model, xt, tt))
    corrector_calls = 0
    t0 = time.perf_counter()
    for i in range(m_levels):
        t, tn = grid[i], grid[i + 1].clamp_min(_T_SAFETY)
        for _ in range(euler_sub):
            dt = (grid[i + 1] - t) / euler_sub
            x = x + counting(x, t) * dt
        s = float(branch.sigma(tn))
        alpha = corrector_scale * s * s
        for _ in range(corrector_steps):
            eps_hat = branch.eps(model, x, tn)
            corrector_calls += 1
            score = -eps_hat / s
            z = torch.randn(x.shape, generator=generator, device=device)
            x = x + 0.5 * alpha * score + alpha ** 0.5 * z
    return SolveResult(x=x, nfe=counting.calls + corrector_calls,
                       seconds=time.perf_counter() - t0, stochastic=True)
