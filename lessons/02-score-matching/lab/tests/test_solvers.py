"""Tests for the numerical solvers: accuracy order, NFE, determinism, marginals."""

from __future__ import annotations

import torch

from score_lab.solvers import SolveResult, nfe_to_grid, ode_solver, euler_maruyama


def _linear_drift(x, t):
    """dx/dt = a·x + b (per-batch constant): exact solution known."""
    a, b = -2.0, 1.0
    return a * x + b


def _exact(x0, t0, t1):
    a, b = -2.0, 1.0
    return (x0 + b / a) * torch.exp(a * (t1 - t0)) - b / a


def test_ode_accuracy_order():
    x0 = torch.tensor([[1.0]])
    t0, t1 = torch.tensor([0.0]), torch.tensor([1.0])
    exact = _exact(x0, t0, t1)
    errs = {"euler": [], "heun": [], "rk4": []}
    for n in (10, 40, 160):
        for m in errs:
            grid = torch.linspace(float(t0), float(t1), n + 1)
            res = ode_solver(_linear_drift, x0.clone(), grid, method=m)
            errs[m].append(abs(float(res.x[0, 0] - exact[0, 0])))
    # errors decrease with h; higher order must beat lower order at n=160
    for m in errs:
        assert errs[m][-1] < errs[m][0]
    assert errs["rk4"][-1] < errs["heun"][-1] < errs["euler"][-1]
    # observed rates: euler ~h, heun ~h²; rk4 is already at float32 machine
    # precision here (~6e-8), so only require it to stay at that floor
    assert errs["euler"][-1] / errs["euler"][1] < 0.35    # h/4 → ~4x smaller
    assert errs["heun"][-1] / errs["heun"][1] < 0.15      # h/4 → ~16x smaller
    assert errs["rk4"][-1] <= errs["rk4"][1] < 1e-5


def test_nfe_counter_matches_calls():
    calls = {"n": 0}

    def drift(x, t):
        calls["n"] += 1
        return -x

    x = torch.randn(3, 1, 4, 4)
    for m, per_step in (("euler", 1), ("heun", 2), ("rk4", 4)):
        calls["n"] = 0
        grid = torch.linspace(1.0, 0.0, 11)  # 10 steps
        res = ode_solver(drift, x.clone(), grid, method=m)
        assert res.nfe == calls["n"] == 10 * per_step


def test_ode_deterministic():
    x = torch.randn(2, 1, 4, 4, generator=torch.Generator().manual_seed(0))
    grid = torch.linspace(1.0, 0.0, 21)
    a = ode_solver(_linear_drift, x.clone(), grid, method="heun")
    b = ode_solver(_linear_drift, x.clone(), grid, method="heun")
    assert torch.equal(a.x, b.x)
    assert not a.stochastic


def test_nfe_to_grid_respects_budget():
    for method, per in (("euler", 1), ("heun", 2), ("rk4", 4)):
        grid = nfe_to_grid(40, method, torch.device("cpu"))
        assert len(grid) - 1 == 40 // per
        assert abs(float(grid[0]) - 1.0) < 1e-9 and abs(float(grid[-1])) < 1e-9


def test_euler_maruyama_converges_on_gaussian():
    """Reverse-time OU: exact stationary distribution recoverable analytically.

    Forward SDE dx = −x dt + √2 dW has N(0,1) as stationary; integrate the
    *reverse* process from a shifted start back to equilibrium.
    """
    drift = lambda x, t: -x
    diffusion = lambda t, batch, device: torch.full((batch, 1, 1, 1), 2.0 ** 0.5)
    x = torch.full((4096, 1, 1, 1), 3.0)
    gen = torch.Generator().manual_seed(0)
    grid = torch.linspace(0.0, 4.0, 400 + 1)
    res = euler_maruyama(drift, diffusion, x, grid, generator=gen)
    mean = float(res.x.mean())
    std = float(res.x.std())
    assert abs(mean) < 0.1 and abs(std - 1.0) < 0.1


def test_sde_vs_ode_marginals_agree():
    """PF-ODE of an OU process has the same marginals as the SDE (§14)."""
    # SDE: dx = −x dt + dW → marginal std at time T from x0=0: sqrt((1−e^{−2T})/2)
    drift = lambda x, t: -x
    g = lambda t, batch, device: torch.ones(batch, 1, 1, 1)
    T = 1.0
    n = 20000

    x_sde = torch.zeros(n, 1, 1, 1)
    res_sde = euler_maruyama(drift, g, x_sde, torch.linspace(0, T, 800 + 1),
                             generator=torch.Generator().manual_seed(1))

    # PF-ODE: same marginal transition if started from the *distribution* at
    # t0 (deterministic map x ↦ x·sqrt(var(T)/var(t0))). Sample N(0, var(t0)).
    def ode_drift(x, t):
        var = (1 - torch.exp(-2 * t.clamp_min(1e-2))) / 2
        return -x + x / (2 * var)

    t0 = 0.02
    var0 = (1 - torch.exp(torch.tensor(-2.0 * t0))) / 2
    x_ode = torch.randn(n, 1, 1, 1, generator=torch.Generator().manual_seed(2)) * var0.sqrt()
    res_ode = ode_solver(ode_drift, x_ode,
                         torch.linspace(t0, T, 4000 + 1), method="heun")

    target_std = ((1 - torch.exp(torch.tensor(-2.0 * T))) / 2).sqrt()
    assert abs(float(res_sde.x.std()) - float(target_std)) < 0.05
    assert abs(float(res_ode.x.std()) - float(target_std)) < 0.05
    assert isinstance(res_sde, SolveResult) and res_sde.stochastic
