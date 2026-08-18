"""Tests for the continuous SDE wrappers: time maps, drifts, forward agreement."""

from __future__ import annotations

import torch

from score_lab.mnist_ve_vp import VESchedule, VPSchedule
from score_lab.sde import ContinuousVE, ContinuousVP
from score_lab.solvers import euler_maruyama


def test_ve_time_map():
    ve = ContinuousVE(VESchedule(sigma_max=5.0, sigma_min=0.01, n_levels=200))
    t = torch.tensor([0.0, 0.5, 1.0])
    std = ve.noise_std(t, batch=3, device="cpu")
    assert float(std[0]) <= ve.sigma_min + 1e-6   # σ(0) = σ_min ≈ 0
    assert abs(float(std[2]) - 5.0) < 1e-5
    # monotone non-decreasing
    ts = torch.linspace(0, 1, 50)
    assert torch.all(ve.noise_std(ts, 50, "cpu")[1:] >= ve.noise_std(ts, 50, "cpu")[:-1] - 1e-8)
    # index quantization hits the endpoints exactly
    assert ve.t_to_index(torch.tensor([0.0, 1.0])).tolist() == [0, 199]


def test_vp_time_map():
    vp = ContinuousVP(VPSchedule(num_timesteps=1000))
    t = torch.tensor([0.0, 1.0])
    std = vp.noise_std(t, 2, "cpu")
    assert abs(float(std[0])) < 1e-4          # ᾱ(0) ≈ 1 → std ≈ 0
    assert abs(float(std[1]) - 1.0) < 1e-4    # ᾱ(1) ≈ 0 → std ≈ 1


def test_forward_sde_matches_discrete_ve():
    """Euler-Maruyama on the VE forward SDE reproduces x0 + σ·ε as dt → 0."""
    ve = ContinuousVE(VESchedule(sigma_max=5.0, sigma_min=0.01, n_levels=200))
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(256, 1, 4, 4, generator=g)
    t_end = 0.7
    sigma_end = float(ve.sigma(torch.tensor(t_end)))

    errs = []
    for n_steps in (8, 64, 512):
        gen = torch.Generator().manual_seed(1)
        x = x0.clone()
        grid = torch.linspace(0.0, t_end, n_steps + 1)
        diffusion_fwd = lambda t, batch, device: ve.diffusion(t, batch, device)
        res = euler_maruyama(lambda xt, tt: torch.zeros_like(xt), diffusion_fwd,
                             x, grid, generator=gen)
        # empirical noise std added over the cloud
        errs.append(((res.x - x0).std() - sigma_end).abs().item())
    assert errs[2] < errs[0]          # error shrinks with finer steps
    assert errs[2] < 0.15 * sigma_end


def test_forward_sde_matches_discrete_vp():
    """VP forward SDE keeps variance ≈ 1 at t=1 (ᾱ(1) ≈ 0)."""
    vp = ContinuousVP(VPSchedule(num_timesteps=1000))
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(512, 1, 4, 4, generator=g)
    gen = torch.Generator().manual_seed(2)
    grid = torch.linspace(0.0, 1.0, 400 + 1)
    res = euler_maruyama(
        lambda xt, tt: vp.forward_drift(xt, tt),
        lambda t, batch, device: vp.diffusion(t, batch, device),
        x0, grid, generator=gen)
    assert abs(float(res.x.std()) - 1.0) < 0.05


def test_pf_ode_deterministic_and_drifts_differ():
    ve = ContinuousVE(VESchedule(sigma_max=5.0, sigma_min=0.01, n_levels=200))

    class ConstEps(torch.nn.Module):
        def forward(self, x, idx):
            return torch.ones_like(x)

    model = ConstEps()
    x = torch.randn(4, 1, 4, 4, generator=torch.Generator().manual_seed(3))
    t = torch.full((4,), 0.5)
    d_ode = ve.pf_ode_drift(model, x, t)
    d_sde = ve.reverse_sde_drift(model, x, t)
    # reverse SDE drift is the ODE drift minus another ½g²s → they differ
    assert not torch.allclose(d_ode, d_sde)
    # with eps ≡ 1: s = −1/σ, g² = 2σ²·ln r → ODE drift = ½·g²/σ = σ·ln r
    sigma = float(ve.sigma(torch.tensor(0.5)))
    log_r = float(torch.log(torch.tensor(ve.sigma_max / ve.sigma_min)))
    assert torch.allclose(d_ode, torch.full_like(x, sigma * log_r), atol=1e-5)
    # and the SDE drift is exactly twice that
    assert torch.allclose(d_sde, torch.full_like(x, 2 * sigma * log_r), atol=1e-5)
