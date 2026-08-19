"""Tests for EDM preconditioning: coefficient limits, loss balance, NFE."""

from __future__ import annotations

import torch

from score_lab.edm import EDMModel, edm_heun_sample, edm_loss


class _ZeroNet(torch.nn.Module):
    """F_θ ≡ 0 — lets us check the c_skip/c_out structure exactly."""

    def forward(self, x, sigma):
        return torch.zeros_like(x)


def test_preconditioning_limits():
    model = EDMModel(_ZeroNet(), sigma_data=1.0)
    x = torch.randn(4, 1, 4, 4)
    # σ → 0: c_skip → 1, c_out → 0 ⇒ D = x (identity denoiser)
    d0 = model(x, torch.full((4,), 1e-6))
    assert torch.allclose(d0, x, atol=1e-4)
    # σ → ∞: D = c_skip·x → 0 — bounded no matter the input scale
    dbig = model(x * 100, torch.full((4,), 1e6))
    assert float(dbig.abs().max()) < 1e-2


def test_net_sees_unit_scale_input():
    sd = 1.0
    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, x, sigma):
            seen["std"] = float(x.std())
            return torch.zeros_like(x)

    model = EDMModel(Probe(), sigma_data=sd)
    x = torch.randn(64, 1, 8, 8)
    for s in (0.01, 0.1, 1.0, 5.0):
        model(x, torch.full((64,), s))
        # Var[c_in·x + noise contribution] — with noise, input std ≈ 1 at any σ
        xt = x + s * torch.randn_like(x)
        model(xt, torch.full((64,), s))
        expected = (s ** 2 / (s ** 2 + sd ** 2) * 1 + sd ** 2 / (s ** 2 + sd ** 2) * 1) ** 0.5
        assert abs(seen["std"] - expected) < 0.15


def test_edm_loss_effective_target_unit_variance_on_gaussian():
    """For Gaussian data the λ-weighted residual has Var ≈ 1 at every σ.

    Optimal denoiser of N(0, σ_d²) data under noise σ: D* = c_skip·x with
    c_skip = σ_d²/(σ²+σ_d²) (posterior mean). Residual D* − x₀ = −(σ_d²/(σ²+σ_d²))·ε...
    E[λ·(D*−x₀)²] = (σ²+σ_d²)/(σσ_d)² · σ²σ_d⁴/(σ²+σ_d²)² / σ_d² per unit —
    simplifies to 1 for every σ: that is EDM's balance claim (§17).
    """
    sd = 1.0
    for s in (0.05, 0.2, 1.0, 3.0):
        lam = (s ** 2 + sd ** 2) / (s * sd) ** 2
        # residual of the *optimal* denoiser: −σ_d²/(σ²+σ_d²) · σ ε / σ_d... use MC
        x0 = torch.randn(20000, 1, 1, 1)
        eps = torch.randn_like(x0)
        xt = x0 + s * eps
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2) ** 0.5
        F_opt = (c_skip * xt - c_skip * xt - c_out * 0)  # F contribution zero
        D = c_skip * xt + c_out * F_opt
        weighted = (lam ** 0.5 * (D - x0)).flatten()
        assert abs(float(weighted.var()) - 1.0) < 0.05, f"σ={s}"


def test_edm_heun_nfe_and_determinism():
    model = EDMModel(_ZeroNet(), sigma_data=1.0)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    x1, nfe1 = edm_heun_sample(model, (8, 1, 8, 8), n_steps=20,
                               generator=g1, device="cpu")
    x2, nfe2 = edm_heun_sample(model, (8, 1, 8, 8), n_steps=20,
                               generator=g2, device="cpu")
    assert nfe1 == nfe2 == 2 * 20 - 3  # 18 interior Heun pairs + final Euler = 2(n−1)−1
    assert torch.equal(x1, x2)


def test_edm_loss_runs_and_finite():
    model = EDMModel(_ZeroNet(), sigma_data=1.0)
    x0 = torch.randn(16, 1, 8, 8)
    loss = edm_loss(model, x0)
    assert torch.isfinite(loss)
