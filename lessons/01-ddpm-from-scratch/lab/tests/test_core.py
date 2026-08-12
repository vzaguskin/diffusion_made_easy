"""Sanity checks for the DDPM core math (task 3.6).

These are small, fast, CPU-only checks — not a full pytest suite. Run with:

    uv run python -m pytest tests/ -q
    # or, without pytest:
    uv run python tests/test_core.py
"""

from __future__ import annotations

import torch

from ddpm_lab.core import DiffusionCore


def _make_core() -> DiffusionCore:
    # Use the canonical DDPM defaults from theory.md §3.
    return DiffusionCore(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)


def test_schedule_default_values():
    """betas span [1e-4, 0.02] over 1000 steps (theory.md §3)."""
    core = _make_core()
    assert core.num_timesteps == 1000
    assert core.betas.shape == (1000,)
    assert torch.allclose(core.betas[0].float(), torch.tensor(1e-4), atol=1e-6)
    assert torch.allclose(core.betas[-1].float(), torch.tensor(0.02), atol=1e-5)


def test_precomputed_coefficients_consistency():
    """alphas_cumprod == cumprod(1 - betas), and the sqrt coefficients match."""
    core = _make_core()
    expected_ac = torch.cumprod(1.0 - core.betas, dim=0)
    assert torch.allclose(core.alphas_cumprod, expected_ac)
    assert torch.allclose(core.sqrt_alphas_cumprod, torch.sqrt(core.alphas_cumprod))
    assert torch.allclose(
        core.sqrt_one_minus_alphas_cumprod,
        torch.sqrt(1.0 - core.alphas_cumprod),
    )


def test_q_sample_with_noise_is_deterministic():
    """q_sample(x0, t, noise=...) == sqrt(ᾱ)·x0 + sqrt(1-ᾱ)·ε exactly (theory.md §4)."""
    core = _make_core().float()
    torch.manual_seed(0)
    x0 = torch.randn(8, 1, 28, 28)
    t = torch.randint(0, core.num_timesteps, (8,))
    noise = torch.randn(8, 1, 28, 28)

    xt = core.q_sample(x0, t, noise=noise)

    sqrt_ac = core.sqrt_alphas_cumprod[t].float().reshape(8, 1, 1, 1)
    sqrt_omac = core.sqrt_one_minus_alphas_cumprod[t].float().reshape(8, 1, 1, 1)
    expected = sqrt_ac * x0 + sqrt_omac * noise
    assert torch.allclose(xt, expected, atol=1e-6)


def test_q_sample_without_noise_is_stochastic():
    """q_sample without noise samples N(0,I): two calls differ."""
    core = _make_core().float()
    x0 = torch.randn(4, 1, 28, 28)
    t = torch.tensor([10, 100, 500, 999])
    a = core.q_sample(x0, t)
    b = core.q_sample(x0, t)
    assert not torch.allclose(a, b)


def test_predict_start_inverts_q_sample():
    """predict_start_from_noise(q_sample(x0, t, ε), t, ε) ≈ x0 (theory.md §14)."""
    core = _make_core().float()
    torch.manual_seed(0)
    x0 = torch.randn(8, 1, 28, 28)
    t = torch.randint(0, core.num_timesteps, (8,))
    noise = torch.randn(8, 1, 28, 28)

    xt = core.q_sample(x0, t, noise=noise)
    x0_rec = core.predict_start_from_noise(xt, t, noise)
    assert torch.allclose(x0_rec, x0, atol=1e-4)


def test_variance_preserving_xT():
    """x_T has variance ≈ 1 when x0 has variance ≈ 1 (theory.md §3, VP property)."""
    core = _make_core().float()
    torch.manual_seed(0)
    # x0 ~ N(0, I): variance is exactly 1.
    x0 = torch.randn(2048, 1, 28, 28)
    t_T = torch.full((2048,), core.num_timesteps - 1)  # t = T (last step)
    xt = core.q_sample(x0, t_T)
    var = xt.var().item()
    # With beta_end = 0.02, alpha_bar_T is very small (~5e-6), so x_T ≈ pure noise.
    # Variance should be within a few % of 1.0.
    assert abs(var - 1.0) < 0.05, f"x_T variance {var} is far from 1.0"


def test_compute_loss_is_plain_mse():
    """compute_loss == torch.mean((pred - tgt)^2) — no weighting (theory.md §15)."""
    torch.manual_seed(0)
    pred = torch.randn(8, 1, 28, 28)
    tgt = torch.randn(8, 1, 28, 28)
    got = DiffusionCore.compute_loss(pred, tgt)
    expected = torch.mean((pred - tgt) ** 2)
    assert torch.allclose(got, expected)


if __name__ == "__main__":
    # Allow running without pytest: `python tests/test_core.py`
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} checks passed")
    raise SystemExit(0 if passed == len(fns) else 1)
