"""Tests for the 2D toy data: analytic score == autograd gradient of log p."""

from __future__ import annotations

import torch

from score_lab.toy_data import DATASETS, build_dataset


def _grid(n: int = 20, lim: float = 3.0) -> torch.Tensor:
    xs = torch.linspace(-lim, lim, n)
    ys = torch.linspace(-lim, lim, n)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def test_analytic_score_matches_autograd():
    """score(x) must equal ∇ log p(x) computed by autograd (atol 1e-4)."""
    grid = _grid()
    for name in DATASETS:
        dist = build_dataset(name)
        x = grid.clone().requires_grad_(True)
        lp = dist.log_p(x)
        (grad,) = torch.autograd.grad(lp.sum(), x)
        analytic = dist.score(grid)
        err = (grad - analytic).abs().max().item()
        assert err < 1e-4, f"{name}: max |autograd - analytic| = {err}"


def test_sampling_shape_and_range():
    for name in DATASETS:
        dist = build_dataset(name)
        s = dist.sample(1000, generator=torch.Generator().manual_seed(0))
        assert s.shape == (1000, 2)
        assert torch.isfinite(s).all()
        assert s.abs().max() < 4.0


def test_gaussians8_score_points_to_center():
    """Far from the ring, the score points inward (toward the mass)."""
    dist = build_dataset("gaussians8")
    x_far = torch.tensor([[6.0, 0.0]])
    s = dist.score(x_far)[0]
    # At (6,0) the pull should be dominantly toward -x direction.
    assert s[0] < 0 and abs(s[0]) > abs(s[1]) * 3


if __name__ == "__main__":
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
