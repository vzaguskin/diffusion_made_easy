"""Tests for Langevin sampling: convergence, reproducibility, annealing wins."""

from __future__ import annotations

import torch

from score_lab.config import geometric_sigma_ladder
from score_lab.langevin import annealed_langevin, langevin_sample, mode_coverage
from score_lab.toy_data import build_dataset


def _analytic_score_fn(dist):
    """Score of the *σ-smoothed* density: each component's std grows as
    sqrt(σ_i² + σ²) — the correct target at noise level σ (theory.md §7-8)."""
    from score_lab.toy_data import GaussianMixture2D

    cache: dict[float, GaussianMixture2D] = {}

    def fn(x, sigma):
        s = float(sigma.reshape(-1)[0])
        if s not in cache:
            cache[s] = GaussianMixture2D(
                means=dist.means,
                stds=(dist.stds ** 2 + s ** 2).sqrt(),
                weights=dist.weights,
            )
        return cache[s].score(x).float()

    return fn


def test_langevin_with_true_score_finds_modes():
    """Langevin + analytic score lands in the modes (theory.md §4)."""
    dist = build_dataset("gaussians8")
    x0 = torch.randn(200, 2, generator=torch.Generator().manual_seed(0)) * 2.5
    x = langevin_sample(_analytic_score_fn(dist), x0, n_steps=800, eps=0.02, sigma=0.12)
    cov = mode_coverage(x, dist.means, radius=0.5)
    assert cov >= 0.9, f"coverage {cov:.2f} < 0.9"


def test_langevin_reproducible():
    dist = build_dataset("gaussians8")
    x0 = torch.randn(50, 2, generator=torch.Generator().manual_seed(1))
    a = langevin_sample(_analytic_score_fn(dist), x0, 100, eps=0.01, sigma=0.2,
                        generator=torch.Generator().manual_seed(2))
    b = langevin_sample(_analytic_score_fn(dist), x0, 100, eps=0.01, sigma=0.2,
                        generator=torch.Generator().manual_seed(2))
    assert torch.allclose(a, b, atol=1e-6)


def test_annealing_beats_single_sigma():
    """§8 in numbers: annealed ≥90% mode coverage; single small σ collapses.

    The demo needs two ingredients to actually show the failure mode:
    * chains start clustered in a low-density spot (near the ring's center),
      where the small-σ score is ≈ 0 and there is nothing to follow;
    * both samplers use the *same final* step size (step_scale·σ_min²) —
      a large fixed ε would let the single-σ walker diffuse anywhere by
      luck, which is not "sampling", just a random walk.
    """
    dist = build_dataset("gaussians8")
    sigmas = geometric_sigma_ladder(1.0, 0.02, 10)
    x0 = torch.randn(300, 2, generator=torch.Generator().manual_seed(3)) * 0.2

    x_ann, _snaps = annealed_langevin(
        _analytic_score_fn(dist), x0, sigmas,
        steps_per_level=100, step_scale=0.05,
        generator=torch.Generator().manual_seed(4),
    )
    cov_ann = mode_coverage(x_ann, dist.means, radius=0.5)
    hit_ann = (torch.cdist(x_ann, dist.means).min(dim=1).values < 0.5).float().mean()

    x_single = langevin_sample(_analytic_score_fn(dist), x0, n_steps=1000,
                               eps=0.05 * 0.02 ** 2, sigma=0.02,
                               generator=torch.Generator().manual_seed(4))
    cov_single = mode_coverage(x_single, dist.means, radius=0.5)
    hit_single = (torch.cdist(x_single, dist.means).min(dim=1).values < 0.5).float().mean()

    assert cov_ann >= 0.9, f"annealed coverage {cov_ann:.2f}"
    assert cov_ann > cov_single, f"annealed {cov_ann:.2f} vs single {cov_single:.2f}"
    assert hit_ann > 0.9 and hit_single < 0.2, (
        f"hit rate: annealed {hit_ann:.2f}, single {hit_single:.2f}")


def test_annealed_snapshots_shape():
    dist = build_dataset("gaussians8")
    sigmas = geometric_sigma_ladder(1.0, 0.1, 3)
    x0 = torch.randn(20, 2)
    _, snaps = annealed_langevin(_analytic_score_fn(dist), x0, sigmas,
                                 steps_per_level=5, return_snapshots=True)
    assert set(snaps.keys()) == {0, 1, 2}
    assert len(snaps[0]) == 6  # initial + 5 steps
    assert snaps[0][0].shape == (20, 2)


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
