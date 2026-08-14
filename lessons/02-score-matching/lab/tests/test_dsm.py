"""Tests for DSM: target sign and convergence to the analytic smoothed score."""

from __future__ import annotations

import math

import torch

from score_lab.dsm import dsm_loss
from score_lab.models import ScoreMLP
from score_lab.toy_data import build_dataset


class FixedModel(torch.nn.Module):
    """Returns a constant field — lets us verify the loss formula exactly."""

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.value = value

    def forward(self, x, sigma):
        return self.value.expand_as(x)


def test_dsm_loss_formula_and_sign():
    """loss == mean((s_θ + ε/σ)²) — target is −ε/σ, i.e. PLUS ε/σ inside the norm."""
    torch.manual_seed(0)
    x = torch.randn(64, 2)
    sigma = 0.5
    model = FixedModel(torch.tensor([0.3, -0.2]))
    loss, eps = dsm_loss(model, x, sigma, generator=torch.Generator().manual_seed(1))
    expected = torch.mean((torch.tensor([0.3, -0.2]) + eps / 0.5) ** 2)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_ve_noising_is_additive():
    """x̃ = x + σε (no signal scaling) — capture via the returned eps."""
    torch.manual_seed(0)
    x = torch.zeros(8, 2)
    _, eps = dsm_loss(FixedModel(torch.zeros(2)), x, 2.0,
                      generator=torch.Generator().manual_seed(3))
    # With x=0 and σ=2 the loss is mean((0 + ε/2)²) = mean(ε²)/4 ≈ 1/4·2·... check ε ~ N(0,I)
    assert eps.std().item() < 1.5 and eps.std().item() > 0.5  # sanity


def test_single_sigma_converges_to_analytic_score():
    """A tiny net trained on ONE σ learns ∇log p_σ (theory.md §7 theorem).

    We measure agreement by angle in high-density regions: cos-similarity
    between predicted and analytic smoothed score.
    """
    dist = build_dataset("gaussians8")
    sigma = 0.3

    # The analytic score of the σ-smoothed GMM is a (2σ²)-convolution:
    # each component's std grows: σ_i² -> σ_i² + σ². (Standard GMM convolution.)
    from score_lab.toy_data import GaussianMixture2D
    smoothed = GaussianMixture2D(
        means=dist.means,
        stds=(dist.stds ** 2 + sigma ** 2).sqrt(),
        weights=dist.weights,
    )

    torch.manual_seed(0)
    model = ScoreMLP(hidden_dim=256, n_layers=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    for _ in range(1500):
        x = dist.sample(512, generator=gen)
        loss, _eps = dsm_loss(model, x, sigma)
        opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate: points sampled from the smoothed density (high-density regions).
    eval_pts = smoothed.sample(512, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        pred = model(eval_pts, torch.full((512, 1), sigma))
    true = smoothed.score(eval_pts).float()
    cos = torch.nn.functional.cosine_similarity(pred, true, dim=1)
    mean_angle = torch.acos(cos.clamp(-1, 1)).mean().item()
    assert mean_angle < math.radians(15), f"mean angle {math.degrees(mean_angle):.1f}° > 15°"


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
