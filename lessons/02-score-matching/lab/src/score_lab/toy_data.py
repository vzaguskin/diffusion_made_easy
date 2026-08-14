"""2D toy distributions with analytic log-density and score.

Theory reference: ``theory.md`` §6 «Score гауссиана: ключевой строительный блок».

Everything here is a Gaussian mixture (GMM), because for a mixture we know both
``log p`` and ``∇ log p`` in closed form:

    p(x)    = Σ_i  w_i · N(x; μ_i, σ_i² I)
    ∇log p  = Σ_i  w_i · N(x; μ_i, σ_i² I) · (−(x − μ_i)/σ_i²)  /  Σ_i w_i · N_i(x)

i.e. a *responsibility-weighted average* of the per-component scores — the
"building block" of §6. This gives us ground truth to compare the network
against, arrow by arrow.

Three datasets:
* ``gaussians8`` — an exact ring of 8 Gaussians (the honest one: density is the
  mixture itself, no approximation).
* ``moons``      — two half-moons, approximated by a GMM of narrow components
  placed along the curve.
* ``swiss-roll`` — a spiral, likewise GMM-approximated.

The approximation is documented in the README: for moons/spiral the "true score"
is the score of the GMM approximation, which is visually indistinguishable from
the data manifold at σ ≥ component width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GaussianMixture2D:
    """A 2D Gaussian mixture with sampling, log-density, and analytic score."""

    means: torch.Tensor      # [K, 2]
    stds: torch.Tensor       # [K]
    weights: torch.Tensor    # [K]

    def __post_init__(self) -> None:
        self.means = torch.as_tensor(self.means, dtype=torch.float32)
        self.stds = torch.as_tensor(self.stds, dtype=torch.float32)
        self.weights = torch.as_tensor(self.weights, dtype=torch.float32)
        assert self.means.shape == (len(self.stds), 2)
        assert torch.allclose(self.weights.sum(), torch.tensor(1.0), atol=1e-5)

    # ------------------------------------------------------------ sampling
    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw n points from the mixture."""
        idx = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        eps = torch.randn(n, 2, generator=generator)
        return self.means[idx] + self.stds[idx].unsqueeze(1) * eps

    # ---------------------------------------------------- log p and score
    def _component_logpdf(self, x: torch.Tensor) -> torch.Tensor:
        """log N(x; μ_i, σ_i² I) for all components: [N, K].

        Computed in float64: with narrow components (σ ≈ 0.06) the score
        magnitudes reach ~1/σ² ≈ 300, and float32 cancellation ruins the
        autograd cross-check below 1e-4.
        """
        x = x.double()
        means = self.means.double()
        stds = self.stds.double()
        diff = x.unsqueeze(1) - means.unsqueeze(0)               # [N, K, 2]
        sq = (diff * diff).sum(dim=2)                             # [N, K]
        return -sq / (2 * stds.unsqueeze(0) ** 2) - torch.log(
            2 * torch.pi * stds.unsqueeze(0) ** 2
        )

    def log_p(self, x: torch.Tensor) -> torch.Tensor:
        """log p(x) via logsumexp: [N] (float64 for precision)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        log_terms = self._component_logpdf(x) + torch.log(self.weights.double()).unsqueeze(0)
        return torch.logsumexp(log_terms, dim=1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Analytic ∇log p(x): [N, 2]  (theory.md §6).

        The responsibility-weighted average of per-component scores:
            ∇log p = Σ_i r_i(x) · (−(x − μ_i)/σ_i²),   r_i = w_i N_i / Σ_j w_j N_j
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        log_terms = self._component_logpdf(x) + torch.log(self.weights.double()).unsqueeze(0)
        # responsibilities r_i(x), softmax over components
        resp = torch.softmax(log_terms, dim=1)                    # [N, K]
        diff = x.double().unsqueeze(1) - self.means.double().unsqueeze(0)
        per_component_score = -diff / (self.stds.double().view(1, -1, 1) ** 2)
        return (resp.unsqueeze(2) * per_component_score).sum(dim=1)


# --------------------------------------------------------------------------
# Dataset builders
# --------------------------------------------------------------------------

def gaussians8(std: float = 0.12, radius: float = 1.6) -> GaussianMixture2D:
    """Ring of 8 Gaussians — the *exact* dataset (density = the mixture itself)."""
    angles = torch.linspace(0, 2 * torch.pi, 9)[:-1]
    means = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    return GaussianMixture2D(
        means=means,
        stds=torch.full((8,), std),
        weights=torch.full((8,), 1 / 8),
    )


def _gmm_along_curve(points: np.ndarray, component_std: float) -> GaussianMixture2D:
    """GMM with a narrow component at each of the given curve points."""
    means = torch.from_numpy(points).float()
    k = len(means)
    return GaussianMixture2D(
        means=means,
        stds=torch.full((k,), component_std),
        weights=torch.full((k,), 1.0 / k),
    )


def moons(n_components: int = 100, component_std: float = 0.08) -> GaussianMixture2D:
    """Two half-moons (sklearn-style geometry), GMM-approximated.

    Components are placed uniformly along each moon's arc; samples are drawn
    from the mixture. The "true score" is the score of this GMM.
    """
    t = np.linspace(0, np.pi, n_components // 2)
    upper = np.stack([np.cos(t), np.sin(t)], axis=1)
    lower = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    return _gmm_along_curve(np.concatenate([upper, lower]), component_std)


def swiss_roll(n_components: int = 200, component_std: float = 0.06,
               n_turns: float = 1.75) -> GaussianMixture2D:
    """A spiral ("swiss roll" projection to 2D), GMM-approximated."""
    t = np.linspace(0.2, 1.0, n_components) ** 0.85 * n_turns      # increasing speed
    angle = 2 * np.pi * t
    radius = 0.15 + 1.9 * (t / n_turns)
    points = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return _gmm_along_curve(points, component_std)


DATASETS = {"gaussians8": gaussians8, "moons": moons, "swiss-roll": swiss_roll}


def build_dataset(name: str, **kwargs) -> GaussianMixture2D:
    """Factory by config name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    return DATASETS[name](**kwargs)
