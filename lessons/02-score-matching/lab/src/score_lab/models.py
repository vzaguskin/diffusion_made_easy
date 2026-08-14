"""The score network s_θ(x, σ) — a small MLP conditioned on the noise level.

One network serves *all* levels of the σ-ladder (theory.md §8): the noise level
is encoded with Fourier features of log σ — the same trick as time-embedding in
lesson 01, but for a *continuous* σ that spans orders of magnitude (1.0 → 0.02),
hence the log scale.

Contract (theory.md §2, §7): the network approximates ∇ log p_σ(x) — the score
of the σ-smoothed density. Note it predicts the score *directly* (units 1/length);
the DSM loss in ``dsm.py`` supplies the right regression target.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SigmaFourierFeatures(nn.Module):
    """Encode a continuous σ > 0 as sin/cos features of log σ.

    log σ is the natural coordinate: the ladder is geometric, so equal steps in
    log σ are equal steps of "smoothing" perception.
    """

    def __init__(self, n_features: int = 16, max_freq: float = 8.0) -> None:
        super().__init__()
        self.n_features = n_features
        # Geometric frequencies from 1 to max_freq (per unit of log σ).
        n_half = max(1, n_features // 2)
        freqs = torch.logspace(0.0, math.log10(max_freq), n_half)
        self.register_buffer("freqs", freqs)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """σ : [B, 1] (positive) -> features [B, n_features]."""
        log_sigma = torch.log(sigma.clamp_min(1e-8))
        args = log_sigma * self.freqs.view(1, -1)                 # [B, n_features/2]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ScoreMLP(nn.Module):
    """MLP: (x, fourier(log σ)) -> score estimate [B, 2]."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 4,
        sigma_features: int = 16,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.sigma_embed = SigmaFourierFeatures(n_features=sigma_features)
        in_dim = 2 + sigma_features

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            if use_residual:
                layers.append(ResidualLinearBlock(hidden_dim))
            else:
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x : [B, 2], sigma : [B] or [B, 1] -> score [B, 2]."""
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(1)
        feats = torch.cat([x, self.sigma_embed(sigma)], dim=1)
        return self.net(feats)


class ResidualLinearBlock(nn.Module):
    """Pre-norm residual MLP block (the MLP cousin of lesson 01's ResidualBlock)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


def build_score_model(cfg) -> ScoreMLP:
    """Factory from the ``model`` config section."""
    m = cfg.model
    return ScoreMLP(
        hidden_dim=int(getattr(m, "hidden_dim", 256)),
        n_layers=int(getattr(m, "n_layers", 4)),
        sigma_features=int(getattr(m, "sigma_features", 16)),
        use_residual=bool(getattr(m, "use_residual", True)),
    )
