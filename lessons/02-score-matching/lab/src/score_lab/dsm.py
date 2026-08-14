"""Denoising score matching: the loss that makes the score learnable.

Theory reference: ``theory.md`` §7 «Denoising score matching: трюк с
зашумлением» and §8 «Многоуровневый шум».

The problem (§5): we can't train ``‖s_θ(x) − ∇log p(x)‖²`` directly — the true
score is unknown. The trick (§7): for the *smoothed* density
``q_σ(x̃) = ∫ q(x) N(x̃; x, σ²I) dx`` the score can be learned by regression on
noise targets:

    L(θ; σ) = E_{x~data, ε~N(0,I)} [ ‖ s_θ(x + σ·ε, σ) + ε/σ ‖² ]

Two things to internalize (and easy to get wrong):

1. **The target is −ε/σ** (with a PLUS in front of it inside the norm).
   Why: for ``x̃ = x + σε``, we have ``∇_{x̃} log q(x̃ | x) = −(x̃ − x)/σ² = −ε/σ``.
   The score of the *conditional* is the regression target; DSM's theorem says
   minimizing this also matches the score of the *smoothed marginal*.

2. **VE noising** (§10): ``x̃ = x + σ·ε`` — purely additive. The signal is not
   scaled (contrast with VP's ``√(1−β)x + √β ε`` from lesson 01).

Multi-level (§8): training samples σ uniformly from the ladder each batch, so
one network learns all smoothing levels simultaneously — the same network later
used level-by-level by annealed Langevin.
"""

from __future__ import annotations

import torch


def dsm_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the DSM loss of theory.md §7 for a batch at one noise level.

    Parameters
    ----------
    model : nn.Module
        Score network ``s_θ(x, σ)``.
    x : Tensor[B, 2]
        Clean data batch.
    sigma : float or Tensor[B] or Tensor[B, 1]
        Noise level(s). A scalar float applies to the whole batch; a tensor
        gives each sample its own level (the train loop uses ladder sampling).
    generator : optional
        For reproducible ε.

    Returns
    -------
    (loss, eps)
        The scalar MSE loss and the sampled noise (for logging/tests).
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma))
    if sigma.ndim == 0:
        sigma = sigma.expand(x.shape[0]).unsqueeze(1)   # [B, 1]
    elif sigma.ndim == 1:
        sigma = sigma.unsqueeze(1)                       # [B, 1]

    eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    x_noisy = x + sigma * eps                            # VE: additive (§10)
    target = -eps / sigma                                # = ∇ log q(x̃|x)  (§7)
    pred = model(x_noisy, sigma)
    loss = torch.mean((pred - target) ** 2)
    return loss, eps


def sample_sigma_level(sigmas: torch.Tensor, batch_size: int, generator=None) -> torch.Tensor:
    """Pick a σ level uniformly from the ladder for each sample (§8)."""
    idx = torch.randint(0, len(sigmas), (batch_size,), generator=generator)
    return sigmas[idx]
