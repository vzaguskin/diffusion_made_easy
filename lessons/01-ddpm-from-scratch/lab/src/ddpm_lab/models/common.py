"""Shared building blocks for the noise-prediction networks ε_θ.

The key shared piece is the **sinusoidal time embedding**: diffusion models must
know *which* timestep ``t`` they are denoising, because the same noisy image ``x_t``
needs very different treatment at ``t = 5`` (almost clean) vs ``t = 999`` (pure
noise). We encode ``t`` with the same sinusoidal positional encoding used by
Transformers, then project it to the network's hidden dimension.

This module is used by both :class:`ddpm_lab.models.mlp.MLP` and
:class:`ddpm_lab.models.unet.UNet`, so the two architectures see the *identical*
encoding of ``t``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding of integer timesteps.

    Maps each integer ``t`` in the batch to a vector of length ``dim`` using
    sine/cosine at geometrically-spaced frequencies, then (optionally) projects
    through an MLP to ``out_dim``.

    The result is added to / concatenated with the image features inside the
    network, conditioning it on the noise level.
    """

    def __init__(self, dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim
        # Two-layer MLP projection after the raw sinusoidal embedding. This is the
        # standard recipe (e.g. Ho et al. 2020, Nichol & Dhariwal 2021).
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.out_dim),
            nn.SiLU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t`` : Tensor[B] integer/float timesteps -> Tensor[B, out_dim]."""
        device = t.device
        half = self.dim // 2
        # Geometrically spaced frequencies: exp(-log(10000) * k / half).
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        # angles[b, k] = t[b] * freqs[k]
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:  # pad to exactly `dim` if odd
            emb = torch.nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class ResidualMLPBlock(nn.Module):
    """A pre-norm residual MLP block used by :class:`MLP`."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))
