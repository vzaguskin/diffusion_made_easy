"""Samplers for generation: stochastic DDPM (§17) and deterministic DDIM (§18).

Both expose the same signature::

    sample(model, core, shape, num_steps=None, *, generator=None, device=None) -> Tensor

so they are interchangeable in callbacks and in ``scripts/sample.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from ..core import DiffusionCore
from . import ddim, ddpm
from ._util import randn_with_generator

# A sampler function follows the signature documented above.
Sampler = Callable[..., torch.Tensor]


def build_sampler(name: str) -> Sampler:
    """Return the sampler function for ``name`` (``"ddpm"`` or ``"ddim"``)."""
    if name == "ddpm":
        return ddpm.sample
    if name == "ddim":
        return ddim.sample
    raise ValueError(f"Unknown sampler '{name}'. Expected 'ddpm' or 'ddim'.")


__all__ = ["ddpm", "ddim", "build_sampler", "Sampler", "randn_with_generator"]

