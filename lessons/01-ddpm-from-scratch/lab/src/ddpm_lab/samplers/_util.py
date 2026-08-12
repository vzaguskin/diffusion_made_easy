"""Small sampler helpers shared by ddpm.py and ddim.py."""

from __future__ import annotations

from typing import Optional

import torch


def randn_with_generator(
    shape: tuple[int, ...],
    generator: Optional[torch.Generator],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample N(0,I) honoring ``generator`` even when it's on a different device.

    ``torch.randn`` requires the generator and the output to share a device. To
    keep callers simple (a CPU generator seeded once is reproducible everywhere),
    we sample on the generator's device and then move the result to ``device``.
    """
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    gen_device = generator.device
    x = torch.randn(shape, generator=generator, device=gen_device, dtype=dtype)
    return x.to(device=device)
