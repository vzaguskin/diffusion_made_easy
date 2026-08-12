"""ε_θ as a simple MLP — the transparent baseline.

Purpose
-------
This model exists for **teaching clarity**, not quality. It flattens the image,
mixes in the time embedding, runs a few residual MLP blocks, and reshapes back.

Honest expectation: a *pure* MLP (``use_local_conv=False``) plateaus around
val/loss ≈ 0.7 on MNIST and produces noise-like samples, because it has no
spatial inductive bias — every pixel pair must be learned independently, which
60k images can't teach well enough. This is exactly the lesson: compare it with
the U-Net, which reaches val/loss ≈ 0.03 and sharp digits.

Optional local mixing (``use_local_conv=True``, the default): a single 3×3 conv
before the MLP lets adjacent pixels talk to each other. This is *not* a U-Net —
there is no downsampling, no skip connections, no multi-scale reasoning — but it
gives the MLP the one thing it's missing (locality), so it can reach loss ~0.1–0.2
and show *recognizable-ish* blobs instead of pure noise. Use it to make the
baseline less discouraging while keeping the architecture simple and readable.

Contract (theory.md §15): the model predicts the noise ``eps`` only — *not* the
variance. Output shape == input shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import ResidualMLPBlock, SinusoidalTimeEmbedding


class MLP(nn.Module):
    """Flatten -> (optional local conv) -> time-conditioned residual MLP -> reshape.

    Parameters
    ----------
    image_size : int
        Height/width of the square image (28 for MNIST).
    in_channels : int
        Number of image channels (1 for MNIST).
    hidden_dim : int
        Width of the MLP.
    num_blocks : int
        Number of residual blocks.
    time_embed_dim : int
        Dimension of the sinusoidal time embedding before projection.
    use_local_conv : bool
        If True (default), apply a single 3×3 conv with ``local_channels`` feature
        maps before flattening. This injects a minimal spatial inductive bias so
        the MLP can at least approximate digit shapes (instead of pure noise).
        Set to False for the "pure MLP, no spatial help" ablation.
    local_channels : int
        Number of feature maps for the optional local-conv layer.
    """

    def __init__(
        self,
        image_size: int = 28,
        in_channels: int = 1,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        time_embed_dim: int = 128,
        use_local_conv: bool = True,
        local_channels: int = 32,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.use_local_conv = use_local_conv

        # Optional: one local-mixing conv. Pads to preserve spatial size, then
        # flattens to `local_channels * H * W`. This is the only place spatial
        # structure enters the model.
        if use_local_conv:
            self.local_conv = nn.Sequential(
                nn.Conv2d(in_channels, local_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            flat_dim = local_channels * image_size * image_size
        else:
            self.local_conv = None
            flat_dim = in_channels * image_size * image_size
        self.flat_dim = flat_dim

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim, out_dim=hidden_dim)

        # Project the flattened features into the hidden space and inject time.
        self.input_proj = nn.Linear(flat_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        # Project back to per-pixel predictions.
        self.output_proj = nn.Linear(hidden_dim, in_channels * image_size * image_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``x`` : [B, C, H, W], ``t`` : [B] -> predicted noise [B, C, H, W]."""
        b, c, h, w = x.shape
        temb = self.time_embed(t)  # [B, hidden_dim]

        if self.local_conv is not None:
            feat = self.local_conv(x)            # [B, local_channels, H, W]
        else:
            feat = x                              # [B, C, H, W]
        flat = feat.reshape(b, -1)

        h_feat = self.input_proj(flat) + temb
        for block in self.blocks:
            h_feat = block(h_feat)
        h_feat = self.output_norm(h_feat)
        out = self.output_proj(h_feat)
        return out.reshape(b, c, h, w)
