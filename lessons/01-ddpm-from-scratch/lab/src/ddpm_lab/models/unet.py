"""ε_θ as a small U-Net — the canonical diffusion architecture.

This is a deliberately *small* U-Net so it fits in 6GB VRAM on MNIST
(28×28, grayscale). It has the ingredients every diffusion U-Net has:

* **Encoder/decoder** with downsampling and upsampling, and skip connections
  ("U" shape) that let gradients and detail flow between resolutions.
* **Time conditioning**: the sinusoidal embedding of ``t`` is projected and added
  into every residual block (so every layer knows the noise level).
* **Self-attention at the bottleneck** (the lowest resolution), which is cheap on
  7×7 feature maps and helps the model reason globally.

Contract (theory.md §15): the network predicts the noise ``eps`` only — *not* the
variance. Output shape == input shape. There is one ``t`` per sample (``t`` shape
``[B]``), as required by theory.md §16 (training algorithm, step 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SinusoidalTimeEmbedding


@dataclass
class UNetConfig:
    """Dimensions of the U-Net. Defaults are sized for 6GB VRAM on MNIST."""

    in_channels: int = 1
    out_channels: int = 1  # we predict eps, same shape & channels as input
    base_channels: int = 64  # width at the top resolution
    channel_mults: tuple[int, ...] = (1, 2, 4)  # one per resolution level
    num_blocks: int = 2  # residual blocks per level
    time_embed_dim: int = 256
    dropout: float = 0.0
    # NOTE: self-attention lives at the bottleneck only (7×7 here) — that is
    # hard-coded in UNet.__init__ (self.mid), see unet.md §6 for the rationale.


def _conv(c_in: int, c_out: int) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)


class SelfAttention(nn.Module):
    """Multi-head self-attention over a small feature map (used at the bottleneck)."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)  # [B, 3C, H, W]
        # Split into per-head q, k, v. Each is [B, heads, c_head, H*W].
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [B, heads, c_head, HW]
        # Attention: dots over the c_head dim, matrix over the spatial (HW) dim.
        # attn[b, h, i, j] = sum_d q[b,h,d,i] * k[b,h,d,j]  -> [B, heads, HW, HW]
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        # out[b, h, d, i] = sum_j attn[b,h,i,j] * v[b,h,d,j]  -> [B, heads, c_head, HW]
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


class ResidualBlock(nn.Module):
    """Residual conv block with time-embedding injection (pre-norm GroupNorm)."""

    def __init__(self, c_in: int, c_out: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = _conv(c_in, c_out)
        self.time_mlp = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = _conv(c_out, c_out)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(self.time_mlp(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """A small U-Net for MNIST-scale images.

    The input is the noisy image ``x_t`` and the timestep ``t``; the output is the
    predicted noise ``eps_theta(x_t, t)`` (theory.md §15).
    """

    def __init__(self, cfg: UNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or UNetConfig()
        c0 = self.cfg.base_channels
        mults = self.cfg.channel_mults
        time_dim = self.cfg.time_embed_dim

        # --- Time embedding (shared) ---------------------------------------
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim // 2, out_dim=time_dim),
        )

        # --- Input convolution ---------------------------------------------
        self.in_conv = _conv(self.cfg.in_channels, c0)

        # --- Encoder -------------------------------------------------------
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        channels = c0
        for level, mult in enumerate(mults):
            out_channels = c0 * mult
            for _ in range(self.cfg.num_blocks):
                self.encoder_blocks.append(ResidualBlock(channels, out_channels, time_dim, self.cfg.dropout))
                channels = out_channels
            # downsample between levels (but not after the last level)
            if level < len(mults) - 1:
                self.encoder_downs.append(Downsample(channels))
            else:
                self.encoder_downs.append(nn.Identity())

        # --- Bottleneck -----------------------------------------------------
        self.mid = nn.ModuleList([
            ResidualBlock(channels, channels, time_dim, self.cfg.dropout),
            SelfAttention(channels),
            ResidualBlock(channels, channels, time_dim, self.cfg.dropout),
        ])

        # --- Decoder --------------------------------------------------------
        # Mirrors the encoder level-by-level, in reverse. At each level we process
        # num_blocks decoder blocks; each one concatenates one skip (with that
        # level's channel count) onto the current features.
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()  # one upsample between consecutive levels
        rev_mults = list(reversed(mults))
        # `channels` currently = bottleneck width = c0 * mults[-1].
        for level, mult in enumerate(rev_mults):
            out_channels = c0 * mult  # the skip at this level has this many channels
            for _ in range(self.cfg.num_blocks):
                self.decoder_blocks.append(
                    ResidualBlock(channels + out_channels, out_channels, time_dim, self.cfg.dropout)
                )
                channels = out_channels
            # Upsample between this level and the next (finer) one. The last decoder
            # level (top of the U) needs no upsample.
            if level < len(rev_mults) - 1:
                self.decoder_ups.append(Upsample(channels))

        # --- Output ---------------------------------------------------------
        self.out_norm = nn.GroupNorm(8, c0)
        self.out_conv = _conv(c0, self.cfg.out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``x`` : [B, C, H, W], ``t`` : [B] -> predicted noise [B, C, H, W]."""
        t_emb = self.time_embed(t)  # [B, time_dim]

        # --- Encoder: walk levels, save one skip per residual block ----------
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        idx = 0  # running index into self.encoder_blocks
        for level in range(len(self.cfg.channel_mults)):
            for _ in range(self.cfg.num_blocks):
                h = self.encoder_blocks[idx](h, t_emb)
                idx += 1
                skips.append(h)  # skip lives at this level's resolution
            h = self.encoder_downs[level](h)

        # Bottleneck
        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        # --- Decoder: mirror the encoder level-by-level.
        # At each level: (optionally upsample to that level's resolution), then
        # process num_blocks decoder blocks, each concatenating one skip from the
        # matching encoder level (popped in reverse = LIFO).
        idx = 0
        num_levels = len(self.cfg.channel_mults)
        for level in range(num_levels):
            # Upsample at the *start* of every level except the deepest (level 0
            # here = deepest, since we reversed the mults). The decoder_ups list is
            # indexed by decoder level; its last entry is Identity (no upsample).
            if level > 0:
                h = self.decoder_ups[level - 1](h)
            for _ in range(self.cfg.num_blocks):
                skip = skips.pop()  # same resolution & channel count as h now
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[idx](h, t_emb)
                idx += 1

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
