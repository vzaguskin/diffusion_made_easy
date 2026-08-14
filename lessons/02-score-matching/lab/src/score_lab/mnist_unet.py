"""Mini U-Net for MNIST — adapted copy of lesson 01's UNet (lab 1, §15).

Deliberately smaller (base_channels 32 by default) so the VE/VP comparison
trains both branches in ~an hour on a laptop. The architecture is *identical*
for both branches (spec: ve-vp-comparison): what differs is only how the
conditioning index ``t`` is interpreted — a σ-ladder level (VE) or a β-schedule
timestep (VP). Both are plain integers fed to the same sinusoidal embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniUNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    channel_mults: tuple[int, ...] = (1, 2, 4)
    num_blocks: int = 2
    time_embed_dim: int = 128
    dropout: float = 0.0


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding of the (integer) conditioning index t → [B, dim]."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), nn.SiLU(),
                                 nn.Linear(out_dim, out_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0))
                          * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


def _conv(c_in: int, c_out: int) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)


class ResidualBlock(nn.Module):
    """Residual conv block with time-embedding injection (lesson-01 recipe)."""

    def __init__(self, c_in: int, c_out: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = _conv(c_in, c_out)
        self.time_proj = nn.Linear(time_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = _conv(c_out, c_out)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention at the bottleneck (7×7 maps — cheap)."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads,
                                             c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        out = torch.einsum("bhij,bhdj->bhdi", attn.softmax(dim=-1), v)
        return x + self.proj(out.reshape(b, c, h, w))


class MiniUNet(nn.Module):
    """ε_θ(x_t, t) for MNIST 28×28 — same class for the VE and VP branches."""

    def __init__(self, cfg: MiniUNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MiniUNetConfig()
        c0, mults = self.cfg.base_channels, self.cfg.channel_mults
        tdim = self.cfg.time_embed_dim
        self.time_embed = SinusoidalEmbedding(tdim // 2, tdim)
        self.in_conv = _conv(self.cfg.in_channels, c0)

        self.encoder_blocks, self.encoder_downs = nn.ModuleList(), nn.ModuleList()
        channels = c0
        for level, mult in enumerate(mults):
            for _ in range(self.cfg.num_blocks):
                self.encoder_blocks.append(ResidualBlock(channels, c0 * mult, tdim,
                                                         self.cfg.dropout))
                channels = c0 * mult
            self.encoder_downs.append(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1)
                if level < len(mults) - 1 else nn.Identity()
            )

        self.mid = nn.ModuleList([
            ResidualBlock(channels, channels, tdim, self.cfg.dropout),
            SelfAttention(channels),
            ResidualBlock(channels, channels, tdim, self.cfg.dropout),
        ])

        self.decoder_blocks, self.decoder_ups = nn.ModuleList(), nn.ModuleList()
        rev = list(reversed(mults))
        for level, mult in enumerate(rev):
            for _ in range(self.cfg.num_blocks):
                self.decoder_blocks.append(
                    ResidualBlock(channels + c0 * mult, c0 * mult, tdim, self.cfg.dropout))
                channels = c0 * mult
            if level < len(rev) - 1:
                self.decoder_ups.append(
                    nn.Sequential(nn.Upsample(scale_factor=2.0, mode="nearest"),
                                  _conv(channels, channels)))

        self.out_norm = nn.GroupNorm(8, c0)
        self.out_conv = _conv(c0, self.cfg.out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x : [B, 1, 28, 28], t : [B] -> predicted ε, same shape as x."""
        t_emb = self.time_embed(t)
        h = self.in_conv(x)
        skips = []
        idx = 0
        for level in range(len(self.cfg.channel_mults)):
            for _ in range(self.cfg.num_blocks):
                h = self.encoder_blocks[idx](h, t_emb)
                idx += 1
                skips.append(h)
            h = self.encoder_downs[level](h)
        for layer in self.mid:
            h = layer(h, t_emb) if isinstance(layer, ResidualBlock) else layer(h)
        idx = 0
        for level in range(len(self.cfg.channel_mults)):
            if level > 0:
                h = self.decoder_ups[level - 1](h)
            for _ in range(self.cfg.num_blocks):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.decoder_blocks[idx](h, t_emb)
                idx += 1
        return self.out_conv(F.silu(self.out_norm(h)))


def build_mnist_unet(cfg) -> MiniUNet:
    """Factory from the ``mnist_ve_vp.model`` config section."""
    m = cfg.mnist_ve_vp.model
    return MiniUNet(MiniUNetConfig(
        base_channels=int(getattr(m, "base_channels", 32)),
        channel_mults=tuple(getattr(m, "channel_mults", [1, 2, 4])),
        num_blocks=int(getattr(m, "num_blocks", 2)),
    ))
