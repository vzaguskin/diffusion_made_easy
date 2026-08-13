"""Draw the U-Net architecture diagrams for the lesson's ``unet.md`` article.

Produces two PNGs in ``../images/``:

1. ``unet-overview.png``  — the full U-shape of *our* U-Net (channels and
   resolutions match ``configs/default.yaml``: base 64, mults (1,2,4), 2 blocks
   per level, attention at the 7×7 bottleneck).
2. ``unet-resblock.png``  — the anatomy of one residual block, including where
   the time embedding enters.

Pure matplotlib (no graphviz/mermaid), so it runs anywhere the venv does::

    uv run python scripts/draw_unet_diagram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LAB_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = LAB_DIR.parent / "images"

# Palette (encoder / decoder / bottleneck / attention / time / io).
C_ENC = "#4C78A8"
C_DEC = "#E4572E"
C_MID = "#54A24B"
C_ATTN = "#B279A2"
C_TIME = "#F2C14E"
C_IO = "#666666"


def box(ax, xy, w, h, label, color, fontsize=9, text_color="white", lw=0):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=color, edgecolor="none", linewidth=lw, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=4, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, start, end, color="#333333", lw=1.6, rad=0.0, style="-|>", zorder=2):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle=style, mutation_scale=14, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder, shrinkA=2, shrinkB=2,
    ))


def draw_overview() -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(-0.7, 3.9)
    ax.axis("off")

    BW, BH = 1.75, 0.72  # encoder/decoder group boxes

    ys = {0: 0.0, 1: 1.35, 2: 2.7}  # level -> y

    # ---------------- input / output ----------------
    box(ax, (0.05, ys[0] + BH / 2 - 0.22), 1.0, 0.5,
        "x_t\n1×28×28", C_IO, fontsize=8.5, text_color="white")
    box(ax, (11.45, ys[0] + BH / 2 - 0.22), 1.0, 0.5,
        "ε_θ\n1×28×28", C_IO, fontsize=8.5, text_color="white")
    arrow(ax, (1.05, ys[0] + BH / 2), (1.45, ys[0] + BH / 2))
    arrow(ax, (11.0, ys[0] + BH / 2), (11.45, ys[0] + BH / 2))

    # in/out convs
    box(ax, (1.45, ys[0] + BH / 2 - 0.22), 0.95, 0.5,
        "in conv\n3×3", C_ENC, fontsize=8)
    arrow(ax, (2.4, ys[0] + BH / 2), (2.8, ys[0] + BH / 2))
    # output norm+conv
    box(ax, (10.0, ys[0] + BH / 2 - 0.22), 1.0, 0.5,
        "GN·SiLU\nout conv", C_DEC, fontsize=8)

    # ---------------- encoder ----------------
    enc_x = 2.8
    enc = {}
    for lvl, (ch, res) in enumerate([(64, "28×28"), (128, "14×14"), (256, "7×7")]):
        enc[lvl] = box(ax, (enc_x, ys[lvl]), BW, BH,
                       f"ResBlock ×2\n{ch}ch, {res}", C_ENC)
        # downsample arrow to the next level
        if lvl < 2:
            arrow(ax, (enc_x + BW / 2, ys[lvl] + BH), (enc_x + BW / 2, ys[lvl + 1]),
                  color=C_ENC, lw=2.0)
            ax.text(enc_x + BW / 2 + 0.12, (ys[lvl] + BH + ys[lvl + 1]) / 2,
                    "down 2×", fontsize=8, color=C_ENC, va="center")
    # encoder -> bottleneck
    arrow(ax, (enc_x + BW, ys[2] + BH / 2), (5.35, ys[2] + BH / 2), lw=2.0)

    # ---------------- bottleneck ----------------
    bx, by = 5.35, ys[2]
    box(ax, (bx, by), 1.0, BH, "Res\nBlock", C_MID)
    box(ax, (bx + 1.15, by), 1.0, BH, "Self-\nAttn", C_ATTN)
    box(ax, (bx + 2.30, by), 1.0, BH, "Res\nBlock", C_MID)
    for x in (bx + 1.0, bx + 2.15):
        arrow(ax, (x, by + BH / 2), (x + 0.15, by + BH / 2), lw=2.0)
    ax.text(bx + 2.25, by + BH + 0.14, "bottleneck: 256ch, 7×7 — global reasoning is cheap here",
            fontsize=8, ha="center", color=C_MID)

    # ---------------- decoder ----------------
    dec_x = 8.05
    arrow(ax, (bx + 3.30, by + BH / 2), (dec_x, by + BH / 2), lw=2.0)
    dec = {}
    for lvl, (ch, res) in enumerate([(256, "7×7"), (128, "14×14"), (64, "28×28")]):
        y = ys[2 - lvl]
        dec[2 - lvl] = box(ax, (dec_x, y), BW, BH,
                           f"ResBlock ×2\n{ch}ch, {res}", C_DEC)
        if lvl < 2:
            # upsample arrow from deeper level to this one
            arrow(ax, (dec_x + BW / 2, y + 0.0), (dec_x + BW / 2, ys[2 - lvl - 1] + BH),
                  color=C_DEC, lw=2.0)
            ax.text(dec_x + BW / 2 + 0.12, (y + ys[2 - lvl - 1] + BH) / 2,
                    "up 2×", fontsize=8, color=C_DEC, va="center")
    arrow(ax, (dec_x + BW, ys[0] + BH / 2), (10.0, ys[0] + BH / 2), lw=2.0)

    # ---------------- skip connections ----------------
    for lvl, (ch, res) in enumerate([(64, "28×28"), (128, "14×14"), (256, "7×7")]):
        x0 = enc_x + BW
        x1 = dec_x
        y = ys[lvl] + BH / 2
        rad = {0: -0.55, 1: -0.42, 2: -0.30}[lvl]
        arrow(ax, (x0, y), (x1, y), color="#B695C0", lw=1.8, rad=rad, zorder=1)
        ax.text((x0 + x1) / 2, y + (0.75 if lvl == 0 else 0.62 if lvl == 1 else 0.50),
                f"2 skips ({ch}ch, {res})", fontsize=8, color="#7B5687", ha="center")

    # ---------------- time embedding ----------------
    t_y = -0.62
    box(ax, (4.6, t_y), 3.4, 0.5,
        "t → sinusoidal embedding → MLP  (shared, fed into every ResBlock)",
        C_TIME, fontsize=8.5, text_color="#333333")
    for x in (3.65, 8.9):
        arrow(ax, (x, t_y + 0.5), (x, 0.0), color=C_TIME, lw=1.4,
              rad=0.0, style="-|>")

    ax.set_title("ε_θ U-Net (our config: base 64, mults (1,2,4), 2 blocks/level) — 12.2M params",
                 fontsize=11, pad=10)
    fig.tight_layout()
    out = OUT_DIR / "unet-overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_resblock() -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-1.1, 2.15)
    ax.axis("off")

    y = 0.8
    bw, bh = 1.05, 0.62
    mid = y + bh / 2

    # Input label.
    ax.text(0.22, mid, "x\n[C_in,H,W]", fontsize=9, ha="center", va="center")

    # Main path boxes (left to right, with gaps where the ⊕ nodes sit).
    #   GN1 -> SiLU1 -> Conv1 -> (⊕ time) -> GN2 -> SiLU2 -> Conv2 -> (⊕ residual) -> out
    xs = {
        "gn1": 0.75, "silu1": 2.00, "conv1": 3.25,     # -> ends at 4.30
        # time ⊕ at 4.70
        "gn2": 5.25, "silu2": 6.50, "conv2": 7.75,     # -> ends at 8.80
        # residual ⊕ at 9.20
    }
    box(ax, (xs["gn1"], y), bw, bh, "GroupNorm", "#88A6C9", fontsize=8.5)
    box(ax, (xs["silu1"], y), bw, bh, "SiLU", "#A8C5E2", fontsize=9, text_color="#1a1a1a")
    box(ax, (xs["conv1"], y), bw, bh, "Conv 3×3\nC_in→C_out", "#4C78A8", fontsize=8.5)
    box(ax, (xs["gn2"], y), bw, bh, "GroupNorm", "#88A6C9", fontsize=8.5)
    box(ax, (xs["silu2"], y), bw, bh, "SiLU", "#A8C5E2", fontsize=9, text_color="#1a1a1a")
    box(ax, (xs["conv2"], y), bw, bh, "Conv 3×3\nC_out→C_out", "#4C78A8", fontsize=8.5)

    # Arrows along the main path.
    arrow(ax, (0.42, mid), (xs["gn1"], mid), lw=1.6)
    arrow(ax, (xs["gn1"] + bw, mid), (xs["silu1"], mid), lw=1.6)
    arrow(ax, (xs["silu1"] + bw, mid), (xs["conv1"], mid), lw=1.6)
    arrow(ax, (xs["conv1"] + bw, mid), (4.57, mid), lw=1.6)   # conv1 -> ⊕time
    arrow(ax, (4.83, mid), (xs["gn2"], mid), lw=1.6)          # ⊕time -> GN2
    arrow(ax, (xs["gn2"] + bw, mid), (xs["silu2"], mid), lw=1.6)
    arrow(ax, (xs["silu2"] + bw, mid), (xs["conv2"], mid), lw=1.6)
    arrow(ax, (xs["conv2"] + bw, mid), (9.07, mid), lw=1.6)   # conv2 -> ⊕res
    arrow(ax, (9.33, mid), (9.85, mid), lw=1.6)               # ⊕res -> out

    # ⊕ time-injection node (after the first conv, before GN2).
    ax.add_patch(plt.Circle((4.70, mid), 0.13, color="#F2C14E", zorder=5))
    ax.text(4.70, mid, "+", ha="center", va="center", fontsize=11, zorder=6)

    # ⊕ residual node.
    ax.add_patch(plt.Circle((9.20, mid), 0.13, color="#54A24B", zorder=5))
    ax.text(9.20, mid, "+", ha="center", va="center", fontsize=11, color="white", zorder=6)

    # Residual skip: from the input, up and over the whole block, into ⊕res.
    sy = y + bh + 0.5
    ax.add_patch(FancyArrowPatch(
        (0.22, y + bh), (0.22, sy), arrowstyle="-", lw=1.6, color="#54A24B", zorder=2))
    ax.add_patch(FancyArrowPatch(
        (0.22, sy), (9.20, sy), arrowstyle="-", lw=1.6, color="#54A24B", zorder=2))
    arrow(ax, (9.20, sy), (9.20, mid + 0.13), lw=1.6, color="#54A24B")
    ax.text(4.7, sy + 0.15, "residual skip (identity if C_in==C_out, else 1×1 conv)",
            fontsize=8.5, color="#2F6B2A", ha="center")

    # Output label.
    ax.text(10.30, mid, "out\n[C_out,H,W]", fontsize=9, ha="center", va="center")

    # Time branch from below into ⊕time.
    t_y = -0.75
    box(ax, (1.6, t_y), 6.2, 0.5,
        "t  →  sinusoidal embedding  →  Linear(time_dim → C_out)  →  added as a bias per channel",
        C_TIME, fontsize=8.5, text_color="#333333")
    arrow(ax, (4.70, t_y + 0.5), (4.70, mid - 0.13), color="#B8860B", lw=1.6)

    ax.set_title("One ResidualBlock — pre-norm convs, time injected once, residual around the whole block",
                 fontsize=10.5, pad=8)
    fig.tight_layout()
    out = OUT_DIR / "unet-resblock.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = draw_overview()
    print(f"wrote {p1}")
    p2 = draw_resblock()
    print(f"wrote {p2}")
