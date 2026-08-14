"""Per-noise-level loss of the trained VE/VP models — why the scalar losses
are not comparable (see README "Известные упрощения").

The headline ε-MSE of each branch averages a *different mixture* of noise
levels: VP's linear-β schedule spends ~half its timesteps at noise_std ≈ 1
(x_t ≈ ε — trivial, loss ~1e-3), while VE's geometric ladder spends half its
levels at σ < 0.2 (ε nearly invisible under the signal — hard, loss ~0.1).
This script measures the loss per level and plots both curves against the
noise std on a shared axis, so the branches are compared at matched SNR.

Usage:
    uv run python scripts/eval_per_level.py            # → runs/ve_vs_vp/
    uv run python scripts/eval_per_level.py mnist_ve_vp.ve_long_epochs=5
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from score_lab.config import load_config
from score_lab.mnist_unet import build_mnist_unet
from score_lab.mnist_ve_vp import VESchedule, VPSchedule, mnist_loaders


@torch.no_grad()
def _per_level(model, sched, xb, idxs, chunk: int = 512) -> dict[int, float]:
    out = {}
    for i in idxs:
        se, n = 0.0, 0
        for s in range(0, xb.shape[0], chunk):
            x0 = xb[s: s + chunk]
            t = torch.full((x0.shape[0],), i, device=x0.device, dtype=torch.long)
            eps = torch.randn_like(x0)
            xt = sched.forward(x0, t, eps)
            se += ((model(xt, t) - eps) ** 2).sum().item()
            n += eps.numel()
        out[i] = se / n
    return out


def main() -> None:
    cfg = load_config()
    section = cfg.mnist_ve_vp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.paths.run_dir).parent / "ve_vs_vp"
    ve_long_name = "model_ve_long.pt" if int(section.ve_long_epochs) > int(section.epochs) else "model_ve.pt"

    ve_sched = VESchedule(**dict(section.ve))
    vp_sched = VPSchedule(**dict(section.vp))

    ve = build_mnist_unet(cfg).to(device).eval()
    ve.load_state_dict(torch.load(out_dir / ve_long_name, map_location=device, weights_only=True))
    vp = build_mnist_unet(cfg).to(device).eval()
    vp.load_state_dict(torch.load(out_dir / "model_vp.pt", map_location=device, weights_only=True))

    loader = mnist_loaders(str(Path(__file__).resolve().parents[1] / "data"),
                           int(section.batch_size))
    xb = torch.cat([x for x, _ in loader])[: 256 * 16].to(device)

    n_ticks = 25
    ve_idx = list(range(0, ve_sched.n_levels, ve_sched.n_levels // n_ticks))
    vp_idx = list(range(0, vp_sched.num_timesteps, vp_sched.num_timesteps // n_ticks))
    ve_l = _per_level(ve, ve_sched, xb, ve_idx)
    vp_l = _per_level(vp, vp_sched, xb, vp_idx)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot([float(ve_sched.sigmas[i]) for i in ve_idx], [ve_l[i] for i in ve_idx],
            "o-", ms=3, color="#d62728", label="VE (long budget)")
    ax.plot([float(vp_sched.sigma(torch.tensor([i]))[0]) for i in vp_idx],
            [vp_l[i] for i in vp_idx], "o-", ms=3, color="#1f77b4", label="VP")
    ax.set_xscale("log")
    ax.set_xlabel("noise std actually added  (VE: σ,  VP: √(1−ᾱ))")
    ax.set_ylabel("ε-MSE at that level")
    ax.set_title("Per-level loss at matched SNR — not the headline scalar!")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.savefig(out_dir / "per_level_loss.png", bbox_inches="tight")
    plt.close(fig)

    print(f"per-level curves → {out_dir}/per_level_loss.png")
    print("matched-SNR table (VE σ ≈ VP √(1−ᾱ)):")
    for target in [0.05, 0.1, 0.2, 0.4, 0.8]:
        vi = min(ve_idx, key=lambda k: abs(float(ve_sched.sigmas[k]) - target))
        pi = min(vp_idx, key=lambda k: abs(float(vp_sched.sigma(torch.tensor([k]))[0]) - target))
        print(f"  noise_std~{target:5.2f}:  VE={ve_l[vi]:.4f}  VP={vp_l[pi]:.4f}")


if __name__ == "__main__":
    main()
