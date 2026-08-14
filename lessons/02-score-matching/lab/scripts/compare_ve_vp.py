"""VE vs VP on MNIST with an equal budget — theory.md §9–§10.

Trains two MiniUNets (identical architecture and hyperparameters):
* VE branch — geometric σ ladder 80 → 0.01, additive forward noising;
* VP branch — linear β as in lesson 01.

Then samples side-by-side grids *from the same starting noise* and saves the
loss-curve comparison. Artifacts → ``runs/ve_vs_vp/``.

Usage:
    uv run python scripts/compare_ve_vp.py [key=value ...]
    uv run python scripts/compare_ve_vp.py mnist_ve_vp.epochs=1   # quick check
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
from score_lab.mnist_ve_vp import (
    VESchedule,
    VPSchedule,
    mnist_loaders,
    sample_ve,
    sample_vp,
    save_sample_grid,
    train_eps_model,
)


def main() -> None:
    cfg = load_config()
    section = cfg.mnist_ve_vp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.paths.run_dir).parent / "ve_vs_vp"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  epochs={section.epochs}  → {out_dir}/")

    loader = mnist_loaders(str(Path(__file__).resolve().parents[1] / "data"),
                           int(section.batch_size))

    ve_sched = VESchedule(**dict(section.ve))
    vp_sched = VPSchedule(**dict(section.vp))

    # --- Train both branches with the identical budget ----------------------
    print("training VE branch (σ: %.2f → %.3f in %d levels)..."
          % (ve_sched.sigma_max, ve_sched.sigma_min, ve_sched.n_levels))
    ve_model = build_mnist_unet(cfg)
    ve_losses = train_eps_model(ve_model, loader, ve_sched,
                                epochs=int(section.epochs), lr=float(section.lr),
                                device=device, seed=int(section.seed),
                                log_csv=out_dir / "loss_ve.csv")
    torch.save(ve_model.state_dict(), out_dir / "model_ve.pt")

    print("training VP branch (β linear, T=%d)..." % vp_sched.num_timesteps)
    vp_model = build_mnist_unet(cfg)
    vp_losses = train_eps_model(vp_model, loader, vp_sched,
                                epochs=int(section.epochs), lr=float(section.lr),
                                device=device, seed=int(section.seed),
                                log_csv=out_dir / "loss_vp.csv")
    torch.save(vp_model.state_dict(), out_dir / "model_vp.pt")

    # --- Side-by-side samples from the SAME starting noise (spec §ve-vp) ----
    n = int(section.n_samples)
    gen = torch.Generator(device=device).manual_seed(int(section.seed))
    x_start = torch.randn(n, 1, 28, 28, generator=gen, device=device)
    torch.save(x_start, out_dir / "x_start.pt")

    x_ve = sample_ve(ve_model, ve_sched, (n, 1, 28, 28),
                     generator=torch.Generator(device=device)
                     .manual_seed(int(section.seed) + 1), device=device)
    x_vp = sample_vp(vp_model, vp_sched, (n, 1, 28, 28),
                     generator=torch.Generator(device=device)
                     .manual_seed(int(section.seed) + 1), device=device)
    save_sample_grid(x_ve, out_dir / "samples_ve.png", "VE samples")
    save_sample_grid(x_vp, out_dir / "samples_vp.png", "VP samples")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    show = lambda ax, x, ttl: ax.imshow(
        (x[:16].detach().cpu() * 0.3081 + 0.1307).clamp(0, 1)
        .reshape(2, 8, 28, 28).permute(0, 2, 1, 3).reshape(56, 224), cmap="gray_r")
    show(axes[0], x_ve, "VE"); axes[0].set_title("VE", fontsize=10); axes[0].axis("off")
    show(axes[1], x_vp, "VP"); axes[1].set_title("VP", fontsize=10); axes[1].axis("off")
    fig.suptitle("Same starting noise: VE (left) vs VP (right)", y=0.98)
    fig.savefig(out_dir / "samples_side_by_side.png", bbox_inches="tight")
    plt.close(fig)

    # --- Loss curves ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(range(1, len(ve_losses) + 1), ve_losses, label="VE (ε-MSE)",
            color="#d62728", lw=1.5)
    ax.plot(range(1, len(vp_losses) + 1), vp_losses, label="VP (ε-MSE)",
            color="#1f77b4", lw=1.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Equal-budget training: VE vs VP")
    fig.savefig(out_dir / "loss_curves.png", bbox_inches="tight")
    plt.close(fig)

    print(f"artifacts → {out_dir}/")


if __name__ == "__main__":
    main()
