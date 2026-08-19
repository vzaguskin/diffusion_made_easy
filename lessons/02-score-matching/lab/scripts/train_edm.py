"""Train the EDM-preconditioned VE branch and benchmark it — theory.md §17.

Trains MiniUNet wrapped in EDM preconditioning (same 20-epoch budget as the
naive ve_long branch), then produces ``runs/edm/``:

* ``loss_curve.png``         — EDM training loss;
* ``per_level_vs_naive.png`` — denoiser MSE per sigma, EDM vs naive VE
  (naive denoiser: x − σ·ε̂ from runs/ve_vs_vp/model_ve_long.pt) on a shared
  sigma axis — shows the balance claim of §17;
* ``budget_strip.png``       — Heun samples at the same 40–1000 NFE budgets as
  the naive branch's strip, plus the naive 600-NFE optimum for reference;
* ``speckle.csv``            — speckle metric, EDM vs naive.

Usage:
    uv run python scripts/train_edm.py            # full run (~16 min on GPU)
    uv run python scripts/train_edm.py edm.epochs=1   # smoke
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from score_lab.config import load_config
from score_lab.edm import EDMModel, edm_heun_sample, edm_loss
from score_lab.mnist_unet import build_mnist_unet
from score_lab.mnist_ve_vp import (
    MNIST_MEAN,
    MNIST_STD,
    VESchedule,
    mnist_loaders,
    sample_ve,
)

SIGMA_AXIS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0, 5.0]


def _unnorm(x):
    return (x * MNIST_STD + MNIST_MEAN).clamp(0, 1)


def speckle(x):
    img = _unnorm(x)
    bright = (img > 0.6).float()
    neigh = F.max_pool2d(bright, 3, stride=1, padding=1) - bright
    return float((bright * (neigh == 0).float()).sum()) / bright.numel()


def _show(ax, x, title):
    img = (_unnorm(x[:16].detach().cpu())
           .reshape(2, 8, 28, 28).permute(0, 2, 1, 3).reshape(56, 224))
    ax.imshow(img, cmap="gray_r")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    cfg = load_config()
    section = cfg.edm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.paths.run_dir).parent / "edm"
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs, seed = int(section.epochs), int(section.seed)
    print(f"device={device}  epochs={epochs}  → {out_dir}/")

    loader = mnist_loaders(str(Path(__file__).resolve().parents[1] / "data"),
                           int(section.batch_size))
    torch.manual_seed(seed)
    model = EDMModel(build_mnist_unet(cfg), float(section.sigma_data)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(section.lr))

    losses = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for x0, _ in loader:
            x0 = x0.to(device)
            loss = edm_loss(model, x0, float(section.p_mean), float(section.p_std))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); n += 1
        losses.append(running / max(n, 1))
        print(f"  [epoch {epoch+1}/{epochs}] loss={losses[-1]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    torch.save(model.net.state_dict(), out_dir / "model_edm.pt")

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(range(1, len(losses) + 1), losses, color="#2ca02c", lw=1.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("EDM loss"); ax.grid(alpha=0.3)
    fig.savefig(out_dir / "loss_curve.png", bbox_inches="tight"); plt.close(fig)

    # --- per-level denoiser MSE vs the naive VE branch -----------------------
    naive_path = Path(cfg.paths.run_dir).parent / "ve_vs_vp" / "model_ve_long.pt"
    xb = torch.cat([x for x, _ in loader])[:2048].to(device)
    ve_sched = VESchedule(**dict(cfg.mnist_ve_vp.ve))
    naive = build_mnist_unet(cfg).to(device).eval()
    naive.load_state_dict(torch.load(naive_path, map_location=device,
                                     weights_only=True))

    def idx_for_sigma(s: float) -> int:
        return int((ve_sched.sigmas - s).abs().argmin())

    edm_curve, naive_curve = [], []
    with torch.no_grad():
        for s in SIGMA_AXIS:
            sig = torch.full((xb.shape[0],), s, device=device)
            eps = torch.randn_like(xb)
            xt = xb + sig.reshape(-1, 1, 1, 1) * eps
            resid = model(xt, sig) - xb
            edm_curve.append(float(resid.pow(2).mean()))
            tb = torch.full((xb.shape[0],), idx_for_sigma(s), device=device,
                            dtype=torch.long)
            x0_hat = xt - s * naive(xt, tb)       # naive denoiser: x − σ·ε̂
            naive_curve.append(float((x0_hat - xb).pow(2).mean()))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(SIGMA_AXIS, edm_curve, "o-", ms=3, color="#2ca02c", label="EDM")
    ax.plot(SIGMA_AXIS, naive_curve, "o-", ms=3, color="#d62728", label="naive VE")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("σ"); ax.set_ylabel("denoiser MSE  ‖D(x+σε,σ) − x₀‖²")
    ax.set_title("Per-level denoiser error (same 20-epoch budget)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.savefig(out_dir / "per_level_vs_naive.png", bbox_inches="tight")
    plt.close(fig)

    # --- budget strip: Heun at the same NFE points as the naive branch -------
    budgets = [int(b) for b in section.budgets]
    n = int(section.n_samples)
    panels = []
    for b in budgets:
        n_steps = (b + 3) // 2                    # NFE = 2(n−1) − 1
        gen = torch.Generator(device=device).manual_seed(seed)
        x, nfe = edm_heun_sample(model, (n, 1, 28, 28),
                                 sigma_max=float(section.sigma_max),
                                 sigma_min=float(section.sigma_min),
                                 n_steps=n_steps, rho=float(section.rho),
                                 generator=gen, device=device)
        panels.append((nfe, x))
        print(f"  Heun nfe={nfe}: speckle={speckle(x):.5f}")

    # reference: the naive branch's 600-NFE optimum, same seed
    naive_ref = None
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        naive_ref = sample_ve(naive, ve_sched, (n, 1, 28, 28), generator=gen,
                              device=device, euler_sub=1, corrector_steps=2)

    n_panels = len(panels) + 1
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 3.0 * nrows))
    axes = list(axes.flat)
    for ax, (nfe, x) in zip(axes, panels):
        _show(ax, x, f"EDM Heun {nfe} NFE")
    _show(axes[len(panels)], naive_ref, "naive PC 600 NFE (ref)")
    for ax in axes[len(panels) + 1:]:
        ax.axis("off")
    fig.suptitle("EDM: budget sweep 40–1000 NFE + naive reference", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "budget_strip.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    with open(out_dir / "speckle.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "nfe", "speckle"])
        for nfe, x in panels:
            w.writerow(["edm_heun", nfe, round(speckle(x), 5)])
        w.writerow(["naive_pc", 600, round(speckle(naive_ref), 5)])

    print(f"artifacts → {out_dir}/")


if __name__ == "__main__":
    main()
