"""Solver benchmark: reverse SDE vs PF-ODE solvers on the trained VE/VP models.

Reuses the checkpoints written by ``compare_ve_vp.py`` (no retraining) and
runs the §15 matrix: {VE, VP} × {euler_maruyama, euler, heun, rk4} × the NFE
budgets from the config, all branches starting from the same noise. Saves:

* ``solvers/<branch>_<method>_nfe<N>.png`` — sample grids per run;
* ``solvers/solver_benchmark.csv``       — method, branch, steps, NFE, time, metrics;
* ``solvers/quality_vs_nfe.png``         — the headline plot (log-NFE vs quality).

Quality metrics (no external deps — see README for their limits):
* sharpness — mean |Laplacian| of the un-normalized samples;
* speckle   — fraction of isolated bright pixels (3×3 max-pool test);
* eps_mse   — per-level eps-MSE of the samples under their own model.

Usage:
    uv run python scripts/compare_solvers.py                 # full matrix
    uv run python scripts/compare_solvers.py solvers.nfe_budgets=[40,100]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from score_lab.config import load_config
from score_lab.mnist_unet import build_mnist_unet
from score_lab.mnist_ve_vp import (
    MNIST_MEAN,
    MNIST_STD,
    VESchedule,
    VPSchedule,
    save_sample_grid,
)
from score_lab.sde import ContinuousVE, ContinuousVP
from score_lab.solvers import euler_maruyama, nfe_to_grid, ode_solver

SDE_METHODS = ["euler_maruyama"]
ODE_METHODS = ["euler", "heun", "rk4"]


def _unnormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * MNIST_STD + MNIST_MEAN).clamp(0, 1)


def sharpness(x: torch.Tensor) -> float:
    img = _unnormalize(x)
    lap = img[:, 0, 2:, 1:-1] + img[:, 0, :-2, 1:-1] + img[:, 0, 1:-1, 2:] \
        + img[:, 0, 1:-1, :-2] - 4 * img[:, 0, 1:-1, 1:-1]
    return float(lap.abs().mean())


def speckle(x: torch.Tensor) -> float:
    """Fraction of isolated bright pixels — the VE background artifact."""
    img = _unnormalize(x)
    bright = (img > 0.6).float()
    neigh = F.max_pool2d(bright, 3, stride=1, padding=1) - bright
    isolated = (bright * (neigh == 0).float()).sum().item()
    return isolated / bright.numel()


def eps_mse(x: torch.Tensor, branch, model, device) -> float:
    """Denoising loss of the samples under their own model (manifold proxy).

    Treat each sample as x0, noise it at several levels t, and measure how
    well the model denoises it back (eps_hat vs the actual eps). Samples that
    sit on the model's learned manifold denoise as well as real data; samples
    that drifted off-manifold score higher. Compare solvers *on one branch*,
    not branches between each other (each model has its own loss floor).
    """
    total, n = 0.0, 0
    ts = torch.linspace(0.05, 0.95, 7, device=device)
    for t in ts.tolist():
        tt = torch.full((x.shape[0],), t, device=device)
        z = torch.randn_like(x)
        xt = branch.forward_noise(x, tt, z)
        with torch.no_grad():
            total += float(((branch.eps(model, xt, tt) - z) ** 2).mean())
        n += 1
    return total / n


def main() -> None:
    cfg = load_config()
    section = cfg.solvers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ve_vp_dir = Path(cfg.paths.run_dir).parent / "ve_vs_vp"
    out_dir = ve_vp_dir / "solvers"
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [p.name for p in (ve_vp_dir / "model_ve.pt", ve_vp_dir / "model_vp.pt")
               if not p.exists()]
    if missing:
        sys.exit(f"missing checkpoints {missing} — run `uv run python scripts/compare_ve_vp.py` first")

    ve_sched = VESchedule(**dict(cfg.mnist_ve_vp.ve))
    vp_sched = VPSchedule(**dict(cfg.mnist_ve_vp.vp))
    branches = {
        "ve": ContinuousVE(ve_sched),
        "vp": ContinuousVP(vp_sched),
    }
    models = {}
    for name in branches:
        m = build_mnist_unet(cfg).to(device).eval()
        m.load_state_dict(torch.load(ve_vp_dir / f"model_{name}.pt",
                                     map_location=device, weights_only=True))
        models[name] = m

    n = int(section.n_samples)
    seed = int(section.seed)
    budgets = [int(b) for b in section.nfe_budgets]
    methods = SDE_METHODS + ODE_METHODS if bool(section.include_rk4) else SDE_METHODS + ["euler", "heun"]

    # warm up CUDA kernels so the first timed run is not an outlier
    dummy = branches["ve"].prior_sample((2, 1, 28, 28), device=device)
    with torch.no_grad():
        branches["ve"].eps(models["ve"], dummy, torch.full((2,), 1.0, device=device))
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    rows = []
    # Baseline: real MNIST under each model — the quality floor to compare
    # solver runs against (a perfect sampler should land near its branch's floor).
    from score_lab.mnist_ve_vp import mnist_loaders
    loader = mnist_loaders(str(Path(__file__).resolve().parents[1] / "data"), n)
    real = next(iter(loader))[0].to(device)
    for branch_name, branch in branches.items():
        base = round(eps_mse(real, branch, models[branch_name], device), 5)
        rows.append({"method": "real_data", "branch": branch_name, "steps": 0,
                     "nfe": 0, "seconds": 0.0, "sharpness": round(sharpness(real), 5),
                     "speckle": round(speckle(real), 5), "eps_mse": base,
                     "diverged": 0})
        print(f"  baseline {branch_name}: real-data eps_mse={base}")

    for branch_name, branch in branches.items():
        model = models[branch_name]
        # same starting cloud for every method of this branch
        gen = torch.Generator(device=device).manual_seed(seed)
        x_start = branch.prior_sample((n, 1, 28, 28), generator=gen, device=device)
        for method in methods:
            for budget in budgets:
                grid = nfe_to_grid(budget, "euler" if method == "euler_maruyama" else method,
                                   torch.device(device))
                gen_run = torch.Generator(device=device).manual_seed(seed + 1)
                with torch.no_grad():
                    if method == "euler_maruyama":
                        drift = lambda x, t, b=branch, m=model: b.reverse_sde_drift(m, x, t)
                        res = euler_maruyama(drift, branch.diffusion, x_start.clone(),
                                             grid, generator=gen_run)
                    else:
                        drift = lambda x, t, b=branch, m=model: b.pf_ode_drift(m, x, t)
                        res = ode_solver(drift, x_start.clone(), grid, method=method)
                diverged = bool(torch.isnan(res.x).any() or torch.isinf(res.x).any())
                if diverged:
                    res.x = torch.nan_to_num(res.x, nan=0.0, posinf=1.0, neginf=-1.0)
                row = {
                    "method": method, "branch": branch_name, "steps": len(grid) - 1,
                    "nfe": res.nfe, "seconds": round(res.seconds, 4),
                    "sharpness": round(sharpness(res.x), 5),
                    "speckle": round(speckle(res.x), 5),
                    "eps_mse": round(eps_mse(res.x, branch, model, device), 5),
                    "diverged": int(diverged),
                }
                rows.append(row)
                tag = f"{branch_name}_{method}_nfe{budget}"
                save_sample_grid(res.x, out_dir / f"{tag}.png",
                                 f"{tag}  ({res.nfe} NFE, {res.seconds:.2f}s)")
                print(f"  {tag}: nfe={res.nfe} t={res.seconds:.2f}s "
                      f"sharp={row['sharpness']} eps_mse={row['eps_mse']} div={row['diverged']}")

    with open(out_dir / "solver_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- quality vs NFE -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = {"ve": "#d62728", "vp": "#1f77b4"}
    markers = {"euler_maruyama": "o", "euler": "s", "heun": "^", "rk4": "D"}
    for row in rows:
        if row["diverged"] or row["method"] == "real_data":
            continue
        ax.scatter(row["nfe"], row["eps_mse"],
                   color=colors[row["branch"]], marker=markers[row["method"]], s=45)
    # real-data floor per branch as a dashed reference line
    for row in rows:
        if row["method"] == "real_data":
            ax.axhline(row["eps_mse"], color=colors[row["branch"]],
                       ls=":", lw=1, alpha=0.6)
    for branch_name, color in colors.items():
        for method, mk in markers.items():
            pts = [r for r in rows if r["branch"] == branch_name
                   and r["method"] == method and not r["diverged"]]
            if pts:
                ax.plot([r["nfe"] for r in pts], [r["eps_mse"] for r in pts],
                        color=color, marker=mk, ms=4, lw=1, alpha=0.5,
                        label=f"{branch_name.upper()} {method}")
    ax.set_xscale("log")
    ax.set_xlabel("NFE (model evaluations, log scale)")
    ax.set_ylabel("eps-MSE of samples (lower = closer to the model's manifold)")
    ax.set_title("Solver quality vs compute (same trained models)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(out_dir / "quality_vs_nfe.png", bbox_inches="tight")
    plt.close(fig)

    print(f"artifacts → {out_dir}/")


if __name__ == "__main__":
    main()
