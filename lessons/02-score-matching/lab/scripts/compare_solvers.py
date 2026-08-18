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
import time as _time
from pathlib import Path
from types import SimpleNamespace

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
    sample_ddim,
    sample_ve,
    sample_vp,
    save_sample_grid,
)
from score_lab.sde import ContinuousVE, ContinuousVP
from score_lab.solvers import euler_maruyama, nfe_to_grid, ode_solver

SDE_METHODS = ["euler_maruyama"]
ODE_METHODS = ["euler", "heun", "rk4"]
# Discrete lesson-01 samplers, for the wall-clock comparison with the
# continuous solvers. Both only make sense on the VP schedule.
DISCRETE_VP_METHODS = ["ddim", "ddpm_ancestral"]
# NCSN predictor-corrector (Euler + Langevin corrector) — the only scheme
# that produces digits with this lab's rough VE model (see README). The three
# presets sweep the NFE budget to show digits *emerging* (600 → hints,
# 1400 → half legible, 3000 → most cells digits).
VE_PC_PRESETS = [(1, 2), (2, 5), (5, 10)]   # (euler_sub, corrector_steps) per level


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
        # prefer the long-trained VE checkpoint if the ve-vp run produced one
        ckpt = f"model_{name}_long.pt" if name == "ve" and \
            (ve_vp_dir / "model_ve_long.pt").exists() else f"model_{name}.pt"
        m = build_mnist_unet(cfg).to(device).eval()
        m.load_state_dict(torch.load(ve_vp_dir / ckpt, map_location=device,
                                     weights_only=True))
        print(f"  {name}: checkpoint {ckpt}")
        models[name] = m

    n = int(section.n_samples)
    seed = int(section.seed)
    budgets = [int(b) for b in section.nfe_budgets]
    solver_methods = ODE_METHODS if bool(section.include_rk4) else ["euler", "heun"]
    methods = {"ve": SDE_METHODS + solver_methods + ["pc_langevin"],
               "vp": SDE_METHODS + solver_methods + DISCRETE_VP_METHODS}

    # warm up CUDA kernels so the first timed run is not an outlier
    dummy = branches["ve"].prior_sample((2, 1, 28, 28), device=device)
    with torch.no_grad():
        branches["ve"].eps(models["ve"], dummy, torch.full((2,), 1.0, device=device))
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    rows = []
    samples: dict[tuple, torch.Tensor] = {}
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
        for method in methods[branch_name]:
            # ddpm_ancestral is a single full-T run; pc_langevin runs its own
            # (sub, corrector) presets — neither follows the NFE budgets
            if method == "ddpm_ancestral":
                method_budgets = [None]
            elif method == "pc_langevin":
                method_budgets = list(VE_PC_PRESETS)
            else:
                method_budgets = budgets
            for budget in method_budgets:
                if method in DISCRETE_VP_METHODS or method == "pc_langevin":
                    grid = None
                else:
                    grid = nfe_to_grid(budget, "euler" if method == "euler_maruyama" else method,
                                       torch.device(device))
                gen_run = torch.Generator(device=device).manual_seed(seed + 1)
                with torch.no_grad():
                    if method == "pc_langevin":
                        sub, corr = budget
                        t0 = _time.perf_counter()
                        x = sample_ve(model, ve_sched, (n, 1, 28, 28),
                                      generator=gen_run, device=device,
                                      euler_sub=sub, corrector_steps=corr)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        res = SimpleNamespace(x=x, nfe=ve_sched.n_levels * (sub + corr),
                                              seconds=_time.perf_counter() - t0)
                        budget = res.nfe  # tag with the real NFE
                    elif method == "ddim":
                        t0 = _time.perf_counter()
                        x = sample_ddim(model, vp_sched, (n, 1, 28, 28), n_steps=budget,
                                        generator=gen_run, device=device)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        res = SimpleNamespace(x=x, nfe=budget,
                                              seconds=_time.perf_counter() - t0)
                    elif method == "ddpm_ancestral":
                        t0 = _time.perf_counter()
                        x = sample_vp(model, vp_sched, (n, 1, 28, 28),
                                      generator=gen_run, device=device)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        res = SimpleNamespace(x=x, nfe=vp_sched.num_timesteps,
                                              seconds=_time.perf_counter() - t0)
                    elif method == "euler_maruyama":
                        drift = lambda x, t, b=branch, m=model: b.reverse_sde_drift(m, x, t)
                        res = euler_maruyama(drift, branch.diffusion, x_start.clone(),
                                             grid, generator=gen_run)
                    else:
                        drift = lambda x, t, b=branch, m=model: b.pf_ode_drift(m, x, t)
                        res = ode_solver(drift, x_start.clone(), grid, method=method)
                diverged = bool(torch.isnan(res.x).any() or torch.isinf(res.x).any())
                if diverged:
                    res.x = torch.nan_to_num(res.x, nan=0.0, posinf=1.0, neginf=-1.0)
                steps = 0 if method in ("ddim", "ddpm_ancestral", "pc_langevin") else len(grid) - 1
                row = {
                    "method": method, "branch": branch_name, "steps": steps,
                    "nfe": res.nfe, "seconds": round(res.seconds, 4),
                    "sharpness": round(sharpness(res.x), 5),
                    "speckle": round(speckle(res.x), 5),
                    "eps_mse": round(eps_mse(res.x, branch, model, device), 5),
                    "diverged": int(diverged),
                }
                rows.append(row)
                samples[(branch_name, method, res.nfe)] = res.x
                tag = f"{branch_name}_{method}_nfe{res.nfe}"
                save_sample_grid(res.x, out_dir / f"{tag}.png",
                                 f"{tag}  ({res.nfe} NFE, {res.seconds:.2f}s)")
                print(f"  {tag}: nfe={res.nfe} t={res.seconds:.2f}s "
                      f"sharp={row['sharpness']} eps_mse={row['eps_mse']} div={row['diverged']}")

        # --- Montage: rows = methods, cols = NFE budgets (same start noise) ---
        montage_methods = [m for m in methods[branch_name]
                           if m not in ("ddpm_ancestral", "pc_langevin")]
        fig, axes = plt.subplots(len(montage_methods), len(budgets),
                                 figsize=(2.6 * len(budgets), 2.5 * len(montage_methods)))
        for r, method in enumerate(montage_methods):
            for c, budget in enumerate(budgets):
                ax = axes[r, c] if len(methods) > 1 else axes[c]
                x = samples[(branch_name, method, budget)][:16].detach().cpu()
                img = (_unnormalize(x)
                       .reshape(2, 8, 28, 28).permute(0, 2, 1, 3).reshape(56, 224))
                ax.imshow(img, cmap="gray_r")
                ax.set_title(f"{budget} NFE", fontsize=9 if c else 10)
                if c == 0:
                    ax.set_ylabel(method, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{branch_name.upper()}: solvers × NFE budgets (same starting noise)",
                     y=0.995)
        fig.tight_layout()
        fig.savefig(out_dir / f"{branch_name}_solver_montage.png", bbox_inches="tight",
                    dpi=120)
        plt.close(fig)

        # --- VE-only: the predictor-corrector NFE emergence strip ------------
        if branch_name == "ve":
            pc_runs = sorted((nfe, x) for (b, meth, nfe), x in samples.items()
                             if b == "ve" and meth == "pc_langevin")
            fig, axes = plt.subplots(1, len(pc_runs), figsize=(2.8 * len(pc_runs), 3.0))
            if len(pc_runs) == 1:
                axes = [axes]
            for ax, (nfe, x) in zip(axes, pc_runs):
                img = (_unnormalize(x[:16].detach().cpu())
                       .reshape(2, 8, 28, 28).permute(0, 2, 1, 3).reshape(56, 224))
                ax.imshow(img, cmap="gray_r")
                ax.set_title(f"{nfe} NFE", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle("VE predictor-corrector: digits emerge with the budget "
                         "(same model, same start)", y=1.02)
            fig.tight_layout()
            fig.savefig(out_dir / "ve_pc_emergence.png", bbox_inches="tight", dpi=120)
            plt.close(fig)

    with open(out_dir / "solver_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- quality vs NFE -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = {"ve": "#d62728", "vp": "#1f77b4"}
    markers = {"euler_maruyama": "o", "euler": "s", "heun": "^", "rk4": "D",
               "ddim": "v", "ddpm_ancestral": "*", "pc_langevin": "P"}
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
