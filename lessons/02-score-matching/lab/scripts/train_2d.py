"""End-to-end 2D pipeline: train a score network, then *see* it (§2–§8).

One command produces, in ``runs/<exp>/``:
* ``loss_curve.png``        — DSM training loss;
* ``density.png``           — heatmap of log p + data samples;
* ``score_field_*.png``     — true (grey) vs learned (red) score quivers at a
                              large / mid / small σ of the ladder;
* ``trajectories.png``      — annealed-Langevin paths over the score field;
* ``annealing_grid.png``    — the "video frame": cloud condensing per σ level;
* ``single_sigma_collapse.png`` — the §8 failure demo: single small σ, chains
  stuck near the low-density start (with coverage numbers in the title);
* ``loss.csv``              — per-step losses incl. per-σ-level breakdown.

Usage:
    uv run python scripts/train_2d.py [key=value ...]
    uv run python scripts/train_2d.py data.dataset=moons paths.run_dir=runs/moons
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from score_lab.config import geometric_sigma_ladder, load_config
from score_lab.langevin import annealed_langevin, langevin_sample, mode_coverage
from score_lab.models import build_score_model
from score_lab.toy_data import DATASETS, GaussianMixture2D
from score_lab.train_loop import train_score_model
from score_lab.viz import (
    plot_annealing_grid,
    plot_density,
    plot_loss_curve,
    plot_score_field,
    plot_trajectories,
)


def smoothed_score_fn(dist: GaussianMixture2D, model, learned: bool):
    """Score of the σ-smoothed density: std_i → √(σ_i² + σ²) (theory.md §7–8).

    For the learned net we simply call it at σ; for the analytic truth we build
    the smoothed mixture — that is the *correct* target at noise level σ.
    """
    if learned:
        def fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            return model(x, sigma.reshape(-1)[0].expand(x.shape[0]))
        return fn

    cache: dict[float, GaussianMixture2D] = {}

    def fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        s = float(sigma.reshape(-1)[0])
        if s not in cache:
            cache[s] = GaussianMixture2D(
                means=dist.means,
                stds=(dist.stds ** 2 + s ** 2).sqrt(),
                weights=dist.weights,
            )
        return cache[s].score(x).float()
    return fn


def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg.train.seed)

    run_dir = Path(cfg.paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Data + σ ladder ----------------------------------------------------
    dist = DATASETS[cfg.data.dataset](**dict(cfg.data[cfg.data.dataset]))
    data = dist.sample(cfg.data.n_samples)
    sigmas = geometric_sigma_ladder(cfg.sigmas.sigma_max, cfg.sigmas.sigma_min,
                                    cfg.sigmas.n_levels)
    print(f"dataset={cfg.data.dataset}  ladder: {float(sigmas[0]):.3f} → "
          f"{float(sigmas[-1]):.3f} in {len(sigmas)} levels")

    # --- Train (§7–§8) ------------------------------------------------------
    model = build_score_model(cfg)
    losses = train_score_model(
        model, dist.sample, sigmas,
        epochs=cfg.train.epochs,
        steps_per_epoch=cfg.train.steps_per_epoch,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        seed=cfg.train.seed,
        log_csv=run_dir / "loss.csv",
    )
    model.eval()
    plot_loss_curve(losses, run_dir / "loss_curve.png")

    # --- Density (§6) --------------------------------------------------------
    plot_density(dist, run_dir / "density.png", data=data)

    # --- True vs learned score fields (§2, §7) -------------------------------
    true_fn = smoothed_score_fn(dist, None, learned=False)
    learned_fn = smoothed_score_fn(dist, model, learned=True)
    for tag, idx in (("large", 0), ("mid", len(sigmas) // 2), ("small", -1)):
        s = float(sigmas[idx])
        plot_score_field(
            lambda xy, _s=s: true_fn(xy, torch.tensor([_s])),
            lambda xy, _s=s: learned_fn(xy, torch.tensor([_s])),
            sigma=s,
            out_path=run_dir / f"score_field_{tag}_sigma{idx}.png",
            data=data,
        )

    # --- Annealed Langevin: trajectories + the annealing grid (§4, §8) -------
    # All sampling demos below use the *learned* score — that is the point.
    gen = torch.Generator().manual_seed(cfg.train.seed + 1)
    x0_cloud = torch.randn(cfg.langevin.n_chains, 2, generator=gen) * 0.2
    x_final, snaps = annealed_langevin(
        learned_fn, x0_cloud, sigmas,
        steps_per_level=cfg.langevin.steps_per_level,
        step_scale=cfg.langevin.step_scale,
        generator=gen, return_snapshots=True,
    )
    plot_annealing_grid(snaps, sigmas, run_dir / "annealing_grid.png")

    x0_traj = torch.randn(cfg.langevin.n_trajectories, 2, generator=gen) * 0.2
    _, traj_snaps = annealed_langevin(
        learned_fn, x0_traj, sigmas,
        steps_per_level=cfg.langevin.steps_per_level,
        step_scale=cfg.langevin.step_scale,
        generator=gen, return_snapshots=True,
    )
    # Full path = level 0 start + every level's states (minus duplicate starts).
    trajs = torch.stack([traj_snaps[0][0]]
                        + [s for lvl in traj_snaps.values() for s in lvl[1:]], dim=0)
    mid = float(sigmas[len(sigmas) // 2])
    plot_trajectories(
        trajs,
        field_fn=lambda xy: true_fn(xy, torch.tensor([mid])),
        out_path=run_dir / "trajectories.png",
        sigma_for_field=mid,
    )

    # --- Single-σ collapse demo (§8) -----------------------------------------
    # Same start, same *final* step size as the annealed run — the only
    # difference is the missing ladder. With the learned score too.
    eps_single = cfg.langevin.step_scale * float(sigmas[-1]) ** 2
    x_single = langevin_sample(learned_fn, x0_cloud,
                               n_steps=cfg.langevin.steps_per_level * len(sigmas),
                               eps=eps_single, sigma=float(sigmas[-1]),
                               generator=gen)
    cov_ann = mode_coverage(x_final, dist.means, radius=0.5)
    cov_single = mode_coverage(x_single, dist.means, radius=0.5)
    print(f"mode coverage: annealed {cov_ann:.2f} vs single-σ {cov_single:.2f}")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, pts, ttl in (
        (axes[0], x_final, f"annealed (coverage {cov_ann:.2f})"),
        (axes[1], x_single, f"single σ={float(sigmas[-1]):.2f} (coverage {cov_single:.2f})"),
    ):
        p = pts.detach().cpu().numpy()
        ax.scatter(p[:, 0], p[:, 1], s=3, c="#1f77b4", alpha=0.5)
        m = dist.means.numpy()
        ax.scatter(m[:, 0], m[:, 1], s=60, c="k", marker="x", label="modes")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=9); ax.legend(fontsize=8)
    fig.suptitle("§8: one small σ is not enough — the ladder is", y=1.02)
    fig.savefig(run_dir / "single_sigma_collapse.png", bbox_inches="tight")
    plt.close(fig)

    print(f"artifacts → {run_dir}/")


if __name__ == "__main__":
    main()
