"""Visualization — a first-class citizen of this lab (see proposal).

The point of the 2D lab is *seeing* the score. Every plot follows one style:

* data samples — blue dots,
* true (analytic) score — grey arrows,
* learned score ``s_θ(x, σ)`` — red arrows.

Every function takes ``out_path`` and writes a PNG there (the caller points it
into ``runs/<exp>/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless-friendly: never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import torch

# The unified palette (spec: score-visualization).
C_DATA = "#1f77b4"      # blue — data samples
C_TRUE = "#7f7f7f"      # grey — analytic score
C_LEARNED = "#d62728"   # red  — learned score

# Global figure defaults.
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 9,
})


def _grid(lim: float, n: int = 17) -> torch.Tensor:
    """A square grid of ``n×n`` points in ``[-lim, lim]²`` — quiver arrow anchors."""
    xs = torch.linspace(-lim, lim, n)
    ys = torch.linspace(-lim, lim, n)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def _quiver(ax, xy: np.ndarray, vec: np.ndarray, color: str, label: str, scale_key: float):
    """One quiver layer with unit-length arrows (magnitude → the colorbar)."""
    norm = np.linalg.norm(vec, axis=1, keepdims=True).clip(1e-12)
    unit = vec / norm
    q = ax.quiver(xy[:, 0], xy[:, 1], unit[:, 0], unit[:, 1],
                  color=color, scale=scale_key, scale_units="xy",
                  width=0.004, alpha=0.9, angles="xy")
    ax.quiverkey(q, 0.85, 1.04, 1, label, color=color, labelpos="E",
                 coordinates="axes", fontproperties={"size": 8})
    return q


def plot_score_field(
    true_fn: Callable[[torch.Tensor], torch.Tensor],
    learned_fn: Callable[[torch.Tensor], torch.Tensor],
    sigma: float,
    out_path: Path,
    lim: float = 3.0,
    n_grid: int = 17,
    data: torch.Tensor | None = None,
    title: str | None = None,
) -> None:
    """True (grey) vs learned (red) score arrows on a common grid.

    Arrows are *unit length* — the direction is what matters visually; magnitude
    is encoded in the colorbar over ``log₁₀‖s‖`` of the learned field (spec:
    score-visualization, "True vs learned score quiver plot").
    """
    xy = _grid(lim, n_grid)
    s_true = true_fn(xy).detach().cpu().numpy()
    s_learned = learned_fn(xy).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    if data is not None:
        d = data.cpu().numpy()
        ax.scatter(d[:, 0], d[:, 1], s=2, c=C_DATA, alpha=0.25, label="data")

    scale_key = 25.0  # arrows per axis unit — keeps unit arrows readable
    _quiver(ax, xy.numpy(), s_true, C_TRUE, f"true score", scale_key)
    q = _quiver(ax, xy.numpy(), s_learned, C_LEARNED, "learned $s_θ$", scale_key)

    # Colorbar on the learned field's log-magnitude.
    norm = np.linalg.norm(s_learned, axis=1)
    sm = plt.cm.ScalarMappable(cmap="Reds",
                               norm=plt.Normalize(vmin=np.log10(norm.min()),
                                                  vmax=np.log10(norm.max())))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\log_{10}\|s_θ\|$")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title or f"Score field at σ = {sigma:g} (grey = truth, red = learned)")
    ax.legend(loc="upper left", fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_density(
    dist,
    out_path: Path,
    data: torch.Tensor | None = None,
    lim: float = 3.0,
    n: int = 200,
    title: str | None = None,
) -> None:
    """Heatmap of ``log p(x)`` with data samples on top (score-visualization spec)."""
    xs = torch.linspace(-lim, lim, n)
    ys = torch.linspace(-lim, lim, n)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    logp = dist.log_p(xy).reshape(n, n).numpy()

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(logp, origin="lower", extent=[-lim, lim, -lim, lim],
                   cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log p(x)$")
    if data is not None:
        d = data.cpu().numpy()
        ax.scatter(d[:, 0], d[:, 1], s=1.5, c=C_DATA, alpha=0.3)
    ax.set_title(title or "Data density (log p)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_trajectories(
    trajs: torch.Tensor,
    field_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    out_path: Path,
    lim: float = 3.0,
    sigma_for_field: float | None = None,
    n_grid: int = 15,
    title: str | None = None,
) -> None:
    """Langevin paths (grey→blue lines) over a quiver/heatmap background.

    ``trajs`` : [n_steps+1, N, 2] — from ``langevin_sample(..., return_trajectory=True)``.
    Final positions get a marker so the eye lands on the modes.
    """
    trajs_np = trajs.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.0, 5.6))

    if field_fn is not None:
        xy = _grid(lim, n_grid)
        s = field_fn(xy).detach().cpu().numpy()
        norm = np.linalg.norm(s, axis=1, keepdims=True).clip(1e-12)
        ax.quiver(xy[:, 0], xy[:, 1], (s / norm)[:, 0], (s / norm)[:, 1],
                  color=C_TRUE, scale=30, scale_units="xy", width=0.003,
                  alpha=0.55, angles="xy")

    n_traj = trajs_np.shape[1]
    cmap = plt.get_cmap("Blues")
    for i in range(n_traj):
        path = trajs_np[:, i, :]
        ax.plot(path[:, 0], path[:, 1], color=cmap(0.55), lw=0.8, alpha=0.8,
                zorder=2 if i else 3)
    finals = trajs_np[-1]
    ax.scatter(finals[:, 0], finals[:, 1], s=18, c=C_LEARNED, zorder=4,
               edgecolors="white", linewidths=0.4, label="final points")
    starts = trajs_np[0]
    ax.scatter(starts[:, 0], starts[:, 1], s=10, c="k", marker="x", zorder=3,
               label="start")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ttl = title or "Langevin trajectories over the score field"
    if sigma_for_field is not None:
        ttl += f" (field at σ = {sigma_for_field:g})"
    ax.set_title(ttl)
    ax.legend(loc="upper left", fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_annealing_grid(
    snapshots: dict[int, list[torch.Tensor]],
    sigmas: torch.Tensor,
    out_path: Path,
    lim: float = 3.5,
    n_cols: int = 4,
    stride: int | None = None,
) -> None:
    """The lab's "video frame": rows = σ levels, cols = steps within a level.

    Each cell is a scatter of the *same* sample cloud as annealing shrinks it:
    wide blur at σ_max (top) → tight modes at σ_min (bottom). Spec:
    score-visualization, "Annealing progress grid".
    """
    n_levels = len(sigmas)
    rows = []
    for lvl in range(n_levels):
        snaps = snapshots[lvl]
        st = stride or max(1, (len(snaps) - 1) // (n_cols - 1))
        idxs = [0] + [i for i in range(st, len(snaps), st)][: n_cols - 1]
        rows.append([snaps[i] for i in idxs])
    n_cols_actual = max(len(r) for r in rows)

    fig, axes = plt.subplots(n_levels, n_cols_actual,
                             figsize=(2.0 * n_cols_actual, 1.9 * n_levels),
                             squeeze=False)
    for lvl, row in enumerate(rows):
        for col in range(n_cols_actual):
            ax = axes[lvl][col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            if col < len(row):
                pts = row[col].detach().cpu().numpy()
                ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c=C_DATA, alpha=0.5)
            if col == 0:
                ax.set_ylabel(f"σ={float(sigmas[lvl]):.2f}", fontsize=8)
    fig.suptitle("Annealed Langevin: the cloud condenses as σ decreases", y=1.005)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_loss_curve(
    losses: list[float],
    out_path: Path,
    title: str = "DSM training loss",
) -> None:
    """Per-epoch training loss — the honest "did it learn?" sanity check."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(range(1, len(losses) + 1), losses, color=C_LEARNED, lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("DSM loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
