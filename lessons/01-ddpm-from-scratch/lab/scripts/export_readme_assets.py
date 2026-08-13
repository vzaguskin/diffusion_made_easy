"""Export README-ready images from a TensorBoard run.

Pulls the assets the lesson README shows straight out of the event files:

1. **Sample grids at different epochs** (tag ``samples/ddim``) — the same fixed
   seed denoised by progressively-trained checkpoints; you watch noise turn
   into digits as training goes.
2. **The final DDPM-vs-DDIM comparison** (tags ``compare/ddim`` /
   ``compare/ddpm``) — same starting noise through both samplers (theory.md
   §17 vs §18).
3. **Loss curves** (``train/loss_epoch``, ``val/loss``) plus the final
   per-timestep-bucket loss bar chart (``val/loss_bucket_*``) — the visual
   counterpart of theory.md §15's "weight dropping".

Usage::

    # latest run, sensible defaults (untrained / early / mid / late / final)
    uv run python scripts/export_readme_assets.py

    # pick a specific run dir and which epochs to show (0-based epochs;
    # index -1 = the pre-training "sanity" frame)
    uv run python scripts/export_readme_assets.py --run runs/ddpm-... --epochs -1,1,5,11

Images are written to ``../images/`` (the lesson's image dir, which the
repo's .gitignore explicitly tracks — see the negation rule there).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorboard.backend.event_processing import event_accumulator

LAB_DIR = Path(__file__).resolve().parents[1]
LESSON_DIR = LAB_DIR.parent
DEFAULT_OUT_DIR = LESSON_DIR / "images"


def find_latest_run(runs_dir: Path) -> Path:
    """Return the newest event file under ``runs_dir``."""
    events = sorted(runs_dir.glob("**/events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"No TensorBoard event files under {runs_dir}")
    return events[-1]


def load_event_file(event_path: Path) -> event_accumulator.EventAccumulator:
    ea = event_accumulator.EventAccumulator(
        str(event_path.parent),
        size_guidance={"scalars": 0, "images": 0, "histograms": 0},
    )
    ea.Reload()
    return ea


def export_epoch_grids(
    ea: event_accumulator.EventAccumulator,
    epochs: list[int],
    tag: str,
    out_dir: Path,
    prefix: str = "samples-epoch",
) -> list[Path]:
    """Save one sample grid per requested epoch index.

    Frame indices in the tag: 0 = the pre-training sanity check (untrained
    model), frame ``i`` = after epoch ``i-1``. Passing ``-1`` selects the
    untrained frame.
    """
    frames = ea.Images(tag)
    written: list[Path] = []
    for e in epochs:
        frame_idx = e + 1  # shift: epoch -1 (untrained) -> frame 0
        if frame_idx < 0 or frame_idx >= len(frames):
            print(f"  [skip] epoch {e}: no frame (have {len(frames)} frames)")
            continue
        im = frames[frame_idx]
        label = "untrained" if e < 0 else f"epoch{e:02d}"
        out = out_dir / f"{prefix}-{label}.png"
        out.write_bytes(im.encoded_image_string)
        written.append(out)
        print(f"  wrote {out.relative_to(out_dir.parents[1])}  ({im.width}x{im.height})")
    return written


def export_comparison(ea: event_accumulator.EventAccumulator, out_dir: Path) -> list[Path]:
    """Save the final DDPM and DDIM comparison grids (same seed)."""
    written = []
    for tag, name in [("compare/ddim", "final-comparison-ddim"), ("compare/ddpm", "final-comparison-ddpm")]:
        if tag not in ea.Tags()["images"]:
            print(f"  [skip] tag {tag} not in this run")
            continue
        frames = ea.Images(tag)
        im = frames[-1]  # the most trained checkpoint
        out = out_dir / f"{name}.png"
        out.write_bytes(im.encoded_image_string)
        written.append(out)
        print(f"  wrote {out.relative_to(out_dir.parents[1])}")
    return written


def export_loss_curves(ea: event_accumulator.EventAccumulator, out_dir: Path) -> Path:
    """Plot train/val loss curves + final per-bucket loss with matplotlib."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    scalars = ea.Tags()["scalars"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Panel 1: loss curves ------------------------------------------------
    ax = axes[0]
    if "train/loss_epoch" in scalars:
        ev = ea.Scalars("train/loss_epoch")
        ax.plot([e.step for e in ev], [e.value for e in ev], label="train", color="#1f77b4")
    if "val/loss" in scalars:
        ev = ea.Scalars("val/loss")
        ax.plot([e.step for e in ev], [e.value for e in ev], label="val", color="#d62728", marker="o", ms=3)
    ax.set_xlabel("global step")
    ax.set_ylabel("MSE loss  ‖ε − ε_θ(x_t, t)‖²")
    ax.set_title("Noise-prediction loss (theory.md §15)")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Panel 2: loss by t-bucket at the last logged epoch -------------------
    ax = axes[1]
    bucket_tags = sorted(
        [t for t in scalars if t.startswith("val/loss_bucket_")],
        key=lambda t: int(t.rsplit("_", 1)[1]),
    )
    if bucket_tags:
        vals = [ea.Scalars(t)[-1].value for t in bucket_tags]
        labels = [str(int(t.rsplit("_", 1)[1])) for t in bucket_tags]
        ax.bar(labels, vals, color="#2ca02c", alpha=0.8)
        ax.set_xlabel("timestep bucket (0 = almost clean … 9 = almost pure noise)")
        ax.set_ylabel("val MSE in bucket")
        ax.set_title("Where the model struggles across noise levels")
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "no bucketed losses logged", ha="center", va="center")

    fig.tight_layout()
    out = out_dir / "loss-curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.relative_to(out_dir.parents[1])}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export README images from a TB run.")
    p.add_argument("--run", default=None, help="Run dir under runs/ (default: latest event file).")
    p.add_argument("--epochs", default="-1,1,4,8,11",
                   help="Comma-separated 0-based epochs for the progression grids; -1 = untrained frame.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Where to write PNGs.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run:
        event_path = find_latest_run(Path(args.run))
    else:
        event_path = find_latest_run(LAB_DIR / "runs")
    print(f"Reading: {event_path}")

    ea = load_event_file(event_path)
    epochs = [int(e) for e in args.epochs.split(",")]

    print("\n[1/3] epoch progression grids (fixed seed, tag 'samples/ddim'):")
    if "samples/ddim" in ea.Tags()["images"]:
        export_epoch_grids(ea, epochs, "samples/ddim", out_dir)
    else:
        print("  [skip] tag 'samples/ddim' not found")

    print("\n[2/3] DDPM vs DDIM comparison (final epoch):")
    export_comparison(ea, out_dir)

    print("\n[3/3] loss curves:")
    export_loss_curves(ea, out_dir)

    print(f"\nDone. Images in {out_dir}")


if __name__ == "__main__":
    main()
