"""Generate images from a trained checkpoint — the entry point for sampling.

Usage::

    uv run python scripts/sample.py --checkpoint checkpoints/<run>-best.ckpt \
        --sampler ddim --num-steps 25 --num-samples 64 --seed 0
    uv run python scripts/sample.py --checkpoint ... --sampler ddpm --num-samples 16

Loads the checkpoint, rebuilds the model/core from the config saved alongside it
(or from ``configs/default.yaml`` if the saved config isn't found), runs the
chosen sampler, and writes a PNG grid to ``samples/`` (and logs to TensorBoard if
a log dir is given).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from ddpm_lab.callbacks import _denormalize_to_uint8
from ddpm_lab.config import DEFAULT_CONFIG_PATH
from ddpm_lab.core import DiffusionCore
from ddpm_lab.models import build_model
from ddpm_lab.samplers import build_sampler


def _find_config_next_to_checkpoint(ckpt_path: Path) -> Path | None:
    """Look for a ``*-config.yaml`` saved by train.py near the checkpoint dir."""
    for cand in ckpt_path.parent.glob("*-config.yaml"):
        return cand
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Generate images from a trained DDPM checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to a .ckpt file from train.py.")
    p.add_argument("--config", default=None, help="Config YAML (default: next to checkpoint or configs/default.yaml).")
    p.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--num-steps", type=int, default=25, help="DDIM steps (ignored for ddpm).")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0 = deterministic).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="samples")
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Resolve config: explicit arg > saved next to checkpoint > default.
    if args.config:
        cfg = OmegaConf.load(args.config)
    else:
        saved = _find_config_next_to_checkpoint(ckpt_path)
        cfg = OmegaConf.load(saved) if saved else OmegaConf.load(DEFAULT_CONFIG_PATH)
    # Apply any CLI-relevant overrides to be safe.
    cfg.callbacks.num_samples = args.num_samples
    cfg.callbacks.num_sample_steps = args.num_steps
    cfg.callbacks.eta = args.eta

    # Device.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Rebuild model + core and load weights.
    core = DiffusionCore(
        num_timesteps=int(cfg.diffusion.num_timesteps),
        beta_start=float(cfg.diffusion.beta_start),
        beta_end=float(cfg.diffusion.beta_end),
        schedule=str(cfg.diffusion.schedule),
    ).float().to(device)
    model = build_model(cfg).to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Lightning saves state_dict under "state_dict" with a "model." prefix.
    state = ckpt.get("state_dict", ckpt)
    # Strip the leading "model." added by the LightningModule.
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")} or state
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected}")
    model.eval()

    # Sample.
    sampler = build_sampler(args.sampler)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    print(f"Generating {args.num_samples} images with {args.sampler} "
          f"({args.num_steps} steps, eta={args.eta}) on {device}...")
    with torch.no_grad():
        samples = sampler(
            model, core,
            (args.num_samples, int(cfg.model.in_channels), 28, 28),
            num_steps=args.num_steps if args.sampler == "ddim" else None,
            eta=args.eta,
            generator=gen,
            device=device,
        ).float()

    # Save PNG grid (denormalized back to displayable range).
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = _denormalize_to_uint8(samples.cpu())
    nrow = int(args.num_samples ** 0.5) or 8
    out_path = out_dir / f"samples-{args.sampler}-seed{args.seed}.png"
    save_image(img, out_path, nrow=nrow)
    print(f"Saved {out_path} ({args.num_samples} images, {nrow} per row)")


if __name__ == "__main__":
    main()
