"""Config loading (OmegaConf) + the geometric sigma ladder.

Same pattern as lesson 01: defaults from ``configs/default.yaml``, CLI overrides
as ``key=value``, whole-file override via ``--config PATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(argv: list[str] | None = None) -> DictConfig:
    """Load default.yaml (+ ``--config PATH`` base) and apply ``key=value`` overrides."""
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = DEFAULT_CONFIG_PATH
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            config_path = Path(argv[i + 1])
            i += 2
        elif arg.startswith("--config="):
            config_path = Path(arg.split("=", 1)[1])
            i += 1
        else:
            overrides.append(arg)
            i += 1
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    base = OmegaConf.load(config_path)
    return OmegaConf.merge(base, OmegaConf.from_cli(overrides))


def geometric_sigma_ladder(sigma_max: float, sigma_min: float, n_levels: int) -> torch.Tensor:
    """The noise ladder of theory.md §8 (multi-level noise, NCSN-style).

    Geometric progression from ``sigma_max`` (≈ data diameter — noise covers
    everything) down to ``sigma_min`` (≈ narrowest mode width). A geometric
    ladder is the natural choice because σ spans orders of magnitude and the
    network is conditioned on log σ.
    """
    if not (0 < sigma_min < sigma_max):
        raise ValueError(f"need 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}")
    # torch has no geomspace: exp(linspace(log σ_max, log σ_min, L)) — same thing.
    logs = torch.linspace(torch.log(torch.tensor(sigma_max)),
                          torch.log(torch.tensor(sigma_min)), n_levels)
    return torch.exp(logs)


__all__ = ["load_config", "geometric_sigma_ladder", "DEFAULT_CONFIG_PATH"]
