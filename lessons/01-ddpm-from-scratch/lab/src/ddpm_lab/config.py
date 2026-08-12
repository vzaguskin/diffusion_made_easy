"""Config loading and CLI overrides via OmegaConf.

Defaults come from ``configs/default.yaml``. Overrides are passed on the CLI as
``key=value`` pairs (e.g. ``optim.lr=5e-4``) or a whole file via ``--config PATH``.

Usage in a script::

    cfg = load_config()           # parses sys.argv for --config and key=value
    print(cfg.optim.lr)           # structured access

The config is structurally validated against :data:`DEFAULTS` so missing keys are
filled in and typos in the *default* file surface early.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# The default config file, relative to the lab root.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(argv: list[str] | None = None) -> DictConfig:
    """Load config from ``default.yaml`` (or a file given via ``--config``) and
    apply ``key=value`` overrides from the remaining CLI args.

    Parameters
    ----------
    argv : list[str], optional
        Argument list (defaults to ``sys.argv[1:]``). Recognized forms:
          * ``--config PATH`` — use PATH as the base config instead of default.
          * ``key=value`` — override a nested key (e.g. ``optim.lr=5e-4``).
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    # Parse --config out first; the rest are key=value overrides.
    config_path = DEFAULT_CONFIG_PATH
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            if i + 1 >= len(argv):
                raise SystemExit("--config requires a path argument")
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
    cli = OmegaConf.from_cli(overrides)
    # cli takes precedence over base.
    cfg = OmegaConf.merge(base, cli)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Persist the *resolved* config to ``path`` (useful inside a run directory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def to_yaml(cfg: DictConfig) -> str:
    """Return the config as a YAML string (for logging)."""
    return OmegaConf.to_yaml(cfg, resolve=True)


__all__ = ["load_config", "save_config", "to_yaml", "DEFAULT_CONFIG_PATH"]
