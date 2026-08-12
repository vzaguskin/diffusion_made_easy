"""Model factory: pick ε_θ by ``cfg.model.type``.

Both architectures share the same contract: ``forward(x, t) -> eps`` where the
output has the same shape as ``x``, and the network predicts the noise only
(theory.md §15). Switching between them is a one-line config change.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .mlp import MLP
from .unet import UNet, UNetConfig


def build_model(cfg: Any) -> nn.Module:
    """Build an ε_θ network from the ``model`` section of the config.

    ``cfg`` is expected to have a ``model`` sub-config with at least ``type``
    (``"mlp"`` or ``"unet"``). Extra fields are forwarded to the model constructor.
    """
    model_cfg = cfg.model
    mtype = getattr(model_cfg, "type", "unet")

    if mtype == "mlp":
        return MLP(
            image_size=getattr(model_cfg, "image_size", 28),
            in_channels=getattr(model_cfg, "in_channels", 1),
            hidden_dim=getattr(model_cfg, "hidden_dim", 512),
            num_blocks=getattr(model_cfg, "mlp_blocks", 4),
            time_embed_dim=getattr(model_cfg, "mlp_time_embed_dim", 128),
            use_local_conv=getattr(model_cfg, "use_local_conv", True),
            local_channels=getattr(model_cfg, "local_channels", 32),
        )
    if mtype == "unet":
        unet_cfg = UNetConfig(
            in_channels=getattr(model_cfg, "in_channels", 1),
            out_channels=getattr(model_cfg, "out_channels", 1),
            base_channels=getattr(model_cfg, "base_channels", 64),
            channel_mults=tuple(getattr(model_cfg, "channel_mults", (1, 2, 4))),
            num_blocks=getattr(model_cfg, "unet_blocks", 2),
            time_embed_dim=getattr(model_cfg, "unet_time_embed_dim", 256),
            dropout=getattr(model_cfg, "dropout", 0.0),
        )
        return UNet(unet_cfg)

    raise ValueError(
        f"Unknown model.type='{mtype}'. Expected 'mlp' or 'unet' (theory.md uses eps-prediction)."
    )


__all__ = ["MLP", "UNet", "UNetConfig", "build_model"]
