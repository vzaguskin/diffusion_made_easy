"""Plain training loop for the score network (no Lightning here — see design D2).

Each step: draw a data batch, sample a σ level per example from the ladder
(theory.md §8), compute the DSM loss (§7), Adam step. Losses are logged to CSV
(overall + per-σ-level) so the run is inspectable without TensorBoard.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm

from .dsm import dsm_loss, sample_sigma_level


def train_score_model(
    model: torch.nn.Module,
    sampler: Callable[[int], torch.Tensor],
    sigmas: torch.Tensor,
    *,
    epochs: int = 30,
    steps_per_epoch: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 42,
    log_csv: Path | None = None,
    log_every: int = 20,
) -> list[float]:
    """Train ``model`` with multi-level DSM.

    Parameters
    ----------
    sampler : callable(n) -> Tensor[n, 2]
        Draws a data batch (e.g. ``dist.sample``).
    sigmas : Tensor[L]
        The noise ladder.
    log_csv : optional Path
        If given, per-step losses (plus a per-level breakdown) are appended
        to this CSV.

    Returns
    -------
    list of per-epoch average losses.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    sigmas = sigmas.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    csv_file = None
    if log_csv is not None:
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "step", "loss"] + [f"loss_sigma_{i}" for i in range(len(sigmas))])

    epoch_losses: list[float] = []
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        running = 0.0
        n_log = 0
        for _ in range(steps_per_epoch):
            x = sampler(batch_size).to(device)
            sigma = sample_sigma_level(sigmas, batch_size).to(device)
            loss, _ = dsm_loss(model, x, sigma)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            n_log += 1
            step += 1

            if csv_file is not None and step % log_every == 0:
                # Per-level breakdown for the logged step (small eval batch).
                with torch.no_grad():
                    x_eval = sampler(64).to(device)
                    row = [epoch, step, round(loss.item(), 5)]
                    for s in sigmas:
                        l, _ = dsm_loss(model, x_eval, s.unsqueeze(0).expand(64))
                        row.append(round(l.item(), 5))
                    writer.writerow(row)
        epoch_losses.append(running / max(n_log, 1))
        tqdm.write(f"[epoch {epoch+1:3d}/{epochs}] loss={epoch_losses[-1]:.4f} "
                   f"({time.time()-t0:.0f}s)")

    if csv_file is not None:
        csv_file.close()
    return epoch_losses
