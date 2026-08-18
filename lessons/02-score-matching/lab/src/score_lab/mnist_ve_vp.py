"""VE vs VP on MNIST — theory.md §9 «Score и ε — одна сеть в разных одеждах»
and §10 «VE vs VP: два способа портить картинку».

Two ways to noise an image, one network recipe:

* **VE** (variance exploding, NCSN): ``x_t = x₀ + σ_t·ε`` — purely additive,
  the signal is never scaled; σ follows a *geometric* ladder from ≈80 (pure
  noise, SNR ≈ 0) down to ≈0.01.
* **VP** (variance preserving, DDPM — lesson 01): ``x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε``
  with linear β; the signal shrinks as noise grows so the variance stays 1.

§9 is the punchline of the whole lesson: an ε-predicting network *is* a score
network. For VP, ``x_t = √ᾱ·x₀ + √(1−ᾱ)·ε`` gives ``∇_{x_t} log p ≡ −ε/√(1−ᾱ_t)``;
for VE, ``x_t = x₀ + σ_t·ε`` gives ``∇_{x_t} log p ≡ −ε/σ_t``. In both cases
``score = −ε / (std of the noise actually added)``, so the same U-Net trained
to predict ε is a score model — only the σ(t) bookkeeping differs.

We train ε-prediction in *both* branches (equal budget, shared loop) and convert
to scores via those identities; the samplers are deliberately simple (see
README "Known simplifications").
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import geometric_sigma_ladder

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# ---------------------------------------------------------------------------
# Schedules (§10)
# ---------------------------------------------------------------------------

@dataclass
class VESchedule:
    """Variance-exploding: additive noise with a geometric σ ladder (§10)."""

    sigma_max: float = 80.0
    sigma_min: float = 0.01
    n_levels: int = 200

    def __post_init__(self) -> None:
        self.sigmas = geometric_sigma_ladder(self.sigma_max, self.sigma_min,
                                             self.n_levels)

    def forward(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """``x_t = x₀ + σ_t·ε`` — additive, x₀ keeps its scale (§10)."""
        sigma = self.sigmas.to(x0.device)[t].reshape(-1, 1, 1, 1)
        return x0 + sigma * eps

    def sigma(self, t: torch.Tensor, device=None) -> torch.Tensor:
        return self.sigmas.to(t.device if device is None else device)[t]


@dataclass
class VPSchedule:
    """Variance-preserving: linear β, exactly as in lesson 01 (§10, lab 1)."""

    beta_start: float = 1e-4
    beta_end: float = 0.02
    num_timesteps: int = 1000

    def __post_init__(self) -> None:
        self.betas = torch.linspace(self.beta_start, self.beta_end,
                                    self.num_timesteps, dtype=torch.float64)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """``x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε`` — signal scaled, variance stays 1."""
        ab = self.alphas_cumprod.to(x0.device)[t].reshape(-1, 1, 1, 1).float()
        return ab.sqrt() * x0 + (1.0 - ab).clamp_min(0).sqrt() * eps

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise std actually added: √(1−ᾱ_t) — the score↔ε bridge denominator."""
        return (1.0 - self.alphas_cumprod.to(t.device)[t]).clamp_min(0).sqrt().float()

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas_cumprod.to(t.device)[t].float()


# ---------------------------------------------------------------------------
# Training: ε-prediction in both branches (§9)
# ---------------------------------------------------------------------------

def train_eps_model(
    model: torch.nn.Module,
    loader,
    schedule,
    *,
    epochs: int,
    lr: float,
    device: str,
    seed: int = 42,
    log_csv: Path | None = None,
) -> list[float]:
    """One shared loop for both branches: sample t, noise the batch, fit ε.

    Only the schedule (how x_t is built and which t's exist) differs between
    VE and VP — the loss is the same ε-MSE, which §9 shows is a score-matching
    loss in disguise (``score = −ε/σ_t``).
    """
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_t = schedule.n_levels if hasattr(schedule, "n_levels") else schedule.num_timesteps

    csv_file = None
    if log_csv is not None:
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "loss"])

    epoch_losses: list[float] = []
    t0 = time.time()
    for epoch in range(epochs):
        running, n = 0.0, 0
        for x0, _labels in loader:
            x0 = x0.to(device)
            t = torch.randint(0, n_t, (x0.shape[0],), device=device)
            eps = torch.randn_like(x0)
            xt = schedule.forward(x0, t, eps)
            pred = model(xt, t)
            loss = torch.mean((pred - eps) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        epoch_losses.append(running / max(n, 1))
        print(f"  [epoch {epoch+1}/{epochs}] loss={epoch_losses[-1]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if csv_file is not None:
            writer.writerow([epoch, round(epoch_losses[-1], 5)])
    if csv_file is not None:
        csv_file.close()
    return epoch_losses


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_vp(
    model, schedule: VPSchedule, shape, generator=None, device="cpu"
) -> torch.Tensor:
    """Ancestral DDPM sampling — the lesson-01 reverse process (§10, lab 1)."""
    model.eval()
    x = torch.randn(shape, generator=generator, device=device)
    betas = schedule.betas.to(device).float()
    alphas = schedule.alphas.to(device).float()
    ab = schedule.alphas_cumprod.to(device).float()
    ab_prev = torch.cat([torch.ones(1, device=device), ab[:-1]])
    sigmas = (betas * (1 - ab_prev) / (1 - ab).clamp_min(1e-20)).clamp_min(0).sqrt()

    for t in reversed(range(schedule.num_timesteps)):
        tb = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = model(x, tb)
        mean = (1.0 / alphas[t].sqrt()) * (
            x - ((1 - alphas[t]) / (1 - ab[t]).sqrt()) * eps
        )
        if t > 0:
            z = torch.randn(shape, generator=generator, device=device)
            x = mean + sigmas[t] * z
        else:
            x = mean
    return x


@torch.no_grad()
def sample_ddim(
    model, schedule: VPSchedule, shape, n_steps: int = 100, eta: float = 0.0,
    generator=None, device="cpu",
) -> torch.Tensor:
    """DDIM (η=0 default): the classical discrete form of the VP PF-ODE.

    Subsamples ``n_steps`` timesteps from the schedule and jumps directly
    between them: ``x_{τ₋₁} = √ᾱ_{τ₋₁}·(x_τ − √(1−ᾱ_τ)·ε̂)/√ᾱ_τ +
    √(1−ᾱ_{τ₋₁})·ε̂`` (+ η-scaled noise for η > 0). With η = 0 the map is
    deterministic — same role as the ODE solvers in ``solvers.py``, but
    derived from the discrete posterior instead of integrating the ODE.
    NFE = n_steps.
    """
    model.eval()
    ab = schedule.alphas_cumprod.to(device).float()
    tau = torch.linspace(schedule.num_timesteps - 1, 0, n_steps).long()
    x = torch.randn(shape, generator=generator, device=device)
    for i in range(len(tau)):  # tau itself already runs T-1 → 0
        tb = torch.full((shape[0],), tau[i], device=device, dtype=torch.long)
        eps = model(x, tb)
        ab_t = ab[tau[i]]
        ab_prev = ab[tau[i + 1]] if i + 1 < len(tau) else torch.ones((), device=device)
        x0_hat = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-8)
        dir = (1 - ab_prev).clamp_min(0).sqrt() * eps
        x = ab_prev.sqrt() * x0_hat + dir
        if eta > 0 and i > 0:
            sigma = eta * ((1 - ab_prev) / (1 - ab_t).clamp_min(1e-8)).sqrt() \
                * (1 - ab_t / ab_prev).sqrt()
            x = x + sigma * torch.randn(shape, generator=generator, device=device)
    return x


@torch.no_grad()
def sample_ve(
    model, schedule: VESchedule, shape, generator=None, device="cpu",
    euler_sub: int = 5, corrector_steps: int = 10, corrector_scale: float = 0.15,
) -> torch.Tensor:
    """Simplified VE sampling: Euler drift down the σ ladder + Langevin corrector.

    Two moves per level (both are "simplified NCSN", documented in the README):

    *Predictor* — Euler step on the probability-flow ODE ``dx = −dσ·ε_θ``
    (from ``x = x₀ + σε``: ``x(σ−Δσ) = x(σ) − Δσ·ε``), subdivided ``euler_sub``
    times per level. A raw predict-x̂₀ jump (``x̂₀ = x − σε_θ``) does NOT work
    here: at high σ an ε-error δ gives an x̂₀-error of σ·δ, which explodes the
    accumulation — VP's ancestral step is protected by its √(1−ᾱ) weighting,
    this Euler+corrector scheme is VE's counterpart.

    *Corrector* — a couple of annealed-Langevin steps at the new σ (the same
    ``x ← x + (α/2)s + √α z``, α = scale·σ², as the 2D part of this lab):
    keeps the cloud at the right spread for its level.
    """
    model.eval()
    sigmas = schedule.sigmas.to(device)
    g = generator
    x = torch.randn(shape, generator=g, device=device) * sigmas[0]
    for i in reversed(range(schedule.n_levels)):
        for _ in range(euler_sub):
            tb = torch.full((shape[0],), i, device=device, dtype=torch.long)
            nxt = sigmas[i - 1] if i > 0 else torch.zeros((), device=device)
            ds = (sigmas[i] - nxt) / euler_sub
            x = x - ds * model(x, tb)
        if i > 0:
            tn = torch.full((shape[0],), i - 1, device=device, dtype=torch.long)
            s = float(sigmas[i - 1])
            alpha = corrector_scale * s * s
            for _ in range(corrector_steps):
                score = -model(x, tn) / s          # §9: score = −ε/σ
                z = torch.randn(shape, generator=g, device=device)
                x = x + 0.5 * alpha * score + alpha ** 0.5 * z
    return x


# ---------------------------------------------------------------------------
# Data + image grids
# ---------------------------------------------------------------------------

def mnist_loaders(data_dir: str, batch_size: int, num_workers: int = 2):
    """Normalized MNIST loaders (same constants as lesson 01)."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    return DataLoader(train, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True)


def save_sample_grid(x: torch.Tensor, path: Path, title: str, n_cols: int = 8) -> None:
    """Save a grid of generated digits, un-normalized back to [0, 1]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = x.detach().cpu() * MNIST_STD + MNIST_MEAN  # undo normalization
    x = x.clamp(0, 1)
    n = x.shape[0]
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(0.75 * n_cols, 0.75 * n_rows))
    for i in range(n_rows * n_cols):
        ax = axes.flat[i] if n > n_cols else axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(x[i, 0], cmap="gray_r")
    fig.suptitle(title, y=1.005)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
