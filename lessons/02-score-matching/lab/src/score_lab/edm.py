"""EDM preconditioning for the VE branch — theory.md §17.

The naive eps-parameterization of this lab shows three documented symptoms
(unbalanced per-level loss, background speckles, a sharp sampling-budget
peak). EDM (Karras et al. 2022) addresses all three at the model level:

* the network F_θ (the same MiniUNet, conditioned on sigma as a float)
  predicts a *residual* of the denoiser, assembled as
  ``D = c_skip·x + c_out·F(c_in·x, sigma)`` with
  ``c_in = 1/sqrt(σ² + σ_d²)``, ``c_out = σσ_d/sqrt(σ² + σ_d²)``,
  ``c_skip = σ_d²/(σ² + σ_d²)`` — input and output stay ~unit scale at every
  sigma, and D → x as σ → 0 without the net learning it;
* the loss ``λ(σ)·‖D(x+σε,σ) − x₀‖²`` with ``λ = (σ²+σ_d²)/(σσ_d)²`` has a
  unit-variance effective target at every sigma (§17, tested below);
* sigma is drawn log-normal during training (P_mean, P_std) instead of
  uniformly over a ladder; sampling is Heun on the VE PF-ODE ``dx/dσ =
  (x − D)/σ`` over a polynomial sigma grid (ρ = 7, NFE = 2N − 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EDMModel(nn.Module):
    """Preconditioned denoiser D_θ(x, σ) around a bare network F_θ."""

    def __init__(self, net: nn.Module, sigma_data: float = 1.0) -> None:
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x: [B,1,H,W]; sigma: [B] (float conditioning, not a ladder index)."""
        sd2 = self.sigma_data ** 2
        s2 = sigma.reshape(-1, 1, 1, 1) ** 2
        c_skip = sd2 / (s2 + sd2)
        c_out = (sigma.reshape(-1, 1, 1, 1) * self.sigma_data) / (s2 + sd2).sqrt()
        c_in = 1.0 / (s2 + sd2).sqrt()
        return c_skip * x + c_out * self.net(c_in * x, sigma)


def edm_loss(model: EDMModel, x0: torch.Tensor, p_mean: float = -1.2,
             p_std: float = 1.2) -> torch.Tensor:
    """λ(σ)-weighted denoiser loss with log-normal sigma sampling (§17)."""
    sigma = torch.randn(x0.shape[0], device=x0.device) * p_std + p_mean
    sigma = sigma.exp().clamp_min(1e-4)
    eps = torch.randn_like(x0)
    xt = x0 + sigma.reshape(-1, 1, 1, 1) * eps
    resid = model(xt, sigma) - x0
    weight = (sigma ** 2 + model.sigma_data ** 2) / (sigma * model.sigma_data) ** 2
    return (weight.reshape(-1, 1, 1, 1) * resid ** 2).mean()


@torch.no_grad()
def edm_heun_sample(model: EDMModel, shape, sigma_max: float = 5.0,
                    sigma_min: float = 0.01, n_steps: int = 40, rho: float = 7.0,
                    generator=None, device="cpu") -> tuple[torch.Tensor, int]:
    """Deterministic Heun sampler on the VE PF-ODE (Karras Alg. 2, no churn).

    The ODE in sigma-parameterization: dx/dσ = (x − D_θ(x, σ))/σ. The grid is
    polynomial in σ^{1/ρ} (denser at small sigma, where the ODE curves).
    Returns (samples, NFE) with NFE = 2·n_steps − 1 (last step is Euler).
    """
    i = torch.arange(n_steps, device=device, dtype=torch.float64)
    s_max, s_min = sigma_max ** (1.0 / rho), sigma_min ** (1.0 / rho)
    sigmas = (s_max + i / (n_steps - 1) * (s_min - s_max)) ** rho
    x = torch.randn(shape, generator=generator, device=device) * sigma_max
    nfe = 0
    for k in range(n_steps - 1):
        s, s_next = float(sigmas[k]), float(sigmas[k + 1])
        d = (x - model(x, torch.full((x.shape[0],), s, device=device))) / s
        nfe += 1
        x_euler = x + (s_next - s) * d
        if k + 2 < n_steps:                      # Heun update (skip the last)
            d2 = (x_euler - model(x_euler,
                                  torch.full((x.shape[0],), s_next, device=device))) / s_next
            nfe += 1
            x = x + (s_next - s) * 0.5 * (d + d2)
        else:
            x = x_euler
    return x, nfe


def edm_sigma_grid(sigma_max: float, sigma_min: float, n_steps: int,
                   rho: float = 7.0) -> torch.Tensor:
    """The Karras polynomial grid — exposed for per-level evaluation."""
    i = torch.arange(n_steps, dtype=torch.float64)
    s_max, s_min = sigma_max ** (1.0 / rho), sigma_min ** (1.0 / rho)
    return (s_max + i / (n_steps - 1) * (s_min - s_max)) ** rho
