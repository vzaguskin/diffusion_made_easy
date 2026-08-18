"""Continuous-time SDE / PF-ODE formulations — theory.md §11–§14.

The lab's eps-models are trained on *integer* conditioning indices (VE: a
geometric sigma-ladder level, VP: a DDPM timestep). This module wraps them in
the shared continuous-time SDE form ``dx = f(x, t)dt + g(t)dW`` with
``t ∈ [0, 1]`` (§12), mapping ``t`` onto the nearest training index. From one
eps-model we get both reverse-time dynamics (§13 Anderson):

* reverse SDE:  ``dx = [f(x,t) − g²(t)·s_θ(x,t)] dt + g(t) dW̄``
* probability-flow ODE (§14):  ``dx/dt = f(x,t) − ½·g²(t)·s_θ(x,t)``

via the §9 score↔eps bridge: VE ``s = −ε/σ(t)``, VP ``s = −ε/√(1−ᾱ_t)``.

Sampling runs in *reverse time* (t: 1 → 0) from the branch's prior; solvers
pass negative ``dt`` and both drifts below are stated in forward-time form.
"""

from __future__ import annotations

import torch

from .mnist_ve_vp import VESchedule, VPSchedule


class _ContinuousBranch:
    """Shared plumbing: index quantization + eps→score + drifts."""

    def __init__(self, schedule, n_levels: int) -> None:
        self.schedule = schedule
        self.n_levels = n_levels

    # --- time bookkeeping ---------------------------------------------------
    def t_to_index(self, t: torch.Tensor) -> torch.Tensor:
        """Map t ∈ [0,1] onto the nearest integer conditioning index."""
        return (t.clamp(0.0, 1.0) * (self.n_levels - 1)).round().long()

    @staticmethod
    def _per_sample(v: torch.Tensor, batch: int) -> torch.Tensor:
        """Broadcast a [B] or scalar tensor to the per-sample [B,1,1,1] shape."""
        if v.numel() == 1:
            v = v.expand(batch)
        return v.reshape(-1, 1, 1, 1)

    # --- model access ---------------------------------------------------------
    def eps(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:  # scalar grid point → per-sample batch
            t = t.expand(x.shape[0])
        return model(x, self.t_to_index(t))

    def score(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        std = self._per_sample(self.noise_std(t, x.shape[0], x.device), x.shape[0])
        return -self.eps(model, x, t) / std.to(x.device)

    # --- drifts (forward-time form; integrate with dt < 0) --------------------
    def reverse_sde_drift(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(x,t) − g²(t)·s_θ(x,t) — the §13 reverse-SDE drift."""
        g2 = self.diffusion(t, x.shape[0], x.device).to(x.device) ** 2
        return self.forward_drift(x, t.to(x.device)) - g2 * self.score(model, x, t)

    def pf_ode_drift(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(x,t) − ½·g²(t)·s_θ(x,t) — the §14 probability-flow ODE drift."""
        g2 = self.diffusion(t, x.shape[0], x.device).to(x.device) ** 2
        return self.forward_drift(x, t.to(x.device)) - 0.5 * g2 * self.score(model, x, t)

    def forward_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def diffusion(self, t: torch.Tensor, batch: int, device) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def noise_std(self, t: torch.Tensor, batch: int, device) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def prior_sample(self, shape, generator=None, device="cpu") -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward_noise(self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """The branch's forward noising at continuous t (VE: additive, VP: ᾱ-mix)."""
        raise NotImplementedError


class ContinuousVE(_ContinuousBranch):
    """VE (NCSN): ``dx = σ(t)·√(2 ln r)·dW`` — additive noise, r = σ_max/σ_min.

    ``σ(t) = σ_min·r^t`` is the *total* noise std at time t (the same
    geometric family as the training ladder). The forward SDE must accumulate
    ``∫₀ᵗ g²ds = σ(t)²``, hence ``g(t) = σ(t)·√(2 ln r)`` — not σ(t) itself.
    """

    def __init__(self, schedule: VESchedule) -> None:
        super().__init__(schedule, schedule.n_levels)
        self.sigma_min = schedule.sigma_min
        self.sigma_max = schedule.sigma_max
        self._log_r = torch.log(torch.tensor(self.sigma_max / self.sigma_min))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def forward_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor, batch: int, device) -> torch.Tensor:
        g = self.sigma(t.to(device)) * (2.0 * self._log_r.to(device)).sqrt()
        return self._per_sample(g, batch)

    def noise_std(self, t: torch.Tensor, batch: int, device) -> torch.Tensor:
        t = t.to(device)
        if t.numel() == 1:
            t = t.expand(batch)
        return self.sigma(t)

    def prior_sample(self, shape, generator=None, device="cpu") -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device) * self.sigma_max

    def forward_noise(self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        std = self._per_sample(self.noise_std(t, x.shape[0], x.device), x.shape[0])
        return x + std * eps


class ContinuousVP(_ContinuousBranch):
    """VP (DDPM): ``dx = −½β(t)x·dt + √β(t)dW`` — variance stays 1 (§10).

    β(t) is the linear interpolation of the discrete lesson-01 betas, so the
    continuous process passes through exactly the training noise levels.
    """

    def __init__(self, schedule: VPSchedule) -> None:
        super().__init__(schedule, schedule.num_timesteps)
        self.betas = schedule.betas.float()          # [T], β_start..β_end
        self._dt_scale = len(self.betas) - 1         # t∈[0,1] → T−1 discrete steps

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Continuous β(t) = discrete β scaled so ∫₀¹ β = Σβ_discrete."""
        t = t.to(self.betas.device)
        pos = t.clamp(0.0, 1.0) * (len(self.betas) - 1)
        lo = pos.floor().long().clamp(max=len(self.betas) - 2)
        frac = (pos - lo.float()).to(self.betas.dtype)
        interp = self.betas[lo] * (1 - frac) + self.betas[lo + 1] * frac
        return interp * self._dt_scale

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """ᾱ(t) = exp(−∫β): the continuous version of the cumprod."""
        return torch.exp(-self._beta_integral(t))

    def _beta_integral(self, t: torch.Tensor) -> torch.Tensor:
        """∫₀^t β(s)ds for the piecewise-linear β (exact via cumsum + trapezoid)."""
        t = t.to(self.betas.device)
        pos = t.clamp(0.0, 1.0) * (len(self.betas) - 1)
        lo = pos.floor().long().clamp(max=len(self.betas) - 2)
        frac = (pos - lo.float()).to(self.betas.dtype)
        seg = self.betas[: len(self.betas) - 1]
        full = torch.cumsum(seg, dim=0)
        full = torch.cat([torch.zeros(1, dtype=self.betas.dtype), full])
        # full[k] = Σ of the first k discrete betas; partial segment via trapezoid
        partial = (self.betas[lo] + self.betas[lo + 1]) / 2 * frac
        return full[lo] + partial

    def forward_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta = self.beta(t).to(x.device)
        return -0.5 * self._per_sample(beta, x.shape[0]) * x

    def diffusion(self, t: torch.Tensor, batch: int, device) -> torch.Tensor:
        g = self.beta(t).sqrt().to(device)
        return self._per_sample(g, batch)

    def noise_std(self, t: torch.Tensor, batch: int, device) -> torch.Tensor:
        return (1.0 - self.alpha_bar(t)).clamp_min(1e-12).sqrt().to(device)

    def prior_sample(self, shape, generator=None, device="cpu") -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device)

    def forward_noise(self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        ab = self._per_sample(self.alpha_bar(t), x.shape[0]).to(x.device)
        return ab.sqrt() * x + (1 - ab).clamp_min(0).sqrt() * eps
