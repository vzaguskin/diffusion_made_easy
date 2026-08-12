"""Validation metrics for the diffusion model — no heavy external networks.

Why these metrics?
    Plain MSE loss hides *where* the model is struggling. theory.md §15 explicitly
    drops the per-``t`` weight, so watching the loss bucketed by ``t`` shows the
    student exactly what that weight was hiding (the model would otherwise barely
    learn on high-noise steps).

    For sample quality, FID/IS need Inception-v3 — which is ~80MB, was trained on
    ImageNet (bad fit for MNIST), and is overkill for a teaching lab. Instead we
    use simple **nearest-neighbor** proxies in pixel space:

    * **coverage** — fraction of real images that are some generated image's NN.
      High coverage ≈ the model covers the data manifold; low coverage ≈ mode
      collapse (the model draws only a few kinds of digits).
    * **mode-collapse distribution** — the class histogram of the real images that
      generated images map to (using the test-set labels). If the model collapses,
      a few digit classes dominate.

    The README documents how to swap pixel space for CNN-feature space for a
    sharper metric — that's left as an exercise, not a dependency.
"""

from __future__ import annotations

import torch


def loss_by_t_bucket(
    eps_pred: torch.Tensor,
    eps_target: torch.Tensor,
    t: torch.Tensor,
    num_timesteps: int,
    num_buckets: int = 10,
) -> torch.Tensor:
    """Mean MSE in each of ``num_buckets`` intervals over ``t in [0, T)``.

    Returns
    -------
    Tensor[num_buckets]
        ``out[i]`` is the mean ``(eps_pred - eps_target) ** 2`` for samples whose
        ``t`` falls in bucket ``i``. Buckets with no samples get NaN (caller can
        mask them before logging).
    """
    # Per-sample squared error, averaged over everything except the batch dim.
    se = (eps_pred - eps_target) ** 2
    while se.ndim > 1:
        se = se.mean(dim=-1)  # -> [B]

    bucket_edges = torch.linspace(0, num_timesteps, num_buckets + 1, device=t.device)
    # bucket_idx[b] = which bucket t[b] falls into (0 .. num_buckets-1)
    bucket_idx = torch.clamp(
        torch.bucketize(t, bucket_edges[1:-1]), 0, num_buckets - 1
    )
    out = torch.full((num_buckets,), float("nan"), device=eps_pred.device)
    for b in range(num_buckets):
        mask = bucket_idx == b
        if mask.any():
            out[b] = se[mask].mean()
    return out


@torch.no_grad()
def nearest_neighbor_indices(
    queries: torch.Tensor, candidates: torch.Tensor, chunk: int = 512
) -> torch.Tensor:
    """For each query, return the index of its L2-nearest candidate (pixel space).

    Both tensors are flattened to ``[N, D]``. Computed in chunks to bound GPU
    memory when ``candidates`` is large.

    Returns
    -------
    Tensor[queries.shape[0]]
        Index into ``candidates`` of each query's nearest neighbor.
    """
    q = queries.reshape(queries.shape[0], -1).float()
    c = candidates.reshape(candidates.shape[0], -1).float()
    # Use squared distance = |q|^2 + |c|^2 - 2 q·c; argmin over candidates.
    c_sq = (c * c).sum(dim=1)  # [M]
    nn = torch.empty(q.shape[0], dtype=torch.long, device=q.device)
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        qb = q[start:end]
        # distances[start:end, :]
        d = (qb * qb).sum(dim=1, keepdim=True) + c_sq.unsqueeze(0) - 2.0 * qb @ c.T
        nn[start:end] = d.argmin(dim=1)
    return nn


@torch.no_grad()
def coverage_and_mode_collapse(
    samples: torch.Tensor,
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Coverage + per-class NN distribution for generated ``samples``.

    Parameters
    ----------
    samples : Tensor[N, ...]
        Generated images.
    real_images : Tensor[M, ...]
        A pool of real images to compare against (e.g. a slice of the test set).
    real_labels : Tensor[M]
        Class labels (0-9) for the real images, used for the mode-collapse histogram.

    Returns
    -------
    coverage : float
        Fraction of the ``M`` real images that are the nearest neighbor of *some*
        generated sample. Range [0, 1]; higher is better (more diverse coverage).
    nn_class_hist : Tensor[10]
        Histogram (counts) of the classes of the NN real images for each generated
        sample. If the model has collapsed, a few bins dominate.
    """
    nn_idx = nearest_neighbor_indices(samples, real_images)
    # Coverage: how many distinct real images are hit.
    hit = torch.zeros(real_images.shape[0], dtype=torch.bool, device=real_images.device)
    hit[nn_idx] = True
    coverage = hit.float().mean().item()

    # Mode-collapse: class distribution of the NN real images.
    nn_labels = real_labels[nn_idx]
    nn_class_hist = torch.bincount(nn_labels, minlength=10).to(torch.long)
    return coverage, nn_class_hist


__all__ = ["loss_by_t_bucket", "nearest_neighbor_indices", "coverage_and_mode_collapse"]
