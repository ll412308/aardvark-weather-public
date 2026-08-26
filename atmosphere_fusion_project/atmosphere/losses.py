from __future__ import annotations

import torch


def latent_mse(pred, target, available, density=None,
               use_density_mask=False, density_threshold=1.0e-6):
    """Instrument-latent MSE, optionally restricted by raw SetConv coverage."""
    available_mask = available.to(pred.dtype).view(-1, 1, 1, 1)
    if use_density_mask:
        if density is None:
            raise ValueError("density is required when use_density_mask=True")
        mask = density.gt(float(density_threshold)).to(pred.dtype)
        mask = mask * available_mask
    else:
        mask = available_mask.expand(-1, 1, pred.shape[-2], pred.shape[-1])
    error = (pred - target).square() * mask
    denom = mask.sum() * pred.shape[1]
    return error.sum() / denom.clamp_min(1.0)
