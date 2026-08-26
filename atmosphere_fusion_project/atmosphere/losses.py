from __future__ import annotations

import torch


def latent_coverage_mse(pred, target, density, available, density_threshold=1.0e-6):
    """MSE only where the target instrument AE had observational support."""
    mask = density.gt(float(density_threshold)).to(pred.dtype)
    mask = mask * available.to(pred.dtype).view(-1, 1, 1, 1)
    error = (pred - target).square() * mask
    denom = mask.sum() * pred.shape[1]
    return error.sum() / denom.clamp_min(1.0)


def log_density_mse(pred_log_density, target_density, available):
    target = torch.log1p(target_density.clamp_min(0.0))
    mask = available.to(pred_log_density.dtype).view(-1, 1, 1, 1)
    error = (pred_log_density - target).square() * mask
    denom = mask.sum() * pred_log_density.shape[-2] * pred_log_density.shape[-1]
    return error.sum() / denom.clamp_min(1.0)
