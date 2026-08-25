import torch


def masked_mse_loss(pred, target, valid):
    """Mean squared error over explicitly valid channel values only."""
    valid = valid.to(pred.dtype)
    return ((pred - target).square() * valid).sum() / valid.sum().clamp_min(1.0)

