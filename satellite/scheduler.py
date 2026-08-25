"""Learning-rate scheduler helpers used by train_bamua.py."""

import math
import torch


def build_scheduler(optimizer, name="reduce_on_plateau", **kwargs):
    name = name.lower()
    if name == "none":
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(kwargs.get("factor", 0.5)),
            patience=int(kwargs.get("patience", 3)),
            min_lr=float(kwargs.get("min_lr", 1.0e-6)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(kwargs.get("step_size", 10)),
            gamma=float(kwargs.get("gamma", 0.5)),
        )
    if name == "warmup_cosine":
        total_steps = int(kwargs["total_steps"])
        steps_per_epoch = int(kwargs["steps_per_epoch"])
        warmup_steps = int(kwargs.get("warmup_epochs", 1) * steps_per_epoch)
        warmup_steps = min(warmup_steps, total_steps)
        base_lr = float(optimizer.param_groups[0]["lr"])
        min_lr = float(kwargs.get("min_lr", 0.0))
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

        def lr_multiplier(step):
            # Linear warmup: a small learning rate rises to the configured lr.
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            # Then smoothly decay from lr to min_lr with a half cosine curve.
            decay_steps = max(total_steps - warmup_steps, 1)
            progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    raise ValueError(
        f"Unknown scheduler: {name}. Use 'none', 'reduce_on_plateau', "
        "'step', or 'warmup_cosine'."
    )


def step_scheduler(scheduler, name, val_loss=None):
    if scheduler is None:
        return
    if name.lower() == "reduce_on_plateau":
        if val_loss is not None:
            scheduler.step(val_loss)
    else:
        scheduler.step()
