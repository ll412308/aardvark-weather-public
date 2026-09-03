"""Small optimizer factory used by train_fusion.py."""

import torch


def build_optimizer(model, name="adamw", lr=1.0e-4, weight_decay=1.0e-4):
    name = str(name).lower()
    # Frozen stages remain in the checkpoint but are deliberately excluded from
    # the optimizer, so separate fusion/forecast training uses less optimizer memory.
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The selected training mode has no trainable parameters")
    if name == "adam":
        return torch.optim.Adam(
            parameters, lr=float(lr), weight_decay=float(weight_decay)
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=float(lr), weight_decay=float(weight_decay)
        )
    raise ValueError(f"Unknown optimizer: {name}. Use 'adam' or 'adamw'.")
