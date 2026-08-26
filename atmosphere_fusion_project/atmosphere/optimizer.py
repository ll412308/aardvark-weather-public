"""Small optimizer factory used by train_fusion.py."""

import torch


def build_optimizer(model, name="adamw", lr=1.0e-4, weight_decay=1.0e-4):
    name = str(name).lower()
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
    raise ValueError(f"Unknown optimizer: {name}. Use 'adam' or 'adamw'.")
