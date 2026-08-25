"""Small optimizer factory used by train_bamua.py."""

import torch


def build_optimizer(model, name="adamw", lr=1.0e-3, weight_decay=1.0e-4):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer: {name}. Use 'adam' or 'adamw'.")
