from __future__ import annotations

import importlib
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def import_object(path: str):
    """Import ``pkg.module:Name`` or ``pkg.module.Name``."""
    if ":" in path:
        module_name, object_name = path.split(":", 1)
    else:
        module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def amp_dtype_from_name(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def cosine_warmup_lambda(step: int, total_steps: int, warmup_steps: int, min_ratio: float):
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1.0e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def save_checkpoint(path: str | Path, payload: dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
