from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def require(mapping: dict, key: str):
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]
