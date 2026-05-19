"""Config loading. Default + experiment YAML overlay."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _deep_update(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load default.yaml then deep-merge the experiment config over it."""
    with open(DEFAULT_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    p = Path(path).resolve()
    if p != DEFAULT_CONFIG:
        with open(p, "r") as f:
            overlay = yaml.safe_load(f) or {}
        cfg = _deep_update(cfg, overlay)
    cfg["_config_path"] = str(p)
    return cfg


def run_dir(cfg: dict[str, Any]) -> Path:
    out = REPO_ROOT / cfg["output_root"] / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    return out
