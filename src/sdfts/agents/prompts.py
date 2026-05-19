"""Versioned prompt loader.

Each prompt has a textual ``v1`` (etc.) on disk under ``prompts/``. We compute a
stable hash of the rendered prompt + model + card so cached results are tied to
a specific prompt version.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sdfts.config import REPO_ROOT


PROMPT_DIR = REPO_ROOT / "prompts"


PROMPT_FILES = {
    "shared_system": "shared_system.txt",
    "recursive_specialist": "recursive_specialist.txt",
    "h_step_specialist": "h_step_specialist.txt",
    "direct_multistep_specialist": "direct_multistep_specialist.txt",
    "foundation_specialist": "foundation_specialist.txt",
    "skeptic": "skeptic.txt",
    "judge": "judge.txt",
    "single_auditor": "single_auditor.txt",
}


def load_prompt(name: str) -> str:
    p = PROMPT_DIR / PROMPT_FILES[name]
    return p.read_text(encoding="utf-8")


def stable_hash(*objs: Any) -> str:
    h = hashlib.sha256()
    for o in objs:
        h.update(json.dumps(o, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()
