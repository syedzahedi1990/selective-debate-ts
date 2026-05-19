"""File-system cache for LLM responses keyed by a stable hash."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LLMCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        sub = self.root / key[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._path(key).write_text(json.dumps(value, indent=2), encoding="utf-8")
