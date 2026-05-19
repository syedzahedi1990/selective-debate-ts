"""IO for forecast cards and private labels.

We keep cards and labels in **separate** JSONL files. Tests assert that the
card file never contains ground-truth fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from sdfts.utils.io import read_jsonl, write_jsonl


CARDS_FILENAME = "forecast_cards.jsonl"
LABELS_FILENAME = "private_labels.jsonl"


def cards_path(run_dir: Path) -> Path:
    return Path(run_dir) / "cards" / CARDS_FILENAME


def labels_path(run_dir: Path) -> Path:
    return Path(run_dir) / "cards" / LABELS_FILENAME


def write_cards(run_dir: Path, cards: list[dict[str, Any]]) -> int:
    return write_jsonl(cards_path(run_dir), cards)


def read_cards(run_dir: Path) -> Iterator[dict[str, Any]]:
    return read_jsonl(cards_path(run_dir))


def write_labels(run_dir: Path, labels: list[dict[str, Any]]) -> int:
    return write_jsonl(labels_path(run_dir), labels)


def read_labels(run_dir: Path) -> Iterator[dict[str, Any]]:
    return read_jsonl(labels_path(run_dir))
