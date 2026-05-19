"""CSV + simple LaTeX table writers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def write_csv(path: str | Path, rows: list[dict[str, Any]], cols: Iterable[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return
    fieldnames = list(cols) if cols else list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_latex(path: str | Path, rows: list[dict[str, Any]], cols: Iterable[str], caption: str = "", label: str = "") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = list(cols)
    lines = ["\\begin{table}[h]", "\\centering"]
    lines.append("\\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(c.replace("_", "\\_") for c in cols) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v).replace("_", "\\_"))
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if caption:
        lines.append("\\caption{" + caption + "}")
    if label:
        lines.append("\\label{" + label + "}")
    lines.append("\\end{table}")
    p.write_text("\n".join(lines), encoding="utf-8")
