"""Generate publication-style figures from evaluation outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sdfts.cards.io import read_cards
from sdfts.config import run_dir
from sdfts.evaluation.plots import (
    plot_ablation_bars,
    plot_calibration,
    plot_disagreement_tensor,
    plot_risk_coverage,
    plot_risk_cost_pareto,
    plot_when_debate_helps,
)
from sdfts.utils.logging import get_logger


log = get_logger(__name__)


def make_figures(cfg: dict[str, Any]) -> dict[str, Any]:
    rd = run_dir(cfg)
    eval_dir = rd / "evaluation"
    figs_dir = rd / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ---------- risk-coverage curves ----------
    rc = json.loads((eval_dir / "risk_coverage_curves.json").read_text(encoding="utf-8"))
    plot_risk_coverage(
        {k: {"coverage": np.asarray(v["coverage"]), "risk": np.asarray(v["risk"])} for k, v in rc.items()},
        figs_dir / "fig3_risk_coverage.png",
    )

    # ---------- calibration ----------
    cal = json.loads((eval_dir / "calibration_pairs.json").read_text(encoding="utf-8"))
    plot_calibration(
        {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in cal.items()},
        figs_dir / "fig4_calibration.png",
    )

    # ---------- risk-cost Pareto ----------
    metrics = _read_csv(eval_dir / "metrics_summary.csv")
    methods = {}
    for r in metrics:
        cost = float(r.get("cost.n_calls") or 0.0)
        risk = float(r.get("mae_mean") or 0.0)
        methods[r["method"]] = {"cost": cost, "risk": risk}
    if methods:
        plot_risk_cost_pareto(methods, figs_dir / "fig2_risk_cost_pareto.png")

    # ---------- disagreement tensor heatmap (from one representative card) ----------
    cards = list(read_cards(rd))
    if cards:
        c0 = cards[len(cards) // 2]
        F = np.array([m["forecast"] for m in c0["candidate_models"]], dtype=np.float64)
        archs = [m["architecture"] for m in c0["candidate_models"]]
        regimes = [m["training_regime"] for m in c0["candidate_models"]]
        plot_disagreement_tensor(F, archs, regimes, figs_dir / "fig6_disagreement_tensor.png")

    # ---------- when debate helps (binned by disagreement) ----------
    by_regime = _when_debate_helps(rd)
    if by_regime:
        plot_when_debate_helps(by_regime, figs_dir / "fig5_when_debate_helps.png")

    # ---------- ablation chart ----------
    if metrics:
        plot_ablation_bars(
            sorted(metrics, key=lambda r: float(r["mae_mean"]))[:8],
            metric="mae_mean",
            out_path=figs_dir / "fig7_ablation.png",
        )

    log.info("Wrote figures to %s", figs_dir)
    return {"figures_dir": str(figs_dir)}


def _read_csv(p: Path) -> list[dict[str, Any]]:
    import csv

    out: list[dict[str, Any]] = []
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(dict(row))
    return out


def _when_debate_helps(rd: Path) -> dict[str, tuple[float, float]]:
    """Bin instances by disagreement; compare default vs debate MAE."""
    cards = {c["instance_id"]: c for c in read_cards(rd)}
    per_inst_path = rd / "evaluation" / "metrics_per_instance.jsonl"
    if not per_inst_path.exists():
        return {}
    base_name = "validation_weighted_ensemble"
    method_name = None
    for cand in ["selective_tool_verified_debate", "always_on_debate", "independent_agents_vote", "single_llm_auditor"]:
        if cand in {row["method"] for row in _read_jsonl(per_inst_path)}:
            method_name = cand
            break
    if method_name is None:
        return {}
    base = {row["instance_id"]: row["mae"] for row in _read_jsonl(per_inst_path) if row["method"] == base_name}
    meth = {row["instance_id"]: row["mae"] for row in _read_jsonl(per_inst_path) if row["method"] == method_name}
    bins = {"disag_low": [], "disag_high": [], "cp_low": [], "cp_high": []}
    for iid, c in cards.items():
        d = c.get("diagnostics", {})
        if iid not in base or iid not in meth:
            continue
        bins["disag_high" if d.get("overall_forecast_dispersion", 0) > 0.5 else "disag_low"].append((base[iid], meth[iid]))
        bins["cp_high" if d.get("changepoint_score", 0) > 0.5 else "cp_low"].append((base[iid], meth[iid]))
    out: dict[str, tuple[float, float]] = {}
    for k, pairs in bins.items():
        if pairs:
            b = float(np.mean([p[0] for p in pairs]))
            m = float(np.mean([p[1] for p in pairs]))
            out[k] = (b, m)
    return out


def _read_jsonl(p: Path):
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
