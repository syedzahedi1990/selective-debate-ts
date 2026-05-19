"""Evaluate forecast-only, non-LLM, and LLM decision systems on the test set.

Outputs ``metrics_summary.csv``, ``metrics_per_instance.jsonl``, and stash
intermediate arrays for figure generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sdfts.cards.io import read_cards, read_labels
from sdfts.config import run_dir
from sdfts.evaluation.metrics import (
    arbitration_metrics,
    compute_forecast_metrics,
    cost_metrics,
    per_instance_mae,
)
from sdfts.evaluation.selective import (
    auprc,
    auroc,
    risk_coverage_curve,
    selective_risk_at_coverage,
)
from sdfts.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
)
from sdfts.evaluation.statistical_tests import bootstrap_ci, paired_bootstrap_diff
from sdfts.evaluation.tables import write_csv, write_latex
from sdfts.experiments.run_forecast_panel import load_panel_outputs
from sdfts.routers.baselines import (
    horizonwise_validation_best,
    median_ensemble,
    simple_mean_ensemble,
    validation_best,
    validation_weighted_ensemble,
)
from sdfts.routers.predict_router import predict_failure_prob
from sdfts.routers.train_router import load_router
from sdfts.utils.io import read_jsonl, write_jsonl
from sdfts.utils.logging import get_logger


log = get_logger(__name__)


def _by_id_test_targets(rd: Path) -> dict[str, np.ndarray]:
    outs = np.load(rd / "panel" / "targets.npz")
    return dict(outs)


def _decision_to_forecast(decision: dict[str, Any], card: dict[str, Any]) -> tuple[np.ndarray, str]:
    cands = {m["model_id"]: np.asarray(m["forecast"], dtype=np.float64) for m in card["candidate_models"]}
    dt = decision["decision_type"]
    if dt == "select_model" and decision.get("selected_model_ids"):
        mid = decision["selected_model_ids"][0]
        if mid in cands:
            return cands[mid], mid
    if dt == "ensemble":
        weights = decision.get("ensemble_weights") or {}
        # If empty weights but we got an ensemble decision, default to validation-weighted.
        if not weights:
            wts = {m["model_id"]: 1.0 / max(1.0, m["validation_metrics"]["mae"] + 1e-9)
                   for m in card["candidate_models"]}
            tot = sum(wts.values()) + 1e-12
            weights = {k: v / tot for k, v in wts.items()}
        f = np.zeros_like(next(iter(cands.values())))
        wsum = 0.0
        for mid, w in weights.items():
            if mid in cands:
                f = f + float(w) * cands[mid]
                wsum += float(w)
        if wsum > 0:
            f = f / wsum
        return f, "ensemble"
    if dt in {"abstain", "flag_failure"}:
        # Still emit a forecast (the default) so we can score risk-coverage.
        return np.asarray(card["default_decision"]["forecast"], dtype=np.float64), dt
    return np.asarray(card["default_decision"]["forecast"], dtype=np.float64), "default"


def _baseline_forecast(card: dict[str, Any], name: str) -> dict[str, Any]:
    table = {
        "validation_best": validation_best,
        "horizonwise_validation_best": horizonwise_validation_best,
        "simple_mean_ensemble": simple_mean_ensemble,
        "median_ensemble": median_ensemble,
        "validation_weighted_ensemble": validation_weighted_ensemble,
    }
    return table[name](card)


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    rd = run_dir(cfg)
    cards = list(read_cards(rd))
    labels = {l["instance_id"]: l for l in read_labels(rd)}
    if not cards:
        raise FileNotFoundError("No forecast cards found.")

    test_targets = _by_id_test_targets(rd)["test"]
    H = test_targets.shape[1]
    N = test_targets.shape[0]
    assert N == len(cards), f"cards/labels/test target mismatch: {len(cards)} vs {N}"

    # ----- Forecast-only baselines -----
    per_inst: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    rc_curves: dict[str, dict[str, list[float]]] = {}
    cal_pairs: dict[str, tuple[list[float], list[float]]] = {}

    forecast_only = list(cfg["decision_systems"]["forecast_only"])
    non_llm = list(cfg["decision_systems"]["non_llm"])

    # Cache router predictions for non-LLM systems.
    router_preds: dict[str, np.ndarray] = {}
    for r in cfg["routers"]["models"]:
        p = rd / "routers" / f"{r}.pkl"
        if p.exists():
            router_preds[r] = predict_failure_prob(load_router(p), cards)

    def add_method(name: str, forecasts: np.ndarray, selected_ids: list[str], conf: np.ndarray,
                    failure_prob: np.ndarray | None = None,
                    cost: dict[str, float] | None = None) -> None:
        errs = per_instance_mae(test_targets, forecasts)
        oracle_errs = np.array([labels[c["instance_id"]]["oracle_best_error"] for c in cards])
        default_errs = np.array([labels[c["instance_id"]]["default_error"] for c in cards])
        oracle_ids = [labels[c["instance_id"]]["oracle_best_model_id"] for c in cards]
        arb = arbitration_metrics(errs, oracle_errs, default_errs, selected_ids=selected_ids, oracle_ids=oracle_ids)
        ci = bootstrap_ci(errs, n_samples=int(cfg["evaluation"]["bootstrap_samples"]))

        # Selective risk: use 1 - confidence as "would abstain" proxy ordering.
        rc = risk_coverage_curve(errs, conf)
        rc_curves[name] = {"coverage": rc["coverage"].tolist(), "risk": rc["risk"].tolist()}
        sel_at_cov = selective_risk_at_coverage(errs, conf, cfg["evaluation"]["coverages"])

        # Failure-detection AUROC/AUPRC/Brier/ECE using failure_label_top20 and provided fp.
        y_fail = np.array([labels[c["instance_id"]]["failure_label_top20"] for c in cards])
        if failure_prob is None:
            failure_prob = 1.0 - conf
        fail_metrics = {
            "auroc_fail": auroc(y_fail, failure_prob),
            "auprc_fail": auprc(y_fail, failure_prob),
            "brier_fail": brier_score(np.asarray(failure_prob), y_fail),
            "ece_fail": expected_calibration_error(np.asarray(failure_prob), y_fail,
                                                    n_bins=int(cfg["evaluation"]["reliability_bins"])),
        }
        cal_pairs[name] = (list(np.asarray(failure_prob)), list(np.asarray(y_fail, dtype=float)))
        row = {
            "method": name,
            "mae_mean": ci["mean"], "mae_lo": ci["lo"], "mae_hi": ci["hi"],
            **arb,
            **{f"sel_risk@{int(k*100)}": v for k, v in sel_at_cov.items()},
            **fail_metrics,
        }
        if cost:
            row.update({f"cost.{k}": v for k, v in cost.items()})
        rows.append(row)
        for i, card in enumerate(cards):
            per_inst.append({
                "method": name,
                "instance_id": card["instance_id"],
                "mae": float(errs[i]),
                "selected_id": selected_ids[i],
                "confidence": float(conf[i]),
                "failure_prob": float(failure_prob[i]),
            })

    # Forecast-only baselines
    for name in forecast_only:
        fcasts = []
        sel = []
        conf = []
        for c in cards:
            r = _baseline_forecast(c, name)
            fcasts.append(np.asarray(r["forecast"], dtype=np.float64))
            sel.append(r["selected_model_id"])
            conf.append(r["confidence"])
        add_method(name, np.stack(fcasts, axis=0), sel, np.array(conf, dtype=np.float64))

    # Oracle (upper bound).
    oracle_f = np.zeros_like(test_targets)
    oracle_ids: list[str] = []
    for i, c in enumerate(cards):
        oid = labels[c["instance_id"]]["oracle_best_model_id"]
        for m in c["candidate_models"]:
            if m["model_id"] == oid:
                oracle_f[i] = np.asarray(m["forecast"], dtype=np.float64)
                break
        oracle_ids.append(oid)
    add_method("oracle_best_upperbound", oracle_f, oracle_ids, np.ones(len(cards)))

    # Non-LLM router systems: route between val-weighted-ensemble and a fallback.
    for r_name, fp in router_preds.items():
        fcasts = []
        sel = []
        conf = []
        for i, c in enumerate(cards):
            if fp[i] < 0.5:
                r = _baseline_forecast(c, "validation_weighted_ensemble")
            else:
                # High failure-prob: pick val-best (more conservative selection).
                r = _baseline_forecast(c, "validation_best")
            fcasts.append(np.asarray(r["forecast"], dtype=np.float64))
            sel.append(r["selected_model_id"])
            conf.append(1.0 - float(fp[i]))
        add_method(f"{r_name}_router", np.stack(fcasts, axis=0), sel,
                    np.array(conf, dtype=np.float64), failure_prob=fp)

    # LLM systems from decisions.jsonl (if present).
    agents_path = rd / "agents" / "decisions.jsonl"
    cost_summary_path = rd / "agents" / "cost_summary.json"
    sys_cost = {}
    if cost_summary_path.exists():
        sys_cost = json.loads(cost_summary_path.read_text(encoding="utf-8"))
    if agents_path.exists():
        by_system: dict[str, list[dict[str, Any]]] = {}
        for d in read_jsonl(agents_path):
            by_system.setdefault(d["decision_system"], []).append(d)
        card_by_id = {c["instance_id"]: c for c in cards}
        for sys_name, items in by_system.items():
            # Align to card order.
            items_by_id = {d["instance_id"]: d for d in items}
            fcasts = []
            sel = []
            conf = []
            fp = []
            for c in cards:
                d = items_by_id[c["instance_id"]]["decision"]
                f, label = _decision_to_forecast(d, c)
                fcasts.append(f)
                sel.append(label)
                conf.append(float(d.get("confidence", 0.5)))
                fp.append(float(d.get("failure_probability", 0.5)))
            cost = sys_cost.get(sys_name) or {}
            add_method(sys_name, np.stack(fcasts, axis=0), sel,
                        np.array(conf, dtype=np.float64), failure_prob=np.array(fp, dtype=np.float64), cost=cost)

    # Persist tables, curves, calibration pairs.
    out_dir = rd / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "metrics_summary.csv", rows)
    cols = ["method", "mae_mean", "mae_lo", "mae_hi",
            "regret_mean", "relative_regret_mean", "selection_accuracy",
            "auroc_fail", "auprc_fail", "brier_fail", "ece_fail"]
    write_latex(out_dir / "metrics_summary.tex", rows, cols=cols, caption="Main metrics", label="tab:main")
    write_jsonl(out_dir / "metrics_per_instance.jsonl", per_inst)
    (out_dir / "risk_coverage_curves.json").write_text(json.dumps(rc_curves, indent=2), encoding="utf-8")
    (out_dir / "calibration_pairs.json").write_text(json.dumps(cal_pairs), encoding="utf-8")
    log.info("Wrote evaluation tables to %s", out_dir)
    return {"n_methods": len(rows), "out_dir": str(out_dir)}
