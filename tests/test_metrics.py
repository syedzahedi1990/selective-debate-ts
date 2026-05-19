"""Metric arithmetic sanity checks."""
from __future__ import annotations

import numpy as np

from sdfts.evaluation.metrics import (
    arbitration_metrics,
    compute_forecast_metrics,
    cost_metrics,
)
from sdfts.evaluation.selective import (
    auprc,
    auroc,
    risk_coverage_curve,
    selective_risk_at_coverage,
)
from sdfts.evaluation.calibration import brier_score, expected_calibration_error


def test_compute_forecast_metrics_perfect():
    yt = np.zeros((10, 4))
    yp = np.zeros((10, 4))
    m = compute_forecast_metrics(yt, yp)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0


def test_compute_forecast_metrics_simple():
    yt = np.array([[0, 1.0, 2.0, 3.0]])
    yp = np.array([[1.0, 2.0, 3.0, 4.0]])
    m = compute_forecast_metrics(yt, yp)
    assert abs(m["mae"] - 1.0) < 1e-9


def test_arbitration_metrics_basic():
    sel = np.array([1.0, 2.0, 3.0])
    orc = np.array([0.5, 1.0, 1.5])
    dflt = np.array([2.0, 4.0, 6.0])
    m = arbitration_metrics(sel, orc, dflt, selected_ids=["a", "b", "c"], oracle_ids=["a", "b", "x"])
    assert m["regret_mean"] > 0
    assert 0 <= m["selection_accuracy"] <= 1
    assert m["improvement_over_default_pct"] > 0


def test_auroc_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(auroc(y, s) - 1.0) < 1e-9


def test_auprc_basic():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    a = auprc(y, s)
    assert 0.5 <= a <= 1.0


def test_risk_coverage_monotone():
    errs = np.array([0.1, 0.2, 0.3, 0.4])
    conf = np.array([0.4, 0.3, 0.2, 0.1])  # highest conf -> lowest err
    rc = risk_coverage_curve(errs, conf)
    assert rc["risk"][0] <= rc["risk"][-1]
    sel = selective_risk_at_coverage(errs, conf, [0.25, 0.5, 1.0])
    assert sel[0.25] <= sel[1.0]


def test_brier_and_ece():
    p = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.5, 1.0])
    assert brier_score(p, y) == 0.0
    assert expected_calibration_error(p, y) == 0.0


def test_cost_metrics_zero():
    c = cost_metrics(0, 0, 0)
    assert c["n_calls"] == 0 and c["cost_usd"] == 0.0
