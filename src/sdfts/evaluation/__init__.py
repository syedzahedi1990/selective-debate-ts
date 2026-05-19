"""Metrics, calibration, selective forecasting, statistical tests, and reports."""
from sdfts.evaluation.metrics import (
    compute_forecast_metrics,
    arbitration_metrics,
    cost_metrics,
)
from sdfts.evaluation.calibration import (
    expected_calibration_error,
    reliability_curve,
    brier_score,
)
from sdfts.evaluation.selective import (
    risk_coverage_curve,
    selective_risk_at_coverage,
    abstention_precision,
)
from sdfts.evaluation.statistical_tests import (
    bootstrap_ci,
    paired_bootstrap_diff,
)

__all__ = [
    "compute_forecast_metrics",
    "arbitration_metrics",
    "cost_metrics",
    "expected_calibration_error",
    "reliability_curve",
    "brier_score",
    "risk_coverage_curve",
    "selective_risk_at_coverage",
    "abstention_precision",
    "bootstrap_ci",
    "paired_bootstrap_diff",
]
