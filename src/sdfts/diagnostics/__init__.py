"""Forecast-card diagnostics: input-window features, disagreement tensor, OOD."""
from sdfts.diagnostics.features import (
    trend_strength,
    seasonality_strength,
    volatility,
    missingness_rate,
    recent_level_shift,
    residual_autocorrelation,
)
from sdfts.diagnostics.disagreement import disagreement_features
from sdfts.diagnostics.changepoint import changepoint_score
from sdfts.diagnostics.ood import ood_score
from sdfts.diagnostics.residuals import recent_residual_summary

__all__ = [
    "trend_strength",
    "seasonality_strength",
    "volatility",
    "missingness_rate",
    "recent_level_shift",
    "residual_autocorrelation",
    "disagreement_features",
    "changepoint_score",
    "ood_score",
    "recent_residual_summary",
]
