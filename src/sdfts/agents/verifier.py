"""Deterministic statistical verifier.

The verifier exposes a small set of inspection functions that the *judge* uses
to validate agent claims. None of these touch hidden ground truth. The
verifier also produces a compact report appended to the judge's prompt so
the judge can reason about whether claimed evidence actually supports a
recommendation.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class StatisticalVerifier:
    """Stateless verifier — operates on forecast cards only."""

    @staticmethod
    def verify_evidence_ids(claim_ids: list[str], allowed: list[str]) -> tuple[list[str], list[str]]:
        allowed_set = set(allowed)
        ok = [c for c in claim_ids if c in allowed_set]
        bad = [c for c in claim_ids if c not in allowed_set]
        return ok, bad

    @staticmethod
    def horizonwise_reliability(card: dict[str, Any]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for m in card["candidate_models"]:
            out[m["model_id"]] = list(m.get("horizonwise_validation_mae") or [])
        return out

    @staticmethod
    def disagreement_summary(card: dict[str, Any]) -> dict[str, Any]:
        d = card.get("diagnostics", {})
        return {
            "architecture_disagreement": d.get("architecture_disagreement", 0.0),
            "training_regime_disagreement": d.get("training_regime_disagreement", 0.0),
            "overall_forecast_dispersion": d.get("overall_forecast_dispersion", 0.0),
            "horizonwise_disagreement_mean": float(np.mean(d.get("horizonwise_disagreement") or [0.0])),
        }

    @staticmethod
    def changepoint_and_ood(card: dict[str, Any]) -> dict[str, Any]:
        d = card.get("diagnostics", {})
        return {
            "changepoint_score": d.get("changepoint_score", 0.0),
            "ood_score": d.get("ood_score", 0.0),
            "recent_level_shift_score": d.get("recent_level_shift_score", 0.0),
        }

    @classmethod
    def report(cls, card: dict[str, Any]) -> dict[str, Any]:
        return {
            "horizonwise_reliability": cls.horizonwise_reliability(card),
            "disagreement_summary": cls.disagreement_summary(card),
            "changepoint_and_ood": cls.changepoint_and_ood(card),
        }

    @staticmethod
    def assert_no_label_leakage(card: dict[str, Any]) -> None:
        """Belt-and-braces check: a card must not contain a 'ground_truth' field
        or candidate *test* errors. Raises ValueError if it does.
        """
        forbidden_top = {"ground_truth", "oracle_best_model_id", "default_error",
                         "candidate_errors", "failure_label_top20", "failure_label_threshold"}
        present = forbidden_top.intersection(card.keys())
        if present:
            raise ValueError(f"Card has forbidden label fields: {present}")
        for m in card.get("candidate_models", []):
            tm = m.get("test_metrics")
            if tm:
                raise ValueError(f"Candidate {m.get('model_id')} carries test_metrics — leakage.")
