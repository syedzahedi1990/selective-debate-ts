"""Publication-style plots (matplotlib only)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdfts.evaluation.calibration import reliability_curve
from sdfts.evaluation.selective import risk_coverage_curve


def plot_risk_cost_pareto(
    methods: dict[str, dict[str, float]],
    out_path: str | Path,
) -> None:
    """``methods`` maps method-name -> dict with keys 'cost' and 'risk'."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, m in methods.items():
        ax.scatter(m["cost"], m["risk"], label=name)
    ax.set_xlabel("Cost (LLM calls or tokens)")
    ax.set_ylabel("Selective risk / Failure rate")
    ax.set_title("Risk-cost Pareto curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_risk_coverage(
    method_to_curve: dict[str, dict[str, np.ndarray]],
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, curve in method_to_curve.items():
        ax.plot(curve["coverage"], curve["risk"], label=name)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("Risk-coverage curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration(
    method_to_pyy: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    n_bins: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
    for name, (p, y) in method_to_pyy.items():
        if p.size == 0:
            continue
        rc = reliability_curve(p, y, n_bins=n_bins)
        ax.plot(rc["mean_predicted"], rc["mean_observed"], "o-", label=name)
    ax.set_xlabel("Predicted failure probability")
    ax.set_ylabel("Observed failure rate")
    ax.set_title("Calibration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_when_debate_helps(
    regime_to_pair: dict[str, tuple[float, float]],
    out_path: str | Path,
    baseline_label: str = "default",
    method_label: str = "selective_debate",
) -> None:
    """``regime_to_pair`` maps regime-name -> (baseline_metric, method_metric).
    Smaller is better for risk-style metrics.
    """
    names = list(regime_to_pair.keys())
    base = [regime_to_pair[n][0] for n in names]
    meth = [regime_to_pair[n][1] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.6), 4))
    ax.bar(x - 0.2, base, width=0.4, label=baseline_label)
    ax.bar(x + 0.2, meth, width=0.4, label=method_label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("When does debate help?")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_disagreement_tensor(
    forecasts: np.ndarray,
    archs: list[str],
    regimes: list[str],
    out_path: str | Path,
) -> None:
    """Average per-arch x per-regime forecast std across horizon."""
    H = forecasts.shape[1]
    uniq_archs = sorted(set(archs))
    uniq_regs = sorted(set(regimes))
    mat = np.full((len(uniq_archs), len(uniq_regs)), np.nan)
    for i, a in enumerate(uniq_archs):
        for j, r in enumerate(uniq_regs):
            idx = [k for k, (aa, rr) in enumerate(zip(archs, regimes)) if aa == a and rr == r]
            if idx:
                mat[i, j] = float(np.std(forecasts[idx], axis=0).mean()) if len(idx) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(max(4, len(uniq_regs) * 0.8 + 2), max(3, len(uniq_archs) * 0.6 + 2)))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(uniq_regs)))
    ax.set_xticklabels(uniq_regs, rotation=30, ha="right")
    ax.set_yticks(range(len(uniq_archs)))
    ax.set_yticklabels(uniq_archs)
    ax.set_title("Forecast disagreement (mean horizon std)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation_bars(
    rows: list[dict[str, Any]],
    metric: str,
    out_path: str | Path,
) -> None:
    names = [r.get("name") or r.get("method") or "?" for r in rows]
    vals = [float(r[metric]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.8), 4))
    ax.bar(names, vals)
    ax.set_ylabel(metric)
    ax.set_title("Ablation")
    plt.xticks(rotation=30, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
