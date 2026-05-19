#!/usr/bin/env bash
# End-to-end smoke test on a tiny synthetic dataset. No LLM API needed.
# Acceptance criteria covered:
#   1. Runs end-to-end on a tiny dataset.
#   2. Trains all 4 architectures under all 3 regimes (or reduced panel).
#   3. Builds forecast cards + private labels.
#   4. Evaluates val-best, mean, weighted ensembles + non-LLM router
#      + mock single auditor, vote, selective debate.
#   5. Writes metrics table + plots.
#   6. Asserts no label leakage (tests/test_no_label_leakage.py).
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/experiments/mvp.yaml}

echo "==> 1. Train panel"
sdfts train-panel --config "$CFG"

echo "==> 2. Predict panel"
sdfts predict-panel --config "$CFG"

echo "==> 3. Build forecast cards + private labels"
sdfts build-cards --config "$CFG"

echo "==> 4. Train non-LLM routers"
sdfts train-router --config "$CFG"

echo "==> 5. Run mock LLM agents (single_auditor, vote, debate, ...)"
sdfts run-agents --config "$CFG" --provider mock

echo "==> 6. Evaluate every decision system"
sdfts evaluate --config "$CFG"

echo "==> 7. Make figures"
sdfts make-figures --config "$CFG"

echo "==> 8. Run unit tests (leakage + schemas + metrics + router)"
pytest -q

echo "==> Smoke test passed."
