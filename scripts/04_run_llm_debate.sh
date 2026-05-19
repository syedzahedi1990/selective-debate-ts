#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/experiments/mvp.yaml}
PROVIDER=${PROVIDER:-mock}
sdfts run-agents --config "$CFG" --provider "$PROVIDER"
