#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/experiments/mvp.yaml}
sdfts build-cards --config "$CFG"
