#!/usr/bin/env bash
# Fetch the four ETT-small CSVs from the official ETDataset repo.
# Idempotent: skips files that already exist.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=${DEST:-data_cache/ETT}
BASE=https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small

mkdir -p "$DEST"
for f in ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv; do
    if [[ -f "$DEST/$f" ]]; then
        echo "[skip] $DEST/$f already exists"
    else
        echo "[get ] $DEST/$f"
        curl -fSL "$BASE/$f" -o "$DEST/$f"
    fi
done

echo "Done. Files in $DEST:"
ls -la "$DEST"
