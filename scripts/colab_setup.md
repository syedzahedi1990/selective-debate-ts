# Colab Pro setup

Run these cells in order on Colab Pro. The smoke test should take roughly
1–2 minutes on a T4/A100 with the MVP config.

## 1. Clone + install

```bash
%cd /content
!git clone https://github.com/<your-fork>/selective-debate-ts.git || true
%cd selective-debate-ts
!pip -q install -e .
```

## 2. Smoke test (no LLM API)

```bash
!bash scripts/00_smoke_test.sh
```

## 3. Inspect outputs

```python
from pathlib import Path
import pandas as pd

run = Path("outputs/mvp_smoke")
print("Cards:", sum(1 for _ in open(run / "cards/forecast_cards.jsonl")))
print("Labels:", sum(1 for _ in open(run / "cards/private_labels.jsonl")))

df = pd.read_csv(run / "evaluation/metrics_summary.csv")
df.sort_values("mae_mean").head(20)
```

## 4. Switch to real LLM (after MVP passes)

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```

```bash
!sdfts run-agents --config configs/experiments/mvp.yaml --provider openai
!sdfts evaluate --config configs/experiments/mvp.yaml
!sdfts make-figures --config configs/experiments/mvp.yaml
```

## 5. Persist outputs to Drive

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/sdfts-cache
!rsync -a outputs/ /content/drive/MyDrive/sdfts-cache/
```
