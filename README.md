# Selective Debate for Reliable Time-Series Forecasting

**SAD-TS**: Selective Auditing Debate for Time-Series Forecasting.

This repository implements the experimental pipeline for the paper
*"Selective Debate for Reliable Time-Series Forecasting: When Should Forecasting Agents Deliberate?"*

The central question is **not** whether LLM debate improves average forecast error.
The question is **when** deliberation is worth running, and whether it improves *reliability*
under cost constraints. We treat multi-agent debate as a **selective, tool-verified audit layer**
over a panel of trained and pretrained forecasters — not as a forecaster itself.

## Quick start (smoke test, no LLM API needed)

```bash
# 1. (one-time) Install
pip install -e .

# 2. End-to-end smoke test on a tiny synthetic dataset
bash scripts/00_smoke_test.sh
```

The smoke test:
- Generates a tiny synthetic dataset (sinusoid + trend + noise, 4 series).
- Trains a reduced forecast panel.
- Builds forecast cards + private labels.
- Runs non-LLM baselines, a router, and a *mock* LLM auditor / vote / debate.
- Writes metrics tables and plots to `outputs/<run_name>/`.
- Asserts no label-leakage and JSON schema validity.

## Pipeline (Phase 1 — non-LLM MVP)

```
sdfts train-panel    --config configs/experiments/mvp.yaml
sdfts predict-panel  --config configs/experiments/mvp.yaml
sdfts build-cards    --config configs/experiments/mvp.yaml
sdfts train-router   --config configs/experiments/mvp.yaml
sdfts run-baselines  --config configs/experiments/mvp.yaml
sdfts run-agents     --config configs/experiments/mvp.yaml --provider mock
sdfts evaluate       --config configs/experiments/mvp.yaml
sdfts make-figures   --config configs/experiments/mvp.yaml
```

## Repo layout

```
selective-debate-ts/
  configs/             # YAML configs
  prompts/             # Versioned LLM prompts (text)
  scripts/             # Shell entry points (smoke, train, eval, ...)
  src/sdfts/
    data/              # loaders, windowing, scaling, splits
    models/            # 4 architectures (LSTM/GRU/Transformer/TCN) x 3 regimes
    diagnostics/       # disagreement tensor, changepoint, OOD, residuals
    cards/             # forecast cards + private labels (JSONL)
    routers/           # non-LLM arbitration (LR / RF / GB / threshold)
    agents/            # LLM providers (Mock/OpenAI/Anthropic), debate, verifier, judge
    evaluation/        # metrics, calibration, selective, stat tests, tables, plots
    experiments/       # CLI sub-commands
  tests/               # leakage + schema + metric tests
  outputs/             # per-run artefacts (gitignored)
```

## Datasets

The default config uses a deterministic synthetic dataset so the smoke test
runs anywhere (no network, no Kaggle/UCI download). Real loaders for ETT-style,
GIFT-Eval, and fev-bench are stubbed and noted in `src/sdfts/data/loaders.py`.

## Leakage rules

- LLM agents only ever see forecast cards (validation metrics + diagnostics).
- Ground truth and candidate **test** errors live in a separate private-labels JSONL,
  read only by evaluation code.
- All evidence IDs cited by agents are validated against the forecast card.

## License

MIT.
