"""``sdfts`` CLI.

Sub-commands wire experiment runners. The smoke test calls them via
``scripts/00_smoke_test.sh``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sdfts.config import load_config, run_dir
from sdfts.experiments.make_figures import make_figures
from sdfts.experiments.run_agents import run_all as run_agents
from sdfts.experiments.run_cards import build as run_build_cards
from sdfts.experiments.run_evaluation import evaluate as run_evaluate
from sdfts.experiments.run_forecast_panel import predict_panel, train_panel
from sdfts.experiments.run_routers import train_routers
from sdfts.utils.logging import get_logger


log = get_logger(__name__)


def _add_cfg_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=str)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser("sdfts")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["train-panel", "predict-panel", "build-cards",
                 "train-router", "run-baselines", "run-agents",
                 "evaluate", "make-figures", "run-all"]:
        p = sub.add_parser(name)
        _add_cfg_arg(p)
        if name == "run-agents":
            p.add_argument("--provider", default=None,
                           help="override agents.provider for this run (mock|openai|anthropic)")
            p.add_argument("--model", default=None,
                           help="override agents.model for this run (e.g. gpt-4o-mini)")
            p.add_argument("--systems", default=None,
                           help="comma-separated list of LLM decision systems to run")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    log.info("Loaded config %s; run_name=%s", args.config, cfg["run_name"])
    run_dir(cfg)  # ensure dir exists

    if args.cmd == "train-panel":
        out = train_panel(cfg)
    elif args.cmd == "predict-panel":
        out = predict_panel(cfg)
    elif args.cmd == "build-cards":
        out = run_build_cards(cfg)
    elif args.cmd == "train-router":
        out = train_routers(cfg)
    elif args.cmd == "run-baselines":
        # Baselines are evaluated inline by `evaluate`; this is a noop kept for the CLI spec.
        out = {"note": "Forecast-only and router baselines are evaluated by the `evaluate` step."}
    elif args.cmd == "run-agents":
        systems = [s.strip() for s in args.systems.split(",")] if args.systems else None
        out = run_agents(cfg, systems=systems,
                         provider_override=args.provider,
                         model_override=args.model)
    elif args.cmd == "evaluate":
        out = run_evaluate(cfg)
    elif args.cmd == "make-figures":
        out = make_figures(cfg)
    elif args.cmd == "run-all":
        out = _run_all(cfg)
    else:
        parser.error(f"unknown command: {args.cmd}")

    print(json.dumps(out, default=str, indent=2))
    return 0


def _run_all(cfg) -> dict:
    """Convenience: full pipeline end-to-end. Used by the smoke test."""
    out = {}
    out["train_panel"] = train_panel(cfg)
    out["predict_panel"] = predict_panel(cfg)
    out["build_cards"] = run_build_cards(cfg)
    out["train_routers"] = train_routers(cfg)
    out["run_agents"] = run_agents(cfg)
    out["evaluate"] = run_evaluate(cfg)
    out["make_figures"] = make_figures(cfg)
    return out


if __name__ == "__main__":
    sys.exit(main())
