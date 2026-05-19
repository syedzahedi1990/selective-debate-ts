"""Build forecast cards + private labels from panel outputs."""
from __future__ import annotations

from typing import Any

from sdfts.cards.build_cards import build_cards
from sdfts.cards.io import write_cards, write_labels
from sdfts.cards.schemas import validate_card, validate_label
from sdfts.config import run_dir
from sdfts.experiments.run_forecast_panel import build_or_load_windows, load_panel_outputs
from sdfts.utils.logging import get_logger


log = get_logger(__name__)


def build(cfg: dict[str, Any]) -> dict[str, Any]:
    ws = build_or_load_windows(cfg)
    outs = load_panel_outputs(cfg)
    meta = outs["meta"]["models"]
    cand_meta = {mid: {"architecture": m["architecture"], "training_regime": m["training_regime"]}
                 for mid, m in meta.items()}
    cards, labels = build_cards(
        cfg=cfg,
        ws=ws,
        test_forecasts=outs["test_forecasts"],
        val_forecasts=outs["val_forecasts"],
        cand_meta=cand_meta,
    )

    # Validate before writing.
    bad = []
    for c in cards:
        errs = validate_card(c)
        if errs:
            bad.append((c["instance_id"], errs[:1]))
    if bad:
        raise ValueError(f"Card schema errors in {len(bad)} cards. First: {bad[0]}")
    bad_l = []
    for l in labels:
        errs = validate_label(l)
        if errs:
            bad_l.append((l["instance_id"], errs[:1]))
    if bad_l:
        raise ValueError(f"Label schema errors in {len(bad_l)} labels. First: {bad_l[0]}")

    rd = run_dir(cfg)
    n_cards = write_cards(rd, cards)
    n_labels = write_labels(rd, labels)
    log.info("Wrote %d cards, %d labels", n_cards, n_labels)
    return {"n_cards": n_cards, "n_labels": n_labels}
