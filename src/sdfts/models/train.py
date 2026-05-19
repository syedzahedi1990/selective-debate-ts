"""Training loops for the forecast panel.

For ``one_step_recursive`` we train next-step MSE only; rollout to ``H`` is done
at prediction time. For ``h_step_direct`` and ``direct_multi_step`` we train the
full horizon vector directly.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sdfts.data.windowing import WindowSet, stack_xy
from sdfts.models.regimes import Candidate
from sdfts.utils.logging import get_logger


log = get_logger(__name__)


def _pick_device(cfg: dict[str, Any]) -> torch.device:
    pref = cfg["model_panel"].get("device", "auto")
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_candidate(cand: Candidate, windows: WindowSet, cfg: dict[str, Any]) -> dict[str, Any]:
    device = _pick_device(cfg)
    mp = cfg["model_panel"]
    cand.module.to(device)
    opt = torch.optim.Adam(cand.module.parameters(), lr=float(mp["lr"]), weight_decay=float(mp["weight_decay"]))
    loss_fn = nn.MSELoss()

    X_tr, Y_tr = stack_xy(windows.train)
    X_val, Y_val = stack_xy(windows.val)
    if X_tr.size == 0:
        raise ValueError("No training windows.")

    if cand.training_regime == "one_step_recursive":
        # Target is the *first* horizon step only.
        Y_tr_use = Y_tr[:, :1]
        Y_val_use = Y_val[:, :1]
    else:
        Y_tr_use = Y_tr
        Y_val_use = Y_val

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr_use))
    loader = DataLoader(ds, batch_size=int(mp["batch_size"]), shuffle=True, drop_last=False)

    best_val = float("inf")
    history = []
    for epoch in range(int(mp["train_epochs"])):
        cand.module.train()
        running = 0.0
        nseen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            yhat = cand.module(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)
            nseen += xb.size(0)
        train_mse = running / max(1, nseen)

        cand.module.eval()
        with torch.no_grad():
            yhat_val = cand.module(torch.from_numpy(X_val).to(device)).cpu().numpy()
        val_mse = float(np.mean((yhat_val - Y_val_use) ** 2))
        if val_mse < best_val:
            best_val = val_mse
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        log.info("%s epoch=%d train_mse=%.4f val_mse=%.4f", cand.model_id, epoch, train_mse, val_mse)

    return {"history": history, "best_val_mse": best_val}
