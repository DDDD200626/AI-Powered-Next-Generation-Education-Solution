"""PyTorch-based contribution model (ensemble + uncertainty)."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edu_tools.team_ml_model import (
    DATASET_PATH,
    MIN_TRAIN_SAMPLES,
    RETRAIN_INTERVAL,
    feature_row,
    load_dataset,
)

DATA_DIR = Path(__file__).with_name("data")
TORCH_MODEL_PATH = DATA_DIR / "team_torch_model.pt"
TORCH_META_PATH = DATA_DIR / "team_torch_model.meta.json"
ENSEMBLE_SIZE = 3


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _save_meta(meta: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_meta() -> dict[str, Any] | None:
    if not TORCH_META_PATH.exists():
        return None
    try:
        raw = json.loads(TORCH_META_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def train_torch_if_needed() -> dict[str, Any]:
    if not torch_available():
        return {"enabled": False, "reason": "torch_unavailable"}

    import torch
    import torch.nn as nn

    data = load_dataset()
    n = len(data)
    meta = _load_meta() or {}
    trained_count = int(meta.get("sample_count", 0) or 0)
    if n < MIN_TRAIN_SAMPLES:
        return {"enabled": True, "trained": False, "sample_count": n, "reason": "insufficient_samples"}
    if TORCH_MODEL_PATH.exists() and (n - trained_count) < RETRAIN_INTERVAL:
        return {
            "enabled": True,
            "trained": False,
            "sample_count": n,
            "model_version": meta.get("model_version"),
            "trained_at": meta.get("trained_at"),
            "reason": "interval_not_reached",
        }

    x_all = torch.tensor([[float(v) for v in r["x"]] for r in data], dtype=torch.float32)
    y_all = torch.tensor([[max(0.0, min(100.0, float(r["y"])))] for r in data], dtype=torch.float32)
    means = x_all.mean(dim=0, keepdim=True)
    stds = x_all.std(dim=0, keepdim=True)
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    xz_all = (x_all - means) / stds

    # simple holdout metric
    idx = list(range(n))
    random.Random(42).shuffle(idx)
    split = max(1, int(n * 0.85))
    tr_idx = idx[:split]
    va_idx = idx[split:] if split < n else idx[-1:]

    ensemble_states: list[dict[str, Any]] = []
    val_maes: list[float] = []
    for m_i in range(ENSEMBLE_SIZE):
        rng = random.Random(1000 + m_i)
        bs_idx = [tr_idx[rng.randrange(len(tr_idx))] for _ in range(len(tr_idx))]
        x_tr = xz_all[bs_idx]
        y_tr = y_all[bs_idx]
        x_va = xz_all[va_idx]
        y_va = y_all[va_idx]

        model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(p=0.08),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        model.train()
        for _ in range(320):
            opt.zero_grad()
            pred = model(x_tr)
            loss = loss_fn(pred, y_tr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p_va = model(x_va)
            mae = torch.mean(torch.abs(p_va - y_va)).item()
        val_maes.append(float(mae))
        ensemble_states.append(model.state_dict())

    state = {
        "ensemble_state_dicts": ensemble_states,
        "means": means.squeeze(0).tolist(),
        "stds": stds.squeeze(0).tolist(),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(state, TORCH_MODEL_PATH)
    now = datetime.now(timezone.utc)
    out_meta = {
        "model_version": f"team-torch-{now.strftime('%Y%m%d%H%M%S')}",
        "trained_at": now.isoformat(),
        "sample_count": n,
        "dataset_path": str(DATASET_PATH),
        "ensemble_size": ENSEMBLE_SIZE,
        "validation_mae_mean": round(sum(val_maes) / max(len(val_maes), 1), 4),
        "validation_mae_each": [round(v, 4) for v in val_maes],
    }
    _save_meta(out_meta)
    return {"enabled": True, "trained": True, **out_meta}


def predict_torch_score(commits: int, prs: int, lines: int, attendance: float, self_report: str) -> dict[str, float] | None:
    if not torch_available() or not TORCH_MODEL_PATH.exists():
        return None
    import torch
    import torch.nn as nn

    state = torch.load(TORCH_MODEL_PATH, map_location="cpu")
    sds = state.get("ensemble_state_dicts")
    if not isinstance(sds, list) or not sds:
        return None
    x = torch.tensor([feature_row(commits, prs, lines, attendance, self_report)], dtype=torch.float32)
    means = torch.tensor([state["means"]], dtype=torch.float32)
    stds = torch.tensor([state["stds"]], dtype=torch.float32)
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    xz = (x - means) / stds
    preds: list[float] = []
    for sd in sds:
        model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(p=0.08),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            preds.append(float(model(xz).squeeze().item()))
    if not preds:
        return None
    mean = sum(preds) / len(preds)
    var = sum((p - mean) ** 2 for p in preds) / max(1, len(preds))
    std = var**0.5
    return {
        "score": max(0.0, min(100.0, mean)),
        "uncertainty": max(0.0, min(100.0, std)),
    }


def torch_model_meta() -> dict[str, Any]:
    return _load_meta() or {}

