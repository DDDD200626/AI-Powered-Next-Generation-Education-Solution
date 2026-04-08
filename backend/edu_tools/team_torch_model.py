"""PyTorch-based contribution model (preferred when torch is available)."""

from __future__ import annotations

import json
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

    x = torch.tensor([[float(v) for v in r["x"]] for r in data], dtype=torch.float32)
    y = torch.tensor([[max(0.0, min(100.0, float(r["y"])))] for r in data], dtype=torch.float32)
    means = x.mean(dim=0, keepdim=True)
    stds = x.std(dim=0, keepdim=True)
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    xz = (x - means) / stds

    model = nn.Sequential(
        nn.Linear(5, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(260):
        opt.zero_grad()
        pred = model(xz)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    state = {
        "model_state_dict": model.state_dict(),
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
    }
    _save_meta(out_meta)
    return {"enabled": True, "trained": True, **out_meta}


def predict_torch_score(commits: int, prs: int, lines: int, attendance: float, self_report: str) -> float | None:
    if not torch_available() or not TORCH_MODEL_PATH.exists():
        return None
    import torch
    import torch.nn as nn

    state = torch.load(TORCH_MODEL_PATH, map_location="cpu")
    model = nn.Sequential(
        nn.Linear(5, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    x = torch.tensor([feature_row(commits, prs, lines, attendance, self_report)], dtype=torch.float32)
    means = torch.tensor([state["means"]], dtype=torch.float32)
    stds = torch.tensor([state["stds"]], dtype=torch.float32)
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)
    xz = (x - means) / stds
    with torch.no_grad():
        y = float(model(xz).squeeze().item())
    return max(0.0, min(100.0, y))


def torch_model_meta() -> dict[str, Any]:
    return _load_meta() or {}

