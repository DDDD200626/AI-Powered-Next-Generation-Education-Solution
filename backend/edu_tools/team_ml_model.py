"""Lightweight data-driven contribution model (no external ML deps)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).with_name("data")
DATASET_PATH = DATA_DIR / "team_ml_dataset.jsonl"
MODEL_PATH = DATA_DIR / "team_ml_model.json"

MIN_TRAIN_SAMPLES = 40
RETRAIN_INTERVAL = 20


@dataclass
class TeamMlModel:
    version: str
    trained_at: str
    sample_count: int
    means: list[float]
    stds: list[float]
    weights: list[float]
    bias: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wc(text: str) -> int:
    return len([x for x in text.split() if x.strip()])


def feature_row(commits: int, prs: int, lines: int, attendance: float, self_report: str) -> list[float]:
    return [
        math.log1p(max(0, commits)),
        math.log1p(max(0, prs)),
        math.log1p(max(0, lines)),
        max(0.0, min(100.0, float(attendance))) / 100.0,
        math.log1p(max(0, _wc(self_report))),
    ]


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def append_samples(rows: list[dict]) -> int:
    _ensure_data_dir()
    with DATASET_PATH.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        return []
    out: list[dict] = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
            if isinstance(x, dict) and "x" in x and "y" in x:
                out.append(x)
        except Exception:
            continue
    return out


def load_model() -> TeamMlModel | None:
    if not MODEL_PATH.exists():
        return None
    try:
        raw = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        return TeamMlModel(
            version=str(raw["version"]),
            trained_at=str(raw["trained_at"]),
            sample_count=int(raw["sample_count"]),
            means=[float(v) for v in raw["means"]],
            stds=[float(v) for v in raw["stds"]],
            weights=[float(v) for v in raw["weights"]],
            bias=float(raw["bias"]),
        )
    except Exception:
        return None


def _save_model(m: TeamMlModel) -> None:
    _ensure_data_dir()
    MODEL_PATH.write_text(
        json.dumps(
            {
                "version": m.version,
                "trained_at": m.trained_at,
                "sample_count": m.sample_count,
                "means": m.means,
                "stds": m.stds,
                "weights": m.weights,
                "bias": m.bias,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _z(x: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(x[i] - means[i]) / max(stds[i], 1e-6) for i in range(len(x))]


def train_if_needed() -> dict:
    data = load_dataset()
    n = len(data)
    model = load_model()
    trained_count = model.sample_count if model else 0

    if n < MIN_TRAIN_SAMPLES:
        return {"enabled": model is not None, "trained": False, "sample_count": n, "reason": "insufficient_samples"}
    if model and (n - trained_count) < RETRAIN_INTERVAL:
        return {
            "enabled": True,
            "trained": False,
            "sample_count": n,
            "model_version": model.version,
            "trained_at": model.trained_at,
            "reason": "interval_not_reached",
        }

    xs = [[float(v) for v in r["x"]] for r in data]
    ys = [max(0.0, min(100.0, float(r["y"]))) for r in data]
    d = len(xs[0])
    means = [sum(row[i] for row in xs) / n for i in range(d)]
    stds = []
    for i in range(d):
        var = sum((row[i] - means[i]) ** 2 for row in xs) / n
        stds.append(math.sqrt(var) if var > 1e-10 else 1.0)
    xz = [_z(row, means, stds) for row in xs]

    w = [0.0] * d
    b = 50.0
    lr = 0.01
    epochs = 220
    for _ in range(epochs):
        for i in range(n):
            pred = b + sum(w[j] * xz[i][j] for j in range(d))
            err = pred - ys[i]
            b -= lr * err
            for j in range(d):
                w[j] -= lr * err * xz[i][j]

    m = TeamMlModel(
        version=f"team-ml-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        trained_at=_now_iso(),
        sample_count=n,
        means=means,
        stds=stds,
        weights=w,
        bias=b,
    )
    _save_model(m)
    return {
        "enabled": True,
        "trained": True,
        "sample_count": n,
        "model_version": m.version,
        "trained_at": m.trained_at,
    }


def predict_score(model: TeamMlModel, x: list[float]) -> float:
    zx = _z(x, model.means, model.stds)
    y = model.bias + sum(model.weights[i] * zx[i] for i in range(len(zx)))
    return max(0.0, min(100.0, y))

