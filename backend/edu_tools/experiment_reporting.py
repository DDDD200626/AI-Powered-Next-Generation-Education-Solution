"""딥러닝 실험 재현·비교용 컨텍스트 수집(성능·일반화 추적)."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edu_tools.team_ml_model import DATASET_PATH, dataset_file_sha256
from edu_tools.team_torch_model import TORCH_META_PATH


def _git_head_short() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    except Exception:
        pass
    return None


def _count_jsonl_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        n = 0
        with path.open("rb") as f:
            for _ in f:
                n += 1
        return n
    except Exception:
        return None


def _env_snapshot(prefixes: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in sorted(os.environ.items()):
        if any(k.startswith(p) for p in prefixes):
            if "KEY" in k.upper() or "SECRET" in k.upper() or "PASSWORD" in k.upper():
                out[k] = "(redacted)"
            else:
                out[k] = v
    return out


def collect_dl_experiment_context() -> dict[str, Any]:
    """학습 산출물·데이터셋·환경 스냅샷(스크립트·노트북에서 재사용)."""
    meta: dict[str, Any] | None = None
    if TORCH_META_PATH.is_file():
        try:
            raw = json.loads(TORCH_META_PATH.read_text(encoding="utf-8"))
            meta = raw if isinstance(raw, dict) else None
        except Exception:
            meta = None

    dq = (meta or {}).get("dl_quality_unified") if isinstance(meta, dict) else None
    cv = (dq or {}).get("cv") if isinstance(dq, dict) else {}
    fit = (dq or {}).get("fit_metrics") if isinstance(dq, dict) else {}
    ho = (dq or {}).get("holdout_time") if isinstance(dq, dict) else {}

    return {
        "schema": "dl_experiment_context_v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_short": _git_head_short(),
        "torch_meta_path": str(TORCH_META_PATH),
        "torch_meta_present": meta is not None,
        "dataset_path": str(DATASET_PATH),
        "dataset_sha256": dataset_file_sha256(),
        "dataset_line_count": _count_jsonl_lines(DATASET_PATH),
        "metrics_summary": {
            "sample_count": meta.get("sample_count") if meta else None,
            "validation_mae_mean": meta.get("validation_mae_mean") if meta else None,
            "cv_mae_mean": meta.get("cv_mae_mean") if meta else None,
            "cv_vs_validation_gap": meta.get("cv_vs_validation_gap") if meta else None,
            "holdout_time_mae": meta.get("holdout_time_mae") if meta else None,
            "calibration_pearson_r": meta.get("calibration_pearson_r") if meta else None,
            "calibration_r2": meta.get("calibration_r2") if meta else None,
            "chronological_tail_mae_mean": meta.get("chronological_tail_mae_mean") if meta else None,
        },
        "dl_quality_unified_excerpt": {
            "cv": cv if isinstance(cv, dict) else {},
            "fit_metrics": {k: fit.get(k) for k in ("calibration_pearson_r", "calibration_r2") if isinstance(fit, dict)},
            "holdout_time": {
                "active": ho.get("active") if isinstance(ho, dict) else None,
                "mae": ho.get("mae") if isinstance(ho, dict) else None,
            }
            if isinstance(ho, dict)
            else {},
        },
        "env_team_torch_semantic_train": _env_snapshot(
            ("TEAM_TORCH_", "TEAM_SEMANTIC_", "TEAM_TRAIN_", "TEAM_RETRAIN_", "TEAM_HOLDOUT_", "STRUCTURAL_TARGET_WEIGHT")
        ),
    }
