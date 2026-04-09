"""DL 보조 모델·데이터 한계·건강도 통합 리포트(심사·운영용)."""

from __future__ import annotations

from typing import Any

from edu_tools.team_data_store import db_profile, model_monitor
from edu_tools.team_ml_model import FEATURE_DIM, FEATURE_VERSION
from edu_tools.team_torch_model import torch_available, torch_model_meta


def build_limits_report(window_days: int = 30) -> dict[str, Any]:
    """학습 가능 범위(한계 1)와 측정 가능 품질(한계 2)을 한 번에 요약."""
    d = max(1, min(365, int(window_days)))
    mon = model_monitor(window_days=d)
    prof = db_profile()
    meta: dict[str, Any] = dict(torch_model_meta()) if torch_available() else {}
    samples_n = int(meta.get("sample_count") or 0)
    cv_m = meta.get("cv_mae_mean")
    val_m = meta.get("validation_mae_mean")
    gap = meta.get("cv_vs_validation_gap")
    tail_m = meta.get("chronological_tail_mae_mean")

    ceiling_hint = "unknown"
    if gap is not None and isinstance(gap, (int, float)):
        g = float(gap)
        if g < 2.0:
            ceiling_hint = "cv_validation_consistent"
        elif g < 5.0:
            ceiling_hint = "moderate_generalization_gap"
        else:
            ceiling_hint = "high_gap_review_data_or_features"
    elif isinstance(cv_m, (int, float)) and isinstance(val_m, (int, float)):
        g = abs(float(cv_m) - float(val_m))
        if g < 2.0:
            ceiling_hint = "cv_validation_consistent"
        elif g < 5.0:
            ceiling_hint = "moderate_generalization_gap"
        else:
            ceiling_hint = "high_gap_review_data_or_features"

    return {
        "feature_schema": {
            "version": FEATURE_VERSION,
            "dim": FEATURE_DIM,
            "torch_meta_version": meta.get("feature_version"),
            "torch_input_dim": meta.get("input_dim"),
            "model_size_profile": meta.get("model_size_profile"),
            "approx_parameters": meta.get("approx_parameters"),
        },
        "dataset": {
            "jsonl_samples_reported": samples_n,
            "sqlite_rows": prof,
        },
        "training_diagnostics": {
            "cv_mae_mean": cv_m,
            "validation_mae_mean": val_m,
            "cv_vs_validation_gap": gap,
            "chronological_tail_mae_mean": tail_m,
            "chronological_tail_note": meta.get("chronological_tail_note"),
        },
        "runtime_window_days": d,
        "monitor": mon,
        "ceiling_hint": ceiling_hint,
        "limits_ko": {
            "L1": "입력·라벨에 없는 정보(코드 품질·리뷰 대화 등)는 피처/외부 데이터 없이 추정 불가.",
            "L2": "CV·검증 MAE가 더 이상 내려가지 않거나 꼬리 MAE가 악화되면 데이터/분포 포화 가능.",
        },
    }
