"""논문·연구 부록 등에서 이식한 **분포(평균·표준편차)**와 입력 피처의 z-정합도로 보조 점수.

- 웹 크롤링 없음. 로컬 JSON 1개(`TEAM_RESEARCH_DATA_FILE`)만 허용.
- 내부 `build_feature_vector` 앞쪽 5차원과 동일 스케일이어야 함(log1p 커밋 등).
- `dl_score`에 소폭 혼합(상한); 피처 차원은 바꾸지 않아 기존 NN 가중치와 호환.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).with_name("data")

# FEATURE_LABELS[0:5] 와 동일 순서
NORM_KEYS = (
    "log_commits",
    "log_prs",
    "log_lines",
    "attendance_norm",
    "log_self_report_words",
)


def _read_json_file(path_str: str) -> dict[str, Any] | None:
    p = Path(path_str)
    if not p.is_file():
        p2 = DATA_DIR / path_str
        if p2.is_file():
            p = p2
        else:
            return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def load_research_evidence() -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """JSON 프로파일과 메타. 비활성 시 (None, meta)."""
    path = (os.environ.get("TEAM_RESEARCH_DATA_FILE", "") or "").strip()
    meta: dict[str, Any] = {
        "active": False,
        "source": None,
        "reason": "no_file",
        "path": path or None,
    }
    if not path:
        return None, meta

    doc = _read_json_file(path)
    if not doc:
        meta["reason"] = "file_not_found_or_invalid"
        return None, meta

    integ = doc.get("integration") if isinstance(doc.get("integration"), dict) else {}
    enabled = bool(integ.get("enabled", False))
    env_on = (os.environ.get("TEAM_RESEARCH_BLEND_ENABLED", "") or "").strip().lower()
    if env_on in ("1", "true", "yes", "on"):
        enabled = True
    elif env_on in ("0", "false", "no", "off"):
        enabled = False

    if not enabled:
        meta.update(
            {
                "active": False,
                "reason": "disabled",
                "source": doc.get("source"),
            }
        )
        return None, meta

    norm = doc.get("normative_on_features")
    if not isinstance(norm, dict) or not norm:
        meta["reason"] = "no_normative_on_features"
        meta["source"] = doc.get("source")
        return None, meta

    try:
        blend_default = float(integ.get("blend_weight", 0.1))
    except (TypeError, ValueError):
        blend_default = 0.1
    blend_default = max(0.0, min(0.45, blend_default))
    bw_env = (os.environ.get("TEAM_RESEARCH_BLEND_WEIGHT", "") or "").strip()
    if bw_env:
        try:
            blend_default = max(0.0, min(0.45, float(bw_env)))
        except ValueError:
            pass

    try:
        z_trim = float(integ.get("z_trim", 2.5))
    except (TypeError, ValueError):
        z_trim = 2.5
    z_trim = max(0.5, min(5.0, z_trim))

    src = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    profile: dict[str, Any] = {
        "normative_on_features": norm,
        "integration": {
            **integ,
            "blend_weight": blend_default,
            "z_trim": z_trim,
        },
        "source": src,
    }
    meta.update(
        {
            "active": True,
            "reason": "ok",
            "blend_weight": blend_default,
            "z_trim": z_trim,
            "source": src,
            "note_ko": str(integ.get("note_ko") or "")[:2000] or None,
            "path": path,
        }
    )
    return profile, meta


def compute_research_z_alignment(
    x: list[float],
    profile: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """논문·연구 분포 대비 z-거리 기반 정합 점수(0–100)와 세부."""
    norm = profile.get("normative_on_features") or {}
    integ = profile.get("integration") if isinstance(profile.get("integration"), dict) else {}
    try:
        z_trim = float(integ.get("z_trim", 2.5))
    except (TypeError, ValueError):
        z_trim = 2.5
    z_trim = max(0.5, min(5.0, z_trim))

    fits: list[float] = []
    per_feature: dict[str, Any] = {}
    for i, key in enumerate(NORM_KEYS):
        if i >= len(x):
            break
        block = norm.get(key)
        if not isinstance(block, dict):
            continue
        try:
            mu = float(block.get("mean", 0.0))
            sig = float(block.get("std", 1.0))
        except (TypeError, ValueError):
            continue
        sig = max(1e-6, sig)
        xv = float(x[i])
        z = (xv - mu) / sig
        fit = max(0.0, 1.0 - min(1.0, abs(z) / z_trim))
        fits.append(fit)
        per_feature[key] = {"z": round(z, 4), "fit": round(fit, 4), "mean": mu, "std": sig}

    if not fits:
        return 50.0, {"error": "no_normative_overlap", "per_feature": per_feature}

    score = 100.0 * sum(fits) / len(fits)
    return max(0.0, min(100.0, score)), {"per_feature": per_feature, "n_used": len(fits)}
