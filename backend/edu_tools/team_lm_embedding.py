"""
자기서술 텍스트용 다국어 문장 임베딩(트랜스포머) → 8차원 요약 피처.

- 기본 TEAM_SEMANTIC_ENCODER=0: 차원은 0으로 채워 CI·경량 배포에 적합.
- TEAM_SEMANTIC_ENCODER=1: sentence-transformers 로드 후 paraphrase-multilingual-MiniLM-L12-v2
  (약 1.2억 파라미터급) 등으로 인코딩 — 탭ular MLP와 결합해 의미 신호를 추가.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

LM_EXTRA_DIM = 8

_ST_MODEL = None
_ST_LOAD_FAILED = False


def _env_enabled() -> bool:
    v = (os.environ.get("TEAM_SEMANTIC_ENCODER", "0") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def default_model_id() -> str:
    return (
        (os.environ.get("TEAM_SEMANTIC_ENCODER_MODEL") or "").strip()
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def is_semantic_enabled() -> bool:
    return _env_enabled()


def semantic_encoder_meta() -> dict[str, Any]:
    mid = default_model_id()
    return {
        "enabled": _env_enabled(),
        "model_id": mid if _env_enabled() else None,
        "approx_parameters_note": "인코더는 sentence-transformers 카탈로그 기준(다국어 MiniLM 계열 약 1e8 파라미터급).",
        "extra_dims": LM_EXTRA_DIM,
        "note_ko": (
            "TEAM_SEMANTIC_ENCODER=1일 때 자기서술을 문장 임베딩으로 인코딩한 뒤 8개 구간 평균+tanh로 압축합니다. "
            "0이면 해당 8차원은 0으로 채워 기존과 동일한 부담으로 동작합니다."
        ),
    }


def _get_sentence_transformer():
    global _ST_MODEL, _ST_LOAD_FAILED
    if _ST_LOAD_FAILED:
        return None
    if _ST_MODEL is not None:
        return _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        _ST_LOAD_FAILED = True
        return None
    try:
        _ST_MODEL = SentenceTransformer(default_model_id())
    except Exception:
        _ST_LOAD_FAILED = True
        return None
    return _ST_MODEL


def _embedding_to_8(emb: np.ndarray) -> list[float]:
    e = np.asarray(emb, dtype=np.float64).reshape(-1)
    n = int(e.size)
    if n == 0:
        return [0.0] * LM_EXTRA_DIM
    chunk = max(1, n // LM_EXTRA_DIM)
    out: list[float] = []
    for i in range(LM_EXTRA_DIM):
        start = i * chunk
        end = n if i == LM_EXTRA_DIM - 1 else min(n, (i + 1) * chunk)
        seg = e[start:end]
        out.append(float(np.tanh(float(seg.mean()))))
    return out


def embed8_self_report(self_report: str) -> list[float]:
    """자기서술 → 8차원 [−1,1] 근처. 비활성·실패 시 0 벡터."""
    if not _env_enabled():
        return [0.0] * LM_EXTRA_DIM
    t = (self_report or "").strip()
    if len(t) < 2:
        return [0.0] * LM_EXTRA_DIM
    model = _get_sentence_transformer()
    if model is None:
        return [0.0] * LM_EXTRA_DIM
    try:
        # 과도한 길이는 잘라 추론 비용 상한
        text = t[:12000]
        emb = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        return [0.0] * LM_EXTRA_DIM
    return _embedding_to_8(emb)
