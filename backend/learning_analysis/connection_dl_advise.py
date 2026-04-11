"""연결 안정성: 로컬 MLP(PyTorch) + 선택적 Gemini 정렬·한국어 코치."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from fastapi import APIRouter
from pydantic import BaseModel, Field

_WEIGHTS_PATH = Path(__file__).with_name("_connection_mlp_weights.json")


def _gemini_key() -> str | None:
    return (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip() or None


def _connection_gemini_enabled() -> bool:
    raw = (os.environ.get("CONNECTION_ADVISE_GEMINI") or "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return _gemini_key() is not None


def _gemini_model_name() -> str:
    return (
        os.environ.get("CONNECTION_ADVISE_GEMINI_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or "gemini-2.0-flash"
    ).strip()


def _build_model() -> nn.Sequential:
    m = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
    raw = json.loads(_WEIGHTS_PATH.read_text(encoding="utf-8"))
    sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}
    m.load_state_dict(sd)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


_MODEL: nn.Sequential | None = None


def get_connection_mlp() -> nn.Sequential:
    global _MODEL
    if _MODEL is None:
        _MODEL = _build_model()
    return _MODEL


class ConnectionSample(BaseModel):
    ok: bool
    latency_ms: float | None = None


class ConnectionAdviseRequest(BaseModel):
    samples: list[ConnectionSample] = Field(default_factory=list, max_length=48)
    navigator_online: bool = True


class ConnectionAdviseResponse(BaseModel):
    stability_score: float = Field(..., description="최종 0~1 (MLP±Gemini 블렌드)")
    aggressive_retry: bool
    suggested_poll_ms: int
    model: str = Field(..., description="사용 조합 라벨")
    mlp_baseline_score: float | None = Field(None, description="경량 MLP 단독 점수")
    coach_brief_ko: str | None = Field(None, description="사용자 행동 힌트(한국어)")
    enhancer: str = Field("mlp", description="mlp | mlp+gemini | mlp+gemini_failed")
    gemini_latency_ms: float | None = None


def features_tensor(body: ConnectionAdviseRequest) -> torch.Tensor:
    """학습 시 사용한 6차원과 동일 스케일(프론트 `connectionFeaturesFromHistory`와 맞춤)."""
    samples = body.samples[-48:]
    if not samples:
        return torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if body.navigator_online else 0.0]],
            dtype=torch.float32,
        )
    n = len(samples)
    fails = sum(1 for s in samples if not s.ok)
    fail_rate = fails / n
    lats = [float(s.latency_ms) for s in samples if s.ok and s.latency_ms is not None]
    if lats:
        mean_lat = sum(lats) / len(lats)
        var = sum((x - mean_lat) ** 2 for x in lats) / len(lats)
        std_lat = var**0.5
    else:
        mean_lat = 2500.0
        std_lat = 400.0
    consec = 0
    for s in reversed(samples):
        if not s.ok:
            consec += 1
        else:
            break
    consec_n = min(1.0, consec / 8.0)
    last_ok_idx: int | None = None
    for i, s in enumerate(reversed(samples)):
        if s.ok:
            last_ok_idx = i
            break
    if last_ok_idx is None:
        t_since_sec = min(180.0, n * 4.0)
    else:
        t_since_sec = min(180.0, float(last_ok_idx) * 3.5)
    t_norm = min(1.0, t_since_sec / 180.0)
    nav = 1.0 if body.navigator_online else 0.0
    vec = [
        fail_rate,
        mean_lat / 2500.0,
        min(1.0, std_lat / 800.0),
        consec_n,
        t_norm,
        nav,
    ]
    return torch.tensor([vec], dtype=torch.float32)


def _extract_json(text: str) -> dict[str, Any]:
    t = text.strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    return json.loads(t)


def _heuristic_coach_ko(score: float, aggressive: bool, nav_online: bool) -> str:
    if not nav_online:
        return "브라우저가 오프라인입니다. 네트워크를 복구한 뒤 탭을 새로고침하세요."
    if score >= 0.72:
        return "패턴상 연결이 양호합니다. 장시간 유휴 후에는 첫 요청이 느릴 수 있어요."
    if aggressive:
        return "최근 실패가 많습니다. 백엔드 터미널·`npm run always:on`으로 :8000 프로세스를 확인하세요."
    return "응답이 불규칙합니다. VPN·프록시·방화벽을 잠시 끄고 다시 시도해 보세요."


def _gemini_connection_coach(body: ConnectionAdviseRequest, mlp_baseline: float) -> dict[str, Any] | None:
    key = _gemini_key()
    if not key:
        return None
    from learning_analysis.llm_clients import gemini_generate_content

    vec = features_tensor(body)[0].tolist()
    payload = {
        "navigator_online": body.navigator_online,
        "mlp_baseline_score": round(mlp_baseline, 4),
        "feature_vector": [round(float(x), 4) for x in vec],
        "recent_samples": [{"ok": s.ok, "latency_ms": s.latency_ms} for s in body.samples[-48:]],
    }
    system = """당신은 로컬 교육용 웹앱(프론트↔127.0.0.1:8000 FastAPI) 연결을 진단하는 SRE 조교입니다.
입력 JSON을 읽고 **반드시 아래 키만 가진 JSON 한 개만** 출력하세요. 마크다운·코드펜스·설명 문장 금지.
키:
- stability_score: 0~1 실수 (높을수록 곧 정상 응답 가능; mlp_baseline_score와 크게 모순되지 않게, 네트워크·타임아웃 패턴이면 소폭 조정 가능)
- aggressive_retry: bool (재시도를 짧게 할지)
- suggested_poll_ms: 400~8000 정수 (다음 헬스 폴링 권장 간격)
- brief_ko: 140자 이내 한국어 한 문장(사용자가 할 행동 한 가지)"""
    model_name = _gemini_model_name()
    t0 = time.perf_counter()
    try:
        text = gemini_generate_content(
            key,
            model_name,
            json.dumps(payload, ensure_ascii=False),
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0.18,
            max_output_tokens=512,
        )
    except Exception:
        return None
    ms = (time.perf_counter() - t0) * 1000.0
    if not text:
        return None
    try:
        data = _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        ss = float(data["stability_score"])
        ss = max(0.0, min(1.0, ss))
        ar = bool(data["aggressive_retry"])
        poll = int(data["suggested_poll_ms"])
        poll = max(400, min(8000, poll))
        brief = str(data.get("brief_ko") or "").strip()[:200]
        if not brief:
            brief = _heuristic_coach_ko(ss, ar, body.navigator_online)
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "stability_score": ss,
        "aggressive_retry": ar,
        "suggested_poll_ms": poll,
        "brief_ko": brief,
        "model": model_name,
        "latency_ms": round(ms, 2),
    }


def _blend_scores(mlp_s: float, gem_s: float) -> float:
    gem_c = max(mlp_s - 0.28, min(mlp_s + 0.28, gem_s))
    return max(0.0, min(1.0, 0.5 * mlp_s + 0.5 * gem_c))


def _derive_poll(score: float, aggressive: bool, nav_online: bool) -> int:
    if aggressive and not nav_online:
        return 900
    if aggressive:
        return 1300 if score < 0.38 else 1600
    if score < 0.55:
        return 2800
    return 4500


router = APIRouter()


@router.post("/advise", response_model=ConnectionAdviseResponse)
async def connection_advise(body: ConnectionAdviseRequest) -> ConnectionAdviseResponse:
    with torch.no_grad():
        x = features_tensor(body)
        mlp_score = float(get_connection_mlp()(x)[0, 0].item())

    enhancer = "mlp"
    gem_latency: float | None = None
    coach = _heuristic_coach_ko(mlp_score, mlp_score < 0.42, body.navigator_online)
    final_score = mlp_score
    aggressive = mlp_score < 0.42
    poll = _derive_poll(final_score, aggressive, body.navigator_online)
    model_label = "connection_mlp_v1"

    if _connection_gemini_enabled():
        g = await asyncio.to_thread(_gemini_connection_coach, body, mlp_score)
        if g:
            final_score = _blend_scores(mlp_score, float(g["stability_score"]))
            aggressive = final_score < 0.42
            poll = int(g["suggested_poll_ms"])
            poll = max(400, min(8000, poll))
            coach = str(g.get("brief_ko") or coach)[:200]
            gem_latency = float(g["latency_ms"])
            model_label = f"connection_mlp_v1+{g.get('model', 'gemini')}"
            enhancer = "mlp+gemini"
        else:
            enhancer = "mlp+gemini_failed"
            coach = _heuristic_coach_ko(mlp_score, aggressive, body.navigator_online)

    return ConnectionAdviseResponse(
        stability_score=round(final_score, 4),
        aggressive_retry=aggressive,
        suggested_poll_ms=poll,
        model=model_label,
        mlp_baseline_score=round(mlp_score, 4),
        coach_brief_ko=coach,
        enhancer=enhancer,
        gemini_latency_ms=gem_latency,
    )
