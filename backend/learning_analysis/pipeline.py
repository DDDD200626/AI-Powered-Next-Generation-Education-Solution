"""병렬 LLM 호출 및 합의 요약."""

from __future__ import annotations

import asyncio
import os
import threading
import time

from learning_analysis.heuristic import heuristic_judgment
from learning_analysis.providers import call_claude, call_gemini, call_grok, call_openai
from learning_analysis.schemas import AnalyzeRequest, AnalyzeResponse, ModelJudgment

DISCLAIMER_KO = (
    "본 시스템은 교육 보조용 분석입니다. 부정행위 여부는 인간의 심사·증거·절차에 따릅니다. "
    "모델 출력은 확률적이며 오류가 있을 수 있습니다. 법적·징계적 단정에 사용하지 마세요."
)


def _gemini_key() -> str | None:
    return (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip() or None


def _openai_key() -> str | None:
    return (os.environ.get("OPENAI_API_KEY") or "").strip() or None


def _anthropic_key() -> str | None:
    return (os.environ.get("ANTHROPIC_API_KEY") or "").strip() or None


def _xai_key() -> str | None:
    return (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or "").strip() or None


async def analyze_async(req: AnalyzeRequest) -> AnalyzeResponse:
    t_start = time.perf_counter()
    used: list[str] = []
    skipped: list[str] = []
    tasks: list[tuple[str, asyncio.Task]] = []

    gk = _gemini_key()
    if gk:
        used.append("gemini")
        tasks.append(("gemini", asyncio.to_thread(call_gemini, req, gk)))
    else:
        skipped.append("gemini (GOOGLE_API_KEY 또는 GEMINI_API_KEY 없음)")

    okey = _openai_key()
    if okey:
        used.append("openai")
        tasks.append(("openai", asyncio.to_thread(call_openai, req, okey)))
    else:
        skipped.append("openai (OPENAI_API_KEY 없음)")

    ak = _anthropic_key()
    if ak:
        used.append("claude")
        tasks.append(("claude", asyncio.to_thread(call_claude, req, ak)))
    else:
        skipped.append("claude (ANTHROPIC_API_KEY 없음)")

    xk = _xai_key()
    if xk:
        used.append("grok")
        tasks.append(("grok", asyncio.to_thread(call_grok, req, xk)))
    else:
        skipped.append("grok (XAI_API_KEY 또는 GROK_API_KEY 없음)")

    judgments: list[ModelJudgment] = []
    llm_parallel_ms = 0.0

    if tasks:
        t_gather = time.perf_counter()
        results = await asyncio.gather(*(t[1] for t in tasks), return_exceptions=True)
        llm_parallel_ms = (time.perf_counter() - t_gather) * 1000
        for (name, _), res in zip(tasks, results):
            if isinstance(res, BaseException):
                judgments.append(
                    ModelJudgment(
                        provider=name,
                        model_label="",
                        ok=False,
                        error=str(res)[:1500],
                    )
                )
            else:
                judgments.append(res)
    else:
        judgments.append(heuristic_judgment(req))
        used = ["heuristic"]
        skipped = [
            "gemini (GOOGLE_API_KEY 또는 GEMINI_API_KEY 없음)",
            "openai (OPENAI_API_KEY 없음)",
            "claude (ANTHROPIC_API_KEY 없음)",
            "grok (XAI_API_KEY 또는 GROK_API_KEY 없음)",
        ]

    scores = [
        j.cheating_likelihood
        for j in judgments
        if j.ok and j.cheating_likelihood is not None
    ]
    consensus_avg = round(sum(scores) / len(scores), 2) if scores else None

    ok_js = [j for j in judgments if j.ok]
    parts = []
    for j in ok_js:
        parts.append(f"[{j.provider}] 부정행위 의심도: {j.cheating_likelihood}")
    consensus_summary = ""
    if consensus_avg is not None:
        consensus_summary = (
            f"응답한 모델 {len(scores)}개 기준, 부정행위 **의심도 평균 약 {consensus_avg}/100** 입니다. "
            "수치는 참고용이며, 교육적 맥락과 증거를 함께 검토하세요.\n"
        )
    if parts:
        consensus_summary += "\n".join(parts[:8])
    if not consensus_summary.strip():
        consensus_summary = "유효한 모델 응답이 없습니다. API 키·네트워크·모델명을 확인하세요."

    total_ms = (time.perf_counter() - t_start) * 1000
    local_ms = max(0.0, total_ms - llm_parallel_ms)
    perf = {
        "llm_parallel_ms": round(llm_parallel_ms, 2),
        "local_ms": round(local_ms, 2),
        "total_ms": round(total_ms, 2),
    }

    return AnalyzeResponse(
        providers_used=used,
        providers_skipped=skipped,
        judgments=judgments,
        consensus_cheating_avg=consensus_avg,
        consensus_summary=consensus_summary[:6000],
        disclaimer=DISCLAIMER_KO,
        perf=perf,
    )


def analyze_sync(req: AnalyzeRequest) -> AnalyzeResponse:
    return asyncio.run(analyze_async(req))


_STATUS_LOCK = threading.Lock()
_STATUS_CACHE: tuple[float, dict[str, bool]] | None = None
_STATUS_TTL_SEC = 2.0


def provider_keys_status() -> dict[str, bool]:
    """환경 변수 조회는 저빈도 캐시 — /api/health 폴링 시 CPU·락 부담 감소."""
    global _STATUS_CACHE
    now = time.monotonic()
    with _STATUS_LOCK:
        if _STATUS_CACHE is not None and (now - _STATUS_CACHE[0]) < _STATUS_TTL_SEC:
            return _STATUS_CACHE[1]
        st = {
            "gemini": _gemini_key() is not None,
            "openai": _openai_key() is not None,
            "claude": _anthropic_key() is not None,
            "grok": _xai_key() is not None,
        }
        _STATUS_CACHE = (now, st)
        return st
