"""Gemini, ChatGPT, Claude, Grok — 동일 프롬프트 자유 응답 병렬 호출."""

from __future__ import annotations

import asyncio
import os
import time

from learning_analysis.llm_clients import (
    gemini_generate_content,
    get_anthropic_client,
    get_openai_client,
    get_openai_xai_client,
)
from learning_analysis.schemas import LLMCompareRequest, LLMCompareResponse, LLMTextResult

DISCLAIMER_KO = (
    "네 모델 출력은 서로 다를 수 있으며 참고용입니다. "
    "채점·징계·법적 판단에는 사용하지 마세요. API 키가 없는 제공자는 건너뜁니다."
)

DEFAULT_SYSTEM = """당신은 교육·연구 보조 AI입니다. 사용자의 요청에 한국어로 답합니다.
필요하면 소제목·목록을 사용하고, 불확실하면 한계를 명시합니다."""


def _gemini_key() -> str | None:
    return (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip() or None


def _openai_key() -> str | None:
    return (os.environ.get("OPENAI_API_KEY") or "").strip() or None


def _anthropic_key() -> str | None:
    return (os.environ.get("ANTHROPIC_API_KEY") or "").strip() or None


def _xai_key() -> str | None:
    return (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or "").strip() or None


def _build_system(req: LLMCompareRequest) -> str:
    parts = [DEFAULT_SYSTEM]
    if req.system_hint.strip():
        parts.append(req.system_hint.strip())
    return "\n\n".join(parts)


def _user_block(req: LLMCompareRequest) -> str:
    if req.task_title.strip():
        return f"[작업 제목]\n{req.task_title.strip()}\n\n[요청]\n{req.prompt}"
    return req.prompt


def call_gemini_freeform(req: LLMCompareRequest, api_key: str) -> LLMTextResult:
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    system = _build_system(req)
    user = _user_block(req)
    try:
        text = gemini_generate_content(
            api_key,
            model_name,
            user,
            system_instruction=system,
            temperature=0.35,
            max_output_tokens=8192,
        )
        return LLMTextResult(provider="gemini", model_label=model_name, ok=True, text=text[:8000])
    except Exception as e:
        return LLMTextResult(provider="gemini", model_label=model_name, ok=False, error=str(e)[:1500])


def call_openai_freeform(req: LLMCompareRequest, api_key: str) -> LLMTextResult:
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = get_openai_client(api_key)
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _build_system(req)},
                {"role": "user", "content": _user_block(req)},
            ],
            temperature=0.35,
            max_tokens=8192,
        )
        text = (res.choices[0].message.content or "").strip()
        return LLMTextResult(provider="openai", model_label=model_name, ok=True, text=text[:8000])
    except Exception as e:
        return LLMTextResult(provider="openai", model_label=model_name, ok=False, error=str(e)[:1500])


def call_claude_freeform(req: LLMCompareRequest, api_key: str) -> LLMTextResult:
    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    client = get_anthropic_client(api_key)
    try:
        res = client.messages.create(
            model=model_name,
            max_tokens=8192,
            system=_build_system(req),
            messages=[{"role": "user", "content": _user_block(req)}],
            temperature=0.35,
        )
        parts: list[str] = []
        for block in res.content:
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
        text = "".join(parts).strip()
        return LLMTextResult(provider="claude", model_label=model_name, ok=True, text=text[:8000])
    except Exception as e:
        return LLMTextResult(provider="claude", model_label=model_name, ok=False, error=str(e)[:1500])


def call_grok_freeform(req: LLMCompareRequest, api_key: str) -> LLMTextResult:
    model_name = os.environ.get("GROK_MODEL", "grok-2-latest")
    base = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
    client = get_openai_xai_client(api_key, base)
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _build_system(req)},
                {"role": "user", "content": _user_block(req)},
            ],
            temperature=0.35,
            max_tokens=8192,
        )
        text = (res.choices[0].message.content or "").strip()
        return LLMTextResult(provider="grok", model_label=model_name, ok=True, text=text[:8000])
    except Exception as e:
        return LLMTextResult(provider="grok", model_label=model_name, ok=False, error=str(e)[:1500])


async def compare_llm_async(req: LLMCompareRequest) -> LLMCompareResponse:
    t_start = time.perf_counter()
    used: list[str] = []
    skipped: list[str] = []
    tasks: list[tuple[str, asyncio.Task]] = []

    gk = _gemini_key()
    if gk:
        used.append("gemini")
        tasks.append(("gemini", asyncio.to_thread(call_gemini_freeform, req, gk)))
    else:
        skipped.append("gemini (GOOGLE_API_KEY 또는 GEMINI_API_KEY 없음)")

    ok = _openai_key()
    if ok:
        used.append("openai")
        tasks.append(("openai", asyncio.to_thread(call_openai_freeform, req, ok)))
    else:
        skipped.append("openai (OPENAI_API_KEY 없음)")

    ak = _anthropic_key()
    if ak:
        used.append("claude")
        tasks.append(("claude", asyncio.to_thread(call_claude_freeform, req, ak)))
    else:
        skipped.append("claude (ANTHROPIC_API_KEY 없음)")

    xk = _xai_key()
    if xk:
        used.append("grok")
        tasks.append(("grok", asyncio.to_thread(call_grok_freeform, req, xk)))
    else:
        skipped.append("grok (XAI_API_KEY 또는 GROK_API_KEY 없음)")

    results: list[LLMTextResult] = []
    llm_parallel_ms = 0.0
    if tasks:
        t_gather = time.perf_counter()
        out = await asyncio.gather(*(t[1] for t in tasks), return_exceptions=True)
        llm_parallel_ms = (time.perf_counter() - t_gather) * 1000
        for (name, _), res in zip(tasks, out):
            if isinstance(res, BaseException):
                results.append(
                    LLMTextResult(provider=name, model_label="", ok=False, error=str(res)[:1500])
                )
            else:
                results.append(res)

    total_ms = (time.perf_counter() - t_start) * 1000
    local_ms = max(0.0, total_ms - llm_parallel_ms)
    perf = {
        "llm_parallel_ms": round(llm_parallel_ms, 2),
        "local_ms": round(local_ms, 2),
        "total_ms": round(total_ms, 2),
    }

    return LLMCompareResponse(
        providers_used=used,
        providers_skipped=skipped,
        results=results,
        disclaimer=DISCLAIMER_KO,
        perf=perf,
    )
