"""OpenAI·Anthropic·Gemini 클라이언트 재사용 — 연결·TLS 핸드셰이크 반복 감소."""

from __future__ import annotations

import os
import threading
from typing import Any

_lock = threading.Lock()
_openai_by_key: dict[str, Any] = {}
_openai_xai: dict[tuple[str, str], Any] = {}
_anthropic_by_key: dict[str, Any] = {}
_gemini_by_key: dict[str, Any] = {}


def _openai_timeout_sec() -> float:
    raw = (os.environ.get("OPENAI_TIMEOUT_SEC") or "120").strip()
    try:
        v = float(raw)
    except ValueError:
        v = 120.0
    return max(5.0, min(v, 600.0))


def get_openai_client(api_key: str) -> Any:
    from openai import OpenAI

    timeout = _openai_timeout_sec()
    with _lock:
        if api_key not in _openai_by_key:
            _openai_by_key[api_key] = OpenAI(api_key=api_key, timeout=timeout)
        return _openai_by_key[api_key]


def get_openai_xai_client(api_key: str, base_url: str) -> Any:
    from openai import OpenAI

    key = (api_key, base_url)
    with _lock:
        if key not in _openai_xai:
            _openai_xai[key] = OpenAI(api_key=api_key, base_url=base_url)
        return _openai_xai[key]


def get_anthropic_client(api_key: str) -> Any:
    import anthropic

    with _lock:
        if api_key not in _anthropic_by_key:
            _anthropic_by_key[api_key] = anthropic.Anthropic(api_key=api_key)
        return _anthropic_by_key[api_key]


def get_gemini_client(api_key: str) -> Any:
    """`google-genai` Client (키별 캐시)."""
    from google import genai

    with _lock:
        if api_key not in _gemini_by_key:
            _gemini_by_key[api_key] = genai.Client(api_key=api_key)
        return _gemini_by_key[api_key]


def ensure_gemini_configured(api_key: str) -> None:
    """호환용: Gemini Client를 준비합니다(구 `google.generativeai.configure` 대체)."""
    get_gemini_client(api_key)


def gemini_generate_content(
    api_key: str,
    model: str,
    contents: str,
    *,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """Gemini 텍스트 생성. `response_mime_type='application/json'` 등은 선택."""
    from google.genai import types

    client = get_gemini_client(api_key)
    cfg_kw: dict[str, Any] = {}
    if system_instruction is not None:
        cfg_kw["system_instruction"] = system_instruction
    if response_mime_type is not None:
        cfg_kw["response_mime_type"] = response_mime_type
    if temperature is not None:
        cfg_kw["temperature"] = temperature
    if max_output_tokens is not None:
        cfg_kw["max_output_tokens"] = max_output_tokens
    call_kw: dict[str, Any] = {"model": model, "contents": contents}
    if cfg_kw:
        call_kw["config"] = types.GenerateContentConfig(**cfg_kw)
    response = client.models.generate_content(**call_kw)
    return (getattr(response, "text", None) or "").strip()
