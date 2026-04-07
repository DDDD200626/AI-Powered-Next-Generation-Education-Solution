"""OpenAI·Anthropic·Gemini 클라이언트 재사용 — 연결·TLS 핸드셰이크 반복 감소."""

from __future__ import annotations

import os
import threading
from typing import Any

_lock = threading.Lock()
_openai_by_key: dict[str, Any] = {}
_openai_xai: dict[tuple[str, str], Any] = {}
_anthropic_by_key: dict[str, Any] = {}
_gemini_configured_for: str | None = None


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


def ensure_gemini_configured(api_key: str) -> None:
    """google.generativeai.configure 는 전역이므로 키가 바뀔 때만 호출."""
    global _gemini_configured_for

    import google.generativeai as genai

    with _lock:
        if _gemini_configured_for != api_key:
            genai.configure(api_key=api_key)
            _gemini_configured_for = api_key
