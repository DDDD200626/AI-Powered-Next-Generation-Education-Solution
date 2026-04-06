"""Gemini, ChatGPT(OpenAI), Claude(Anthropic), Grok(xAI) 호출."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from learning_analysis.prompts import SYSTEM_KO, user_message
from learning_analysis.schemas import AnalyzeRequest, ModelJudgment


def _parse_json_from_text(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    return json.loads(t)


def _to_judgment(provider: str, model_label: str, data: dict[str, Any]) -> ModelJudgment:
    cl = data.get("cheating_likelihood")
    try:
        clf = float(cl) if cl is not None else None
        if clf is not None:
            clf = max(0.0, min(100.0, clf))
    except (TypeError, ValueError):
        clf = None
    return ModelJudgment(
        provider=provider,
        model_label=model_label,
        ok=True,
        cheating_likelihood=clf,
        learning_state_summary=str(data.get("learning_state_summary") or "")[:4000],
        mismatch_analysis=str(data.get("mismatch_analysis") or "")[:4000],
        future_prediction=str(data.get("future_prediction") or "")[:4000],
        confidence_note=str(data.get("confidence_note") or "")[:2000],
    )


def _fail(provider: str, model_label: str, err: str) -> ModelJudgment:
    return ModelJudgment(
        provider=provider,
        model_label=model_label,
        ok=False,
        error=err[:1500],
    )


def call_gemini(req: AnalyzeRequest, api_key: str) -> ModelJudgment:
    import google.generativeai as genai

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_KO)
    try:
        res = model.generate_content(
            user_message(req),
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        )
        text = (res.text or "").strip()
        data = _parse_json_from_text(text)
        return _to_judgment("gemini", model_name, data)
    except Exception as e:
        return _fail("gemini", model_name, str(e))


def call_openai(req: AnalyzeRequest, api_key: str) -> ModelJudgment:
    from openai import OpenAI

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_KO},
                {"role": "user", "content": user_message(req)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        text = (res.choices[0].message.content or "").strip()
        data = _parse_json_from_text(text)
        return _to_judgment("openai", model_name, data)
    except Exception as e:
        return _fail("openai", model_name, str(e))


def call_claude(req: AnalyzeRequest, api_key: str) -> ModelJudgment:
    import anthropic

    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    client = anthropic.Anthropic(api_key=api_key)
    try:
        res = client.messages.create(
            model=model_name,
            max_tokens=2048,
            system=SYSTEM_KO,
            messages=[{"role": "user", "content": user_message(req)}],
            temperature=0.2,
        )
        parts: list[str] = []
        for block in res.content:
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
        text = "".join(parts) if parts else ""
        data = _parse_json_from_text(text)
        return _to_judgment("claude", model_name, data)
    except Exception as e:
        return _fail("claude", model_name, str(e))


def call_grok(req: AnalyzeRequest, api_key: str) -> ModelJudgment:
    from openai import OpenAI

    model_name = os.environ.get("GROK_MODEL", "grok-2-latest")
    base = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
    client = OpenAI(api_key=api_key, base_url=base)
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_KO},
                {"role": "user", "content": user_message(req)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        text = (res.choices[0].message.content or "").strip()
        data = _parse_json_from_text(text)
        return _to_judgment("grok", model_name, data)
    except Exception as e:
        return _fail("grok", model_name, str(e))
