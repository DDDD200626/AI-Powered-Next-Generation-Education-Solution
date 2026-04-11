"""선택적 LLM 정렬 레이어: PyTorch/선형 dl_score에 언어·맥락 신호를 소폭 혼합.

- 전체 팀을 **한 번의 JSON 응답**으로 처리(지연·비용 통제).
- 키 없거나 TEAM_DL_LLM_ASSIST=0 이면 무동작.
- 숫자는 교육·참고용; 생성형 모델의 한계·편향은 contest_transparency와 동일하게 고지.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from learning_analysis.llm_clients import gemini_generate_content, get_openai_client


def _enabled() -> bool:
    return (os.environ.get("TEAM_DL_LLM_ASSIST", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _blend_w() -> float:
    raw = (os.environ.get("TEAM_DL_LLM_BLEND", "0.18") or "0.18").strip()
    try:
        v = float(raw)
    except ValueError:
        v = 0.18
    return max(0.0, min(0.45, v))


def _rule_dl_blend_weights(dl_confidence: float | None, dl_uncertainty: float | None) -> tuple[float, float]:
    c = float(dl_confidence or 0.0)
    u = float(dl_uncertainty or 0.0)
    conf_term = max(0.0, min(1.0, c / 100.0))
    unc_term = max(0.0, min(1.0, 1.0 - (u / 100.0)))
    dl_w = 0.22 + 0.26 * ((0.65 * conf_term) + (0.35 * unc_term))
    dl_w = max(0.22, min(0.48, dl_w))
    return 1.0 - dl_w, dl_w


def _extract_json(text: str) -> dict[str, Any]:
    t = text.strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    return json.loads(t)


def _gemini_json(system: str, user_payload: dict[str, Any], api_key: str) -> dict[str, Any] | None:
    model_name = (os.environ.get("TEAM_DL_LLM_GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()
    t0 = time.perf_counter()
    text = gemini_generate_content(
        api_key,
        model_name,
        json.dumps(user_payload, ensure_ascii=False),
        system_instruction=system,
        response_mime_type="application/json",
        temperature=0.15,
        max_output_tokens=4096,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    if not text:
        return None
    return {"data": _extract_json(text), "model": model_name, "latency_ms": round(ms, 2), "provider": "gemini"}


def _openai_json(system: str, user_payload: dict[str, Any], api_key: str) -> dict[str, Any] | None:
    model = (os.environ.get("TEAM_DL_LLM_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    client = get_openai_client(api_key)
    t0 = time.perf_counter()
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.15,
        max_tokens=4096,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    text = (res.choices[0].message.content or "").strip()
    if not text:
        return None
    return {"data": _extract_json(text), "model": model, "latency_ms": round(ms, 2), "provider": "openai"}


_SYSTEM_KO = """당신은 대학 팀 프로젝트 평가를 돕는 조교입니다.
입력 JSON의 각 멤버에 대해 **기여도 서술·수치가 얼마나 일관되고 타당한지** 0~100 alignment_score를 부여합니다.
숫자만 재계산하지 말고, 자기서술과 Git 지표의 정합성·구체성을 고려합니다.
출력은 JSON 한 개뿐: {"members":[{"member_name":"이름과 입력 동일","alignment_score":0-100,"rationale_one_line_ko":"한 줄"}]}
멤버 수·순서·이름은 입력과 정확히 일치해야 합니다."""


def apply_dl_llm_assist_layer(
    users: list[Any],
    scores: list[Any],
    unc_list: list[float | None],
) -> dict[str, Any] | None:
    """dl_score·blendedScore를 갱신하고 메타를 반환. 비활성/실패 시 None."""
    if not _enabled() or not users or not scores or len(users) != len(scores):
        return None
    if len(unc_list) != len(scores):
        return None

    gk = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    ok = (os.environ.get("OPENAI_API_KEY") or "").strip()

    rows = []
    for u, s in zip(users, scores):
        excerpt = (u.selfReport or "").strip()[:900]
        rows.append(
            {
                "member_name": u.name.strip(),
                "commits": u.commits,
                "prs": u.prs,
                "lines": u.lines,
                "attendance": u.attendance,
                "self_report_excerpt": excerpt,
                "rule_normalized_score": round(float(s.normalizedScore), 2),
                "dl_score_before_llm": s.dl_score,
            }
        )

    payload = {
        "instruction": "각 멤버 alignment_score와 한 줄 근거를 한국어로.",
        "members": rows,
    }

    raw_out: dict[str, Any] | None = None
    if gk:
        try:
            raw_out = _gemini_json(_SYSTEM_KO, payload, gk)
        except Exception:
            raw_out = None
    if raw_out is None and ok:
        try:
            raw_out = _openai_json(_SYSTEM_KO, payload, ok)
        except Exception:
            raw_out = None

    if not raw_out or not isinstance(raw_out.get("data"), dict):
        return {
            "enabled": True,
            "applied": False,
            "reason": "no_api_or_parse_failed",
            "note_ko": "TEAM_DL_LLM_ASSIST=1 이지만 Gemini/OpenAI 호출 실패 또는 키 없음",
        }

    data = raw_out["data"]
    arr = data.get("members")
    if not isinstance(arr, list) or len(arr) != len(scores):
        return {
            "enabled": True,
            "applied": False,
            "reason": "bad_member_count",
            "note_ko": "LLM JSON 멤버 수 불일치",
        }

    blend = _blend_w()
    by_name = {str(x.get("member_name", "")).strip(): x for x in arr if isinstance(x, dict)}

    for i, s in enumerate(scores):
        name = users[i].name.strip()
        row = by_name.get(name)
        if not row:
            return {
                "enabled": True,
                "applied": False,
                "reason": "name_mismatch",
                "note_ko": f"LLM 응답에 멤버 누락: {name!r}",
            }
        try:
            align = float(row.get("alignment_score", 50))
        except (TypeError, ValueError):
            align = 50.0
        align = max(0.0, min(100.0, align))
        rationale = str(row.get("rationale_one_line_ko") or "")[:400]
        s.llm_alignment_score = round(align, 2)
        base_dl = float(s.dl_score) if s.dl_score is not None else float(s.normalizedScore)
        merged = max(0.0, min(100.0, (1.0 - blend) * base_dl + blend * align))
        s.dl_score = round(merged, 2)
        rule_w, dl_w = _rule_dl_blend_weights(s.dl_confidence, unc_list[i])
        s.blendedScore = round(rule_w * s.normalizedScore + dl_w * s.dl_score, 2)
        # 근거를 dl_top_factors 앞에 한 줄(옵션)
        if rationale:
            s.dl_top_factors = [f"LLM 정합: {rationale[:120]}"] + (s.dl_top_factors or [])[:4]

    return {
        "enabled": True,
        "applied": True,
        "blend_weight": blend,
        "provider": raw_out.get("provider"),
        "model": raw_out.get("model"),
        "latency_ms": raw_out.get("latency_ms"),
        "note_ko": (
            "대형 언어모델이 자기서술·지표 정합성을 참고해 alignment_score를 제시하고, "
            f"이를 dl_score에 최대 {blend:.0%}까지 혼합했습니다. 참고용이며 단정·징계 근거로 단독 사용하지 마세요."
        ),
    }
