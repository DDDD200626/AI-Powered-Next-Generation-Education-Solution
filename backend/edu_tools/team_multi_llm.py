"""Gemini·ChatGPT·Claude·Grok 병렬 팀 기여도 평가 — 응답을 평균·병합."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from learning_analysis.llm_clients import (
    ensure_gemini_configured,
    get_anthropic_client,
    get_openai_client,
    get_openai_xai_client,
)

from edu_tools.team import (
    DimensionScores,
    MemberOut,
    TeamEvaluateRequest,
    TeamEvaluateResponse,
    _finalize_members,
    _parse_json,
)

TEAM_SYSTEM_KO = """교육용 팀 기여도 조교입니다. JSON만 출력하세요.
{
  "fairness_notes": "문자열",
  "members": [
    {
      "name": "이름",
      "role": "역할",
      "contribution_index": 0-100,
      "dimensions": {"technical":0-100,"collaboration":0-100,"initiative":0-100},
      "evidence_summary": "근거",
      "caveats": "주의"
    }
  ]
}
커밋 수만으로 순위 매기지 말고 self_report·peer_notes·timeline으로 비가시 기여를 반영하세요."""


def _user_payload(req: TeamEvaluateRequest) -> str:
    return json.dumps(
        {
            "project_name": req.project_name,
            "project_description": req.project_description,
            "evaluation_criteria": req.evaluation_criteria,
            "members": [m.model_dump() for m in req.members],
        },
        ensure_ascii=False,
    )


def _parse_one_model(data: dict[str, Any], req: TeamEvaluateRequest) -> tuple[list[MemberOut], list[float], str] | None:
    rows = {str(r.get("name", "")).strip(): r for r in (data.get("members") or []) if isinstance(r, dict)}
    out: list[MemberOut] = []
    contribution_indices: list[float] = []
    for m in req.members:
        raw = rows.get(m.name.strip()) or rows.get(m.name)
        if raw is None and len(data.get("members") or []) == len(req.members):
            raw = (data.get("members") or [])[len(out)]
        if not isinstance(raw, dict):
            return None
        d = raw.get("dimensions") or {}
        ci = max(0, min(100, float(raw.get("contribution_index", 50))))
        contribution_indices.append(ci)
        out.append(
            MemberOut(
                name=str(raw.get("name") or m.name),
                role=str(raw.get("role") or m.role),
                contribution_index=ci,
                dimensions=DimensionScores(
                    technical=max(0, min(100, float(d.get("technical", 50)))),
                    collaboration=max(0, min(100, float(d.get("collaboration", 50)))),
                    initiative=max(0, min(100, float(d.get("initiative", 50)))),
                ),
                evidence_summary=str(raw.get("evidence_summary", ""))[:2000],
                caveats=str(raw.get("caveats", ""))[:1000],
            )
        )
    if len(out) != len(req.members):
        return None
    fn = str(data.get("fairness_notes", ""))[:2000]
    return out, contribution_indices, fn


def _merge_parsed(
    req: TeamEvaluateRequest,
    parsed: list[tuple[str, list[MemberOut], list[float], str]],
) -> tuple[list[MemberOut], list[float], str]:
    provs = [p for p, _, _, _ in parsed]
    fairness_parts = [fn for _, _, _, fn in parsed if fn.strip()]
    merged_fairness = " | ".join(fairness_parts[:12])
    header = f"[다중 LLM 합의: {', '.join(provs)}] "
    merged_fairness = header + merged_fairness

    out: list[MemberOut] = []
    cis: list[float] = []

    for i, m in enumerate(req.members):
        name = m.name.strip()
        chunks: list[tuple[str, MemberOut]] = []
        for prov, members, _, _ in parsed:
            mm = next((x for x in members if x.name.strip() == name), None)
            if mm is None and i < len(members):
                mm = members[i]
            if mm is not None:
                chunks.append((prov, mm))

        if not chunks:
            for prov, members, _, _ in parsed:
                if i < len(members):
                    chunks.append((prov, members[i]))
                    break
        if not chunks:
            raise ValueError("merge: no member row")

        ci = sum(x[1].contribution_index for x in chunks) / len(chunks)
        tech = sum(x[1].dimensions.technical for x in chunks) / len(chunks)
        coll = sum(x[1].dimensions.collaboration for x in chunks) / len(chunks)
        ini = sum(x[1].dimensions.initiative for x in chunks) / len(chunks)
        evidence = "\n\n".join(f"[{p}] {x.evidence_summary}" for p, x in chunks if x.evidence_summary)
        caveats_m = "\n\n".join(f"[{p}] {x.caveats}" for p, x in chunks if x.caveats)
        role0 = chunks[0][1].role or m.role

        out.append(
            MemberOut(
                name=name,
                role=role0,
                contribution_index=round(ci, 1),
                dimensions=DimensionScores(
                    technical=round(tech, 1),
                    collaboration=round(coll, 1),
                    initiative=round(ini, 1),
                ),
                evidence_summary=evidence[:2000] or chunks[0][1].evidence_summary,
                caveats=caveats_m[:1000] or chunks[0][1].caveats,
            )
        )
        cis.append(round(ci, 1))

    if len(out) != len(req.members):
        raise ValueError("merge mismatch")

    return out, cis, merged_fairness[:4000]


def _call_gemini(req: TeamEvaluateRequest, api_key: str) -> dict[str, Any] | None:
    import google.generativeai as genai

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    ensure_gemini_configured(api_key)
    model = genai.GenerativeModel(model_name=model_name, system_instruction=TEAM_SYSTEM_KO)
    res = model.generate_content(
        _user_payload(req),
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.25,
        },
    )
    text = (res.text or "").strip()
    return _parse_json(text)


def _call_openai(req: TeamEvaluateRequest, api_key: str) -> dict[str, Any] | None:
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = get_openai_client(api_key)
    res = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": TEAM_SYSTEM_KO},
            {"role": "user", "content": _user_payload(req)},
        ],
        response_format={"type": "json_object"},
        temperature=0.25,
    )
    text = (res.choices[0].message.content or "").strip()
    return _parse_json(text)


def _call_claude(req: TeamEvaluateRequest, api_key: str) -> dict[str, Any] | None:
    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    client = get_anthropic_client(api_key)
    res = client.messages.create(
        model=model_name,
        max_tokens=4096,
        system=TEAM_SYSTEM_KO,
        messages=[{"role": "user", "content": _user_payload(req)}],
        temperature=0.25,
    )
    parts: list[str] = []
    for block in res.content:
        t = getattr(block, "text", None)
        if t:
            parts.append(t)
    text = "".join(parts) if parts else ""
    return _parse_json(text)


def _call_grok(req: TeamEvaluateRequest, api_key: str) -> dict[str, Any] | None:
    model_name = os.environ.get("GROK_MODEL", "grok-2-latest")
    base = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
    client = get_openai_xai_client(api_key, base)
    res = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": TEAM_SYSTEM_KO},
            {"role": "user", "content": _user_payload(req)},
        ],
        response_format={"type": "json_object"},
        temperature=0.25,
    )
    text = (res.choices[0].message.content or "").strip()
    return _parse_json(text)


def run_parallel_team_eval(req: TeamEvaluateRequest) -> TeamEvaluateResponse | None:
    """
    설정된 API 키가 있는 제공자만 병렬 호출하고, 성공한 응답들을 팀원별로 평균 병합합니다.
    성공한 모델이 하나도 없으면 None.
    """
    gk = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    ok_openai = (os.environ.get("OPENAI_API_KEY") or "").strip()
    ak = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    xk = (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or "").strip()

    jobs: list[tuple[str, Any]] = []
    if gk:
        jobs.append(("gemini", lambda: _call_gemini(req, gk)))
    if ok_openai:
        jobs.append(("openai", lambda: _call_openai(req, ok_openai)))
    if ak:
        jobs.append(("claude", lambda: _call_claude(req, ak)))
    if xk:
        jobs.append(("grok", lambda: _call_grok(req, xk)))

    if not jobs:
        return None

    raw_ok: list[tuple[str, dict[str, Any]]] = []
    max_workers = min(4, len(jobs))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(fn): name for name, fn in jobs}
        for fut in as_completed(future_map):
            name = future_map[fut]
            try:
                data = fut.result()
                if isinstance(data, dict) and data.get("members"):
                    raw_ok.append((name, data))
            except Exception:
                continue

    if not raw_ok:
        return None

    parsed: list[tuple[str, list[MemberOut], list[float], str]] = []
    for name, data in raw_ok:
        p = _parse_one_model(data, req)
        if p is not None:
            parsed.append((name, *p))

    if not parsed:
        return None

    if len(parsed) == 1:
        prov, members, cis, fn = parsed[0]
        merged_fn = f"[단일 LLM: {prov}] {fn}"
        return _finalize_members(req, members, cis, "ai", merged_fn)

    try:
        out, cis, merged_fn = _merge_parsed(req, parsed)
    except ValueError:
        return None
    return _finalize_members(req, out, cis, "ai", merged_fn)
