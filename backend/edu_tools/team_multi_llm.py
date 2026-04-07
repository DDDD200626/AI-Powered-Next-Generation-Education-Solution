"""Gemini·ChatGPT·Claude·Grok 병렬 팀 기여도 평가 — 응답을 평균·병합."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from learning_analysis.llm_clients import (
    ensure_gemini_configured,
    get_anthropic_client,
    get_openai_client,
    get_openai_xai_client,
)

from edu_tools.team import (
    DimensionScores,
    DivergenceAxisSummary,
    EvaluationPipelineStep,
    ExplainabilityEntry,
    MemberContributionCompare,
    MemberOut,
    ModelEvalBundle,
    OpinionDivergenceAnalysis,
    TeamCompareResponse,
    TeamEvaluateRequest,
    TeamEvaluateResponse,
    TrustScoreBlock,
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


def _default_model_label(provider: str) -> str:
    """각 제공자별 최신에 가까운 기본 모델(환경변수로 덮어쓰기)."""
    return {
        "gemini": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        "openai": os.environ.get("OPENAI_MODEL", "gpt-4o"),
        "claude": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        "grok": os.environ.get("GROK_MODEL", "grok-3-latest"),
    }.get(provider, provider)


def _call_gemini(req: TeamEvaluateRequest, api_key: str) -> dict[str, Any] | None:
    import google.generativeai as genai

    model_name = _default_model_label("gemini")
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
    model_name = _default_model_label("openai")
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
    model_name = _default_model_label("claude")
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
    model_name = _default_model_label("grok")
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


def _run_one_model(
    key: str,
    label: str,
    req: TeamEvaluateRequest,
    call: Callable[[], dict[str, Any] | None],
) -> ModelEvalBundle:
    try:
        data = call()
        if data is None:
            return ModelEvalBundle(key=key, label=label, ok=False, error="모델 응답 없음")
        if not isinstance(data, dict) or not data.get("members"):
            return ModelEvalBundle(key=key, label=label, ok=False, error="응답에 members가 없습니다.")
        p = _parse_one_model(data, req)
        if p is None:
            return ModelEvalBundle(key=key, label=label, ok=False, error="멤버 매칭 실패")
        members, cis, fn = p
        resp = _finalize_members(req, members, cis, "ai", fn, enrich_openai=False)
        return ModelEvalBundle(key=key, label=label, ok=True, result=resp)
    except Exception as e:
        return ModelEvalBundle(key=key, label=label, ok=False, error=str(e)[:800])


def _find_member_in_bundle(b: ModelEvalBundle, name: str) -> MemberOut | None:
    if not b.ok or b.result is None:
        return None
    nm = name.strip()
    return next((x for x in b.result.members if x.name.strip() == nm), None)


def _dimension_spreads_for_member(nm: str, ok_b: list[ModelEvalBundle]) -> dict[str, float]:
    out: dict[str, float] = {}
    for axis, attr in ("technical", "technical"), ("collaboration", "collaboration"), ("initiative", "initiative"):
        vals: list[float] = []
        for b in ok_b:
            mm = _find_member_in_bundle(b, nm)
            if mm is not None:
                vals.append(float(getattr(mm.dimensions, attr)))
        if len(vals) > 1:
            out[axis] = round(max(vals) - min(vals), 2)
        elif vals:
            out[axis] = 0.0
        else:
            out[axis] = 0.0
    return out


def _member_comparison_rows(
    req: TeamEvaluateRequest,
    bundles: list[ModelEvalBundle],
) -> tuple[list[MemberContributionCompare], str]:
    ok_b = [b for b in bundles if b.ok and b.result is not None]
    if not ok_b:
        return [], "성공한 모델이 없습니다. API 키·모델명(최신 ID)을 확인하세요."
    rows: list[MemberContributionCompare] = []
    for m in req.members:
        nm = m.name.strip()
        by_model: dict[str, float] = {}
        for b in ok_b:
            if b.result is None:
                continue
            mm = next((x for x in b.result.members if x.name.strip() == nm), None)
            if mm is not None:
                by_model[b.key] = float(mm.contribution_index)
        if not by_model:
            continue
        vals = list(by_model.values())
        spread = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
        mean = float(sum(vals) / len(vals))
        dim_sp = _dimension_spreads_for_member(nm, ok_b)
        rows.append(
            MemberContributionCompare(
                member_name=nm,
                contribution_index_by_model=by_model,
                spread=round(spread, 2),
                mean=round(mean, 2),
                dimension_spread=dim_sp,
            )
        )
    if not rows:
        return [], "팀원 이름과 모델 출력 이름이 일치하지 않습니다."
    avg_spread = sum(r.spread for r in rows) / len(rows)
    summary = (
        f"[AI 멀티평가] 연동 모델 {len(ok_b)}개. 기여 지수 편차 평균 약 {avg_spread:.1f}점. "
        "아래 ‘의견 차이·신뢰도·설명’ 패널을 함께 확인하세요."
    )
    return rows, summary


def _criteria_segments(rubric: str) -> list[str]:
    if not rubric.strip():
        return []
    parts = re.split(r"[\n■]+", rubric)
    return [p.strip() for p in parts if p.strip()][:12]


def _rubric_keyword_overlap_score(rubric: str, texts: list[str]) -> tuple[float, str]:
    if not rubric.strip():
        return 50.0, "평가 기준이 비어 있어 루브릭 일치도는 중립(50)으로 두었습니다."
    words = set(re.findall(r"[\w가-힣]{2,}", rubric.lower()))
    if not words:
        return 50.0, "평가 기준에서 추출할 키워드가 없습니다."
    joined = " ".join(texts).lower()
    hit = sum(1 for w in words if w in joined)
    ratio = hit / max(1, len(words))
    score = min(100.0, ratio * 120.0)
    note = f"평가 기준 키워드 {len(words)}개 중 모델 근거 문장에 등장한 유형 약 {hit}개 (정합 점수 참고)."
    return round(score, 1), note


def _explanation_quality_score(ok_b: list[ModelEvalBundle]) -> float:
    scores: list[float] = []
    for b in ok_b:
        if b.result is None:
            continue
        for mm in b.result.members:
            ev = (mm.evidence_summary or "").strip()
            cv = (mm.caveats or "").strip()
            base = min(100.0, len(ev) / 12.0) * 0.72 + min(100.0, len(cv) / 6.0) * 0.18 + (10.0 if ev else 0.0)
            scores.append(min(100.0, base))
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 1)


def _build_divergence_analysis(
    req: TeamEvaluateRequest,
    ok_b: list[ModelEvalBundle],
    mrows: list[MemberContributionCompare],
) -> OpinionDivergenceAnalysis | None:
    if not ok_b or not mrows:
        return None
    axis_defs = [
        ("contribution_index", "기여 지수(종합)"),
        ("technical", "기술·구현 차원"),
        ("collaboration", "협업·소통 차원"),
        ("initiative", "주도성·기획 차원"),
    ]
    summaries: list[DivergenceAxisSummary] = []
    for axis_id, label_ko in axis_defs:
        spreads: list[float] = []
        max_member = ""
        max_sp = -1.0
        for row in mrows:
            if axis_id == "contribution_index":
                sp = row.spread
            elif axis_id == "technical":
                sp = row.dimension_spread.get("technical", 0.0)
            elif axis_id == "collaboration":
                sp = row.dimension_spread.get("collaboration", 0.0)
            else:
                sp = row.dimension_spread.get("initiative", 0.0)
            spreads.append(sp)
            if sp > max_sp:
                max_sp = sp
                max_member = row.member_name
        mean_sp = sum(spreads) / len(spreads) if spreads else 0.0
        interp = (
            f"{label_ko}에서 모델 간 평균 편차는 약 {mean_sp:.1f}점입니다. "
            f"가장 크게 엇갈린 팀원은 {max_member or '—'}입니다."
        )
        summaries.append(
            DivergenceAxisSummary(
                axis_id=axis_id,
                axis_label_ko=label_ko,
                mean_spread=round(mean_sp, 2),
                max_spread_member=max_member,
                interpretation=interp,
            )
        )
    primary = sorted(summaries, key=lambda s: s.mean_spread, reverse=True)[:3]
    crit = _criteria_segments(req.evaluation_criteria)
    texts: list[str] = []
    for b in ok_b:
        if b.result:
            for mm in b.result.members:
                texts.append(mm.evidence_summary or "")
                texts.append(mm.caveats or "")
    _rub_s, rub_note = _rubric_keyword_overlap_score(req.evaluation_criteria, texts)
    if not primary:
        nav = ""
    else:
        nav = (
            f"모델 간 가장 크게 갈린 축(평균 편차 기준)은 "
            f"「{primary[0].axis_label_ko}」({primary[0].mean_spread:.1f}점)입니다."
        )
        if len(primary) > 1:
            nav += f" 다음으로 「{primary[1].axis_label_ko}」({primary[1].mean_spread:.1f}점)."
        if len(primary) > 2:
            nav += f" 「{primary[2].axis_label_ko}」({primary[2].mean_spread:.1f}점)에서도 차이가 누적됩니다."
    return OpinionDivergenceAnalysis(
        primary_axes=primary,
        criteria_segments=crit,
        criteria_keyword_overlap_note=rub_note,
        narrative=nav,
    )


def _build_trust_scores(
    req: TeamEvaluateRequest,
    ok_b: list[ModelEvalBundle],
    mrows: list[MemberContributionCompare],
) -> TrustScoreBlock | None:
    if not ok_b or not mrows:
        return None
    avg_spread = sum(r.spread for r in mrows) / len(mrows)
    consistency = max(0.0, min(100.0, 100.0 - avg_spread * 1.35))
    texts: list[str] = []
    for b in ok_b:
        if b.result:
            for mm in b.result.members:
                texts.append(mm.evidence_summary or "")
    rub_s, _ = _rubric_keyword_overlap_score(req.evaluation_criteria, texts)
    expl = _explanation_quality_score(ok_b)
    overall = round(consistency * 0.42 + rub_s * 0.33 + expl * 0.25, 1)
    notes = [
        f"일관성: 모델 간 기여 지수 편차가 작을수록 높음(현재 평균 편차 {avg_spread:.1f}점).",
        f"루브릭 일치: 입력 평가 기준 키워드가 근거 문장에 얼마나 반영됐는지 근사.",
        f"설명 품질: 근거·주의 문장 길이·구체성 휴리스틱.",
    ]
    return TrustScoreBlock(
        consistency_0_100=round(consistency, 1),
        rubric_alignment_0_100=round(rub_s, 1),
        explanation_quality_0_100=expl,
        overall_trust_0_100=overall,
        notes=notes,
    )


def _build_explainability(ok_b: list[ModelEvalBundle]) -> list[ExplainabilityEntry]:
    out: list[ExplainabilityEntry] = []
    for b in ok_b:
        if not b.ok or b.result is None:
            continue
        for mm in b.result.members:
            d = mm.dimensions
            why = (
                f"{b.label}: 기여 {mm.contribution_index:.0f}점 — "
                f"기술 {d.technical:.0f} · 협업 {d.collaboration:.0f} · 주도 {d.initiative:.0f}. "
                f"{(mm.evidence_summary or '')[:160]}"
            )
            out.append(
                ExplainabilityEntry(
                    model_key=b.key,
                    model_label=b.label,
                    member_name=mm.name,
                    contribution_index=float(mm.contribution_index),
                    technical=float(d.technical),
                    collaboration=float(d.collaboration),
                    initiative=float(d.initiative),
                    evidence_summary=mm.evidence_summary or "",
                    caveats=mm.caveats or "",
                    why_one_liner=why[:500],
                )
            )
    return out


def _build_pipeline_steps(
    req: TeamEvaluateRequest,
    *,
    no_jobs: bool,
    bundles: list[ModelEvalBundle],
    mrows: list[MemberContributionCompare],
    divergence: OpinionDivergenceAnalysis | None,
    trust_scores: TrustScoreBlock | None,
    explainability: list[ExplainabilityEntry],
) -> list[EvaluationPipelineStep]:
    """STEP 1~6 고정 라벨 + 이번 실행 상태."""
    titles = (
        (1, "데이터 수집 (Git, 협업 로그)"),
        (2, "각 AI 평가 수행"),
        (3, "루브릭 기반 점수 생성"),
        (4, "AI 간 차이 분석"),
        (5, "신뢰도 계산"),
        (6, "최종 점수 + 피드백 생성"),
    )
    n_mem = len(req.members)
    n_edge = len(req.collaboration_edges)
    s1 = f"팀원 {n_mem}명·협업 간선 {n_edge}건 등 입력 지표를 수집했습니다."
    if no_jobs:
        return [
            EvaluationPipelineStep(step=1, title_ko=titles[0][1], status="completed", detail=s1),
            EvaluationPipelineStep(
                step=2,
                title_ko=titles[1][1],
                status="skipped",
                detail="호출할 LLM API 키가 없어 평가를 건너뜁니다.",
            ),
            EvaluationPipelineStep(
                step=3,
                title_ko=titles[2][1],
                status="skipped",
                detail="모델 출력이 없어 루브릭 반영 점수를 만들지 못했습니다.",
            ),
            EvaluationPipelineStep(
                step=4,
                title_ko=titles[3][1],
                status="skipped",
                detail="복수 모델 결과가 없어 의견 차이 분석을 생략했습니다.",
            ),
            EvaluationPipelineStep(
                step=5,
                title_ko=titles[4][1],
                status="skipped",
                detail="신뢰도 지표를 계산할 데이터가 없습니다.",
            ),
            EvaluationPipelineStep(
                step=6,
                title_ko=titles[5][1],
                status="skipped",
                detail="비교표·피드백을 생성하지 못했습니다.",
            ),
        ]

    ok_b = [b for b in bundles if b.ok and b.result is not None]
    failed = [b for b in bundles if not b.ok]
    st2 = (
        f"연동 모델 {len(ok_b)}개 성공"
        + (f", {len(failed)}개 실패 또는 미호출" if failed else "")
        + "."
    )
    if not ok_b:
        return [
            EvaluationPipelineStep(step=1, title_ko=titles[0][1], status="completed", detail=s1),
            EvaluationPipelineStep(
                step=2,
                title_ko=titles[1][1],
                status="skipped",
                detail="성공한 모델이 없습니다. API 키·응답 형식을 확인하세요.",
            ),
            EvaluationPipelineStep(
                step=3,
                title_ko=titles[2][1],
                status="skipped",
                detail="모델 출력이 없어 루브릭 반영 점수를 만들지 못했습니다.",
            ),
            EvaluationPipelineStep(
                step=4,
                title_ko=titles[3][1],
                status="skipped",
                detail="복수 모델 결과가 없어 의견 차이 분석을 생략했습니다.",
            ),
            EvaluationPipelineStep(
                step=5,
                title_ko=titles[4][1],
                status="skipped",
                detail="신뢰도 지표를 계산할 데이터가 없습니다.",
            ),
            EvaluationPipelineStep(
                step=6,
                title_ko=titles[5][1],
                status="skipped",
                detail="최종 비교표·피드백을 생성하지 못했습니다.",
            ),
        ]

    st2_status = "partial" if failed else "completed"
    n_crit = len(_criteria_segments(req.evaluation_criteria))
    if (req.evaluation_criteria or "").strip():
        st3 = f"입력 평가 기준을 반영해 차원별 점수 산출(기준 발췌 {n_crit}개)."
    else:
        st3 = "평가 기준 미입력 시 모델 기본 해석으로 차원 점수를 산출했습니다."

    if divergence is not None:
        st4 = "축별 편차·루브릭 키워드 정합 등으로 모델 간 의견 차이를 요약했습니다."
        st4_status = "completed"
    else:
        st4 = "비교에 필요한 모델·팀원 매칭이 없어 생략했습니다."
        st4_status = "skipped"

    if trust_scores is not None:
        st5 = (
            f"일관성·루브릭 일치·설명 품질을 종합한 신뢰도 약 "
            f"{trust_scores.overall_trust_0_100:.1f}점."
        )
        st5_status = "completed"
    else:
        st5 = "신뢰도 지표를 계산할 표본이 없어 생략했습니다."
        st5_status = "skipped"

    if mrows:
        st6 = (
            f"모델 간 기여 지수 비교표와 모델×팀원 설명 {len(explainability)}개를 생성했습니다."
        )
        st6_status = "completed"
    elif explainability:
        st6 = "비교표는 이름 불일치 등으로 비었으나, 모델별 근거·피드백 문장은 일부 있습니다."
        st6_status = "partial"
    else:
        st6 = "최종 비교표·피드백 블록을 생성하지 못했습니다."
        st6_status = "skipped"

    return [
        EvaluationPipelineStep(step=1, title_ko=titles[0][1], status="completed", detail=s1),
        EvaluationPipelineStep(step=2, title_ko=titles[1][1], status=st2_status, detail=st2),
        EvaluationPipelineStep(step=3, title_ko=titles[2][1], status="completed", detail=st3),
        EvaluationPipelineStep(step=4, title_ko=titles[3][1], status=st4_status, detail=st4),
        EvaluationPipelineStep(step=5, title_ko=titles[4][1], status=st5_status, detail=st5),
        EvaluationPipelineStep(step=6, title_ko=titles[5][1], status=st6_status, detail=st6),
    ]


def run_team_compare(req: TeamEvaluateRequest) -> TeamCompareResponse:
    """ChatGPT·Gemini·Claude·Grok(최신 기본 모델) 각각 독립 평가 후 비교표 생성."""
    gk = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    ok_openai = (os.environ.get("OPENAI_API_KEY") or "").strip()
    ak = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    xk = (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or "").strip()

    jobs: list[tuple[str, str, Callable[[], dict[str, Any] | None]]] = []
    if gk:
        jobs.append(("gemini", _default_model_label("gemini"), lambda: _call_gemini(req, gk)))
    if ok_openai:
        jobs.append(("openai", _default_model_label("openai"), lambda: _call_openai(req, ok_openai)))
    if ak:
        jobs.append(("claude", _default_model_label("claude"), lambda: _call_claude(req, ak)))
    if xk:
        jobs.append(("grok", _default_model_label("grok"), lambda: _call_grok(req, xk)))

    if not jobs:
        pipe = _build_pipeline_steps(
            req,
            no_jobs=True,
            bundles=[],
            mrows=[],
            divergence=None,
            trust_scores=None,
            explainability=[],
        )
        return TeamCompareResponse(
            pipeline_steps=pipe,
            models=[
                ModelEvalBundle(key="gemini", label=_default_model_label("gemini"), ok=False, error="API 키 없음"),
                ModelEvalBundle(key="openai", label=_default_model_label("openai"), ok=False, error="API 키 없음"),
                ModelEvalBundle(key="claude", label=_default_model_label("claude"), ok=False, error="API 키 없음"),
                ModelEvalBundle(key="grok", label=_default_model_label("grok"), ok=False, error="API 키 없음"),
            ],
            comparison_summary="GOOGLE/GEMINI·OPENAI·ANTHROPIC·XAI 키 중 하나 이상을 backend/.env에 설정하세요.",
            divergence=None,
            trust_scores=None,
            explainability=[],
        )

    max_workers = min(4, len(jobs))
    bundles_map: dict[str, ModelEvalBundle] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_key = {
            pool.submit(_run_one_model, k, lbl, req, fn): k for k, lbl, fn in jobs
        }
        for fut in as_completed(fut_to_key):
            b = fut.result()
            bundles_map[b.key] = b

    bundles = [bundles_map[k] for k, _, _ in jobs if k in bundles_map]
    mrows, summary = _member_comparison_rows(req, bundles)
    ok_b = [b for b in bundles if b.ok and b.result is not None]
    divergence = _build_divergence_analysis(req, ok_b, mrows) if ok_b and mrows else None
    trust_scores = _build_trust_scores(req, ok_b, mrows) if ok_b and mrows else None
    explainability = _build_explainability(ok_b) if ok_b else []
    pipe = _build_pipeline_steps(
        req,
        no_jobs=False,
        bundles=bundles,
        mrows=mrows,
        divergence=divergence,
        trust_scores=trust_scores,
        explainability=explainability,
    )
    return TeamCompareResponse(
        pipeline_steps=pipe,
        models=bundles,
        member_comparison=mrows,
        comparison_summary=summary,
        divergence=divergence,
        trust_scores=trust_scores,
        explainability=explainability,
    )
