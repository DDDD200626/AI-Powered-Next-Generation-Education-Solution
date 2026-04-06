"""팀 프로젝트 기여도 평가 — 무임승차 의심, 타임라인, 팀원별 AI 피드백."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from edu_tools.team_advanced import (
    AnomalyAlert,
    CollaborationEdgeIn,
    MismatchItem,
    NetworkGraph,
    build_network,
    compute_anomalies,
    compute_mismatches,
    heuristic_roles,
    openai_enrich_advanced,
)

router = APIRouter()


class TimelinePointIn(BaseModel):
    period_label: str = Field(..., min_length=1)
    activity_score: float = Field(..., ge=0, le=100, description="해당 기간 활동·기여 점수(주관)")


class MemberIn(BaseModel):
    name: str = Field(..., min_length=1)
    role: str = ""
    commits: int | None = Field(None, ge=0)
    pull_requests: int | None = Field(None, ge=0)
    lines_changed: int | None = Field(None, ge=0)
    tasks_completed: int | None = Field(None, ge=0)
    meetings_attended: int | None = Field(None, ge=0)
    self_report: str = ""
    peer_notes: str = ""
    timeline: list[TimelinePointIn] = Field(default_factory=list)
    outcome_score: float | None = Field(
        None,
        ge=0,
        le=100,
        description="프로젝트/발표/동료 평가 등 결과 점수(선택, 기여 추정과 비교)",
    )


class TeamEvaluateRequest(BaseModel):
    project_name: str = Field(..., min_length=1)
    project_description: str = ""
    evaluation_criteria: str = ""
    members: list[MemberIn] = Field(..., min_length=1)
    collaboration_edges: list[CollaborationEdgeIn] = Field(
        default_factory=list,
        description="팀원 간 상호작용(리뷰·회의·페어 등). 없으면 기여도 기반 완전 그래프로 추정",
    )


class DimensionScores(BaseModel):
    technical: float
    collaboration: float
    initiative: float


class TimelinePointOut(BaseModel):
    period_label: str
    share_percent: float = Field(..., ge=0, le=100, description="해당 시점 팀 내 상대 기여 비율")
    activity_score: float | None = Field(None, ge=0, le=100, description="입력값 또는 추정")


class MemberOut(BaseModel):
    name: str
    role: str = ""
    contribution_index: float
    dimensions: DimensionScores
    evidence_summary: str = ""
    caveats: str = ""
    free_rider_suspected: bool = False
    free_rider_risk: float = Field(0, ge=0, le=100, description="무임승차 의심도(높을수록 의심)")
    free_rider_signals: list[str] = Field(default_factory=list)
    ai_feedback: str = ""
    timeline: list[TimelinePointOut] = Field(default_factory=list)
    contribution_type_label: str = Field("", description="개발형·문서형·리더형·서포터형")
    role_scores: dict[str, float] = Field(
        default_factory=dict,
        description="dev, doc, leader, supporter 합산 100 근사",
    )


class TeamEvaluateResponse(BaseModel):
    mode: str
    members: list[MemberOut]
    fairness_notes: str = ""
    free_rider_summary: str = ""
    collaboration_network: NetworkGraph = Field(default_factory=NetworkGraph)
    contribution_outcome_summary: str = ""
    mismatches: list[MismatchItem] = Field(default_factory=list)
    anomaly_alerts: list[AnomalyAlert] = Field(default_factory=list)
    advanced_mode: str = "heuristic"
    disclaimer: str = (
        "팀 프로젝트 기여도 자동 평가(보조) 결과입니다. 최종 성적·인사 판단을 대체하지 않습니다. "
        "무임승차·불일치·네트워크·이상 탐지는 추정이며 면담·추가 증거로 확인하세요."
    )


def _parse_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    return json.loads(t)


def _norm(vals: list[float]) -> list[float]:
    s = sum(vals) or 1.0
    return [v / s for v in vals]


def _synthetic_period_labels(n: int) -> list[str]:
    return [f"{i + 1}주차" for i in range(n)]


def _compute_timelines_from_input(req: TeamEvaluateRequest) -> list[list[TimelinePointOut]]:
    """주차별 입력이 있으면 정규화된 share 시퀀스, 없으면 빈 리스트(후처리에서 합성)."""
    members = req.members
    n = len(members)
    all_periods: list[str] = []
    for m in members:
        for t in m.timeline:
            lab = t.period_label.strip()
            if lab and lab not in all_periods:
                all_periods.append(lab)

    if not all_periods:
        return [[] for _ in range(n)]

    out: list[list[TimelinePointOut]] = []
    for mi, m in enumerate(members):
        by_label = {tp.period_label.strip(): tp.activity_score for tp in m.timeline}
        row: list[TimelinePointOut] = []
        scores_at: list[float] = []
        for lab in all_periods:
            scores_at.append(by_label.get(lab, 0.0))
        total_at = [0.0] * len(all_periods)
        for j in range(len(all_periods)):
            for mi2 in range(n):
                m2 = members[mi2]
                d = {tp.period_label.strip(): tp.activity_score for tp in m2.timeline}
                total_at[j] += d.get(all_periods[j], 0.0)
        for j, lab in enumerate(all_periods):
            tot = total_at[j] or 1.0
            share = 100.0 * scores_at[j] / tot
            row.append(
                TimelinePointOut(
                    period_label=lab,
                    share_percent=round(share, 1),
                    activity_score=round(scores_at[j], 1),
                )
            )
        out.append(row)
    return out


def _synthetic_timelines(contribution_indices: list[float]) -> list[list[TimelinePointOut]]:
    """총합 기여만 있을 때 4주 가상 시계열(팀 내 상대 비중)."""
    n = len(contribution_indices)
    n_periods = 4
    labels = _synthetic_period_labels(n_periods)
    raw: list[list[float]] = [[0.0] * n_periods for _ in range(n)]
    for i, ci in enumerate(contribution_indices):
        for t in range(n_periods):
            w = 0.55 + 0.15 * math.sin((i + 1) * 0.9 + t * 0.7) + 0.12 * (t / max(1, n_periods - 1))
            raw[i][t] = max(0.01, ci * w)
    for t in range(n_periods):
        col = sum(raw[i][t] for i in range(n)) or 1.0
        for i in range(n):
            raw[i][t] = 100.0 * raw[i][t] / col
    out: list[list[TimelinePointOut]] = []
    for i in range(n):
        row = [
            TimelinePointOut(
                period_label=labels[t],
                share_percent=round(raw[i][t], 1),
                activity_score=None,
            )
            for t in range(n_periods)
        ]
        out.append(row)
    return out


def _merge_timelines(
    req: TeamEvaluateRequest, contribution_indices: list[float]
) -> list[list[TimelinePointOut]]:
    from_input = _compute_timelines_from_input(req)
    if from_input and all(len(x) > 0 for x in from_input):
        return from_input
    return _synthetic_timelines(contribution_indices)


def _free_rider_analysis(
    req: TeamEvaluateRequest, contribution_indices: list[float], timelines: list[list[TimelinePointOut]]
) -> tuple[list[bool], list[float], list[list[str]]]:
    n = len(contribution_indices)
    avg = sum(contribution_indices) / n if n else 0.0
    suspected: list[bool] = []
    risks: list[float] = []
    signals: list[list[str]] = []

    for i, m in enumerate(req.members):
        ci = contribution_indices[i]
        sig: list[str] = []
        risk = 0.0

        if n >= 2 and avg > 0 and ci < avg * 0.48:
            risk += 38.0
            sig.append("팀 평균 기여 대비 유의미하게 낮음")

        c0 = (m.commits or 0) + (m.tasks_completed or 0) + (m.pull_requests or 0)
        if c0 == 0 and (m.lines_changed or 0) < 5:
            risk += 22.0
            sig.append("정량 활동(커밋·태스크·PR)이 거의 없음")

        tl = timelines[i] if i < len(timelines) else []
        if len(tl) >= 2:
            shares = [p.share_percent for p in tl]
            if shares[-1] < 12.0 and shares[0] > 18.0:
                risk += 28.0
                sig.append("후반부 팀 내 상대 기여 비중이 급감")

        if len(tl) >= 2:
            shares = [p.share_percent for p in tl]
            if max(shares) - min(shares) > 35:
                sig.append("주차별 기여 편차가 큼(기획·리뷰 등 비가시 기여 확인)")

        peer = (m.peer_notes or "").lower()
        if any(x in peer for x in ["불참", "기여 없", "안 함", "free", "무임"]):
            risk += 15.0
            sig.append("동료 메모에 부정적 키워드가 포함됨")

        risk = max(0.0, min(100.0, risk))
        suspected.append(risk >= 42.0)
        risks.append(round(risk, 1))
        if not sig:
            sig.append("자동 규칙상 특이 신호 없음")
        signals.append(sig)

    return suspected, risks, signals


def _template_feedback(m: MemberOut) -> str:
    return (
        f"{m.name}님의 기여 지수는 {m.contribution_index:.1f}입니다. "
        f"기술·협업·주도 차원은 각각 {m.dimensions.technical:.0f}, {m.dimensions.collaboration:.0f}, {m.dimensions.initiative:.0f} 수준으로 보입니다. "
        f"{'무임승차 의심 신호가 있어 면담·활동 로그 확인을 권장합니다. ' if m.free_rider_suspected else ''}"
        f"{m.evidence_summary[:120] if m.evidence_summary else '구체적 역할과 산출물을 문서화하면 평가 정당성이 높아집니다.'}"
    )


def _attach_advanced(
    req: TeamEvaluateRequest,
    enriched: list[MemberOut],
    suspected: list[bool],
) -> tuple[list[MemberOut], NetworkGraph, list[MismatchItem], list[AnomalyAlert], str, str]:
    names = [m.name for m in req.members]
    cis = [mm.contribution_index for mm in enriched]
    outcomes = [m.outcome_score for m in req.members]

    out_members: list[MemberOut] = []
    for i, mem in enumerate(req.members):
        c = float(mem.commits or 0)
        l_ = float(mem.lines_changed or 0)
        pr = float(mem.pull_requests or 0)
        t = float(mem.tasks_completed or 0)
        meet = float(mem.meetings_attended or 0)
        words = float(max(0, len((mem.self_report or "").split())))
        en = enriched[i]
        dim = en.dimensions
        label, scores = heuristic_roles(
            c, l_, pr, t, meet, words, dim.technical, dim.collaboration, dim.initiative
        )
        out_members.append(
            en.model_copy(update={"contribution_type_label": label, "role_scores": scores})
        )

    mm, summary = compute_mismatches(names, cis, outcomes)
    net = build_network(names, cis, req.collaboration_edges)
    anom = compute_anomalies(names, cis, suspected, mm, net.edges)

    advanced_mode = "heuristic"
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            project = {
                "name": req.project_name,
                "description": req.project_description,
                "criteria": req.evaluation_criteria,
            }
            members_payload = [m.model_dump() for m in req.members]
            heuristic_bundle = {
                "mismatch_summary": summary[:2000],
                "role_labels": [m.contribution_type_label for m in out_members],
            }
            data = openai_enrich_advanced(project, members_payload, heuristic_bundle, key)
            comment = str(data.get("contribution_outcome_comment") or "").strip()
            if comment:
                summary = comment + "\n\n" + summary
            advanced_mode = "openai_enriched"
        except Exception:
            pass

    return out_members, net, mm, anom, summary, advanced_mode


def _openai_feedbacks(req: TeamEvaluateRequest, members: list[MemberOut], api_key: str) -> list[str]:
    from openai import OpenAI

    payload = {
        "project": req.project_name,
        "members": [
            {
                "name": m.name,
                "contribution_index": m.contribution_index,
                "dimensions": m.dimensions.model_dump(),
                "free_rider_suspected": m.free_rider_suspected,
                "signals": m.free_rider_signals[:5],
            }
            for m in members
        ],
    }
    sys = """팀 프로젝트 조교입니다. 각 팀원에게 한국어로 짧은 격려·개선 피드백(2~4문장)을 JSON으로만 출력하세요.
{
  "feedbacks": [{"name": "이름", "feedback": "문자열"}]
}
비난하지 말고, 무임승차 의심이 있으면 사실 확인과 협업 개선을 권하는 톤으로 작성하세요."""

    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.35,
    )
    data = _parse_json(res.choices[0].message.content or "{}")
    by_name = {str(x.get("name", "")).strip(): str(x.get("feedback", "")) for x in (data.get("feedbacks") or []) if isinstance(x, dict)}
    out: list[str] = []
    for m in members:
        fb = by_name.get(m.name.strip()) or by_name.get(m.name)
        out.append((fb or _template_feedback(m))[:2500])
    return out


def _finalize_members(
    req: TeamEvaluateRequest,
    base: list[MemberOut],
    contribution_indices: list[float],
    mode: str,
    fairness_notes: str = "",
) -> TeamEvaluateResponse:
    timelines = _merge_timelines(req, contribution_indices)
    suspected, risks, sigs = _free_rider_analysis(req, contribution_indices, timelines)

    enriched: list[MemberOut] = []
    for i, m in enumerate(base):
        enriched.append(
            m.model_copy(
                update={
                    "timeline": timelines[i] if i < len(timelines) else [],
                    "free_rider_suspected": suspected[i],
                    "free_rider_risk": risks[i],
                    "free_rider_signals": sigs[i],
                }
            )
        )

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            fbs = _openai_feedbacks(req, enriched, key)
            enriched = [
                mm.model_copy(update={"ai_feedback": fbs[j]})
                for j, mm in enumerate(enriched)
            ]
        except Exception:
            enriched = [mm.model_copy(update={"ai_feedback": _template_feedback(mm)}) for mm in enriched]
    else:
        enriched = [mm.model_copy(update={"ai_feedback": _template_feedback(mm)}) for mm in enriched]

    n_sus = sum(1 for m in enriched if m.free_rider_suspected)
    fr_summary = (
        f"자동 탐지: 무임승차 의심 플래그 {n_sus}명. "
        "표시는 통계·키워드 기반이며, 최종 판단은 교수·팀 면담으로 하세요."
    )

    out_members, net, mm, anom, co_summary, adv_mode = _attach_advanced(req, enriched, suspected)

    return TeamEvaluateResponse(
        mode=mode,
        members=out_members,
        fairness_notes=fairness_notes,
        free_rider_summary=fr_summary,
        collaboration_network=net,
        contribution_outcome_summary=co_summary,
        mismatches=mm,
        anomaly_alerts=anom,
        advanced_mode=adv_mode,
    )


def _heuristic(req: TeamEvaluateRequest) -> TeamEvaluateResponse:
    members = req.members
    n = len(members)

    commits = [float(m.commits or 0) for m in members]
    tasks = [float(m.tasks_completed or 0) for m in members]
    lines = [float(m.lines_changed or 0) for m in members]
    prs = [float(m.pull_requests or 0) for m in members]
    meet = [float(m.meetings_attended or 0) for m in members]
    words = [max(0, len((m.self_report or "").split())) for m in members]

    nc, nt, nl, npr, nw, nmeet = (
        _norm(commits),
        _norm(tasks),
        _norm(lines),
        _norm(prs),
        _norm([float(w) for w in words]),
        _norm(meet),
    )

    raw_scores: list[float] = []
    for i in range(n):
        blend = 0.22 * nc[i] + 0.18 * nt[i] + 0.18 * nl[i] + 0.12 * npr[i] + 0.30 * nw[i]
        raw_scores.append(blend)
    raw_sum = sum(raw_scores) or 1.0
    contribution_indices: list[float] = []

    out: list[MemberOut] = []
    for i, m in enumerate(members):
        idx = 100.0 * (raw_scores[i] / raw_sum)
        contribution_indices.append(idx)
        q = nc[i] + nl[i] + npr[i] + 1e-6
        tech = min(100.0, 30.0 + 70.0 * (0.45 * nc[i] + 0.35 * nl[i] + 0.20 * npr[i]) / q)
        collab = min(100.0, 35.0 + 45.0 * nw[i] + 20.0 * nmeet[i])
        init_ = min(100.0, 32.0 + 40.0 * nt[i] + 28.0 * npr[i])
        out.append(
            MemberOut(
                name=m.name,
                role=m.role,
                contribution_index=round(idx, 1),
                dimensions=DimensionScores(
                    technical=round(tech, 1),
                    collaboration=round(collab, 1),
                    initiative=round(init_, 1),
                ),
                evidence_summary="정량·자기서술 가중 합성입니다. 문서·리뷰 기여는 수치에 약합니다.",
                caveats="OpenAI 키가 없어 휴리스틱만 사용했습니다.",
            )
        )

    note = "정량 지표는 참고용입니다."
    if n > 1 and commits and max(commits) > 0 and max(commits) / (sum(commits) / n) > 2.5:
        note += " 커밋 편차가 큽니다. 리뷰·기획 기여를 확인하세요."

    return _finalize_members(req, out, contribution_indices, "heuristic", note)


def _openai_eval(req: TeamEvaluateRequest, api_key: str) -> TeamEvaluateResponse:
    from openai import OpenAI

    payload = {
        "project_name": req.project_name,
        "project_description": req.project_description,
        "evaluation_criteria": req.evaluation_criteria,
        "members": [m.model_dump() for m in req.members],
    }
    sys = """교육용 팀 기여도 조교입니다. JSON만 출력하세요.
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

    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.25,
    )
    data = _parse_json(res.choices[0].message.content or "{}")
    rows = {str(r.get("name", "")).strip(): r for r in (data.get("members") or []) if isinstance(r, dict)}
    out: list[MemberOut] = []
    contribution_indices: list[float] = []
    for m in req.members:
        raw = rows.get(m.name.strip()) or rows.get(m.name)
        if raw is None and len(data.get("members") or []) == len(req.members):
            raw = (data.get("members") or [])[len(out)]
        if not isinstance(raw, dict):
            return _heuristic(req)
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
        return _heuristic(req)

    fn = str(data.get("fairness_notes", ""))[:2000]
    return _finalize_members(req, out, contribution_indices, "ai", fn)


@router.post("/evaluate", response_model=TeamEvaluateResponse)
async def evaluate_team(body: TeamEvaluateRequest) -> TeamEvaluateResponse:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            return _openai_eval(body, key)
        except Exception:
            return _heuristic(body)
    return _heuristic(body)
