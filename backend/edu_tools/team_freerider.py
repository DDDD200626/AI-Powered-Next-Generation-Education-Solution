"""무임승차 자동 탐지 — Rule(4지표) + AI 요약 + 팀 대시보드."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from learning_analysis.llm_clients import get_openai_client

from edu_tools.team import (
    FreeriderDetectionReport,
    FreeriderRuleMetrics,
    MemberDashboardRow,
    MemberOut,
    TeamEvaluateRequest,
    TimelinePointOut,
)
from edu_tools.team_advanced import NetworkEdge, NetworkGraph

PRODUCT_TAGLINE_KO = (
    "본 시스템은 기여도 계산은 데이터 기반으로 수행하고,\n"
    "AI를 활용하여 협업 패턴을 해석하고 평가 결과를 설명하며\n"
    "개선 방향까지 제시합니다."
)


def _explicit_user_edges(req: TeamEvaluateRequest) -> bool:
    return bool(req.collaboration_edges and len(req.collaboration_edges) > 0)


def _undirected_degree(name: str, edges: list[NetworkEdge]) -> int:
    nbr: set[str] = set()
    for e in edges:
        if e.source == name:
            nbr.add(e.target)
        if e.target == name:
            nbr.add(e.source)
    return len(nbr)


def _activity_composite(mi: Any) -> float:
    """커밋·라인·문서(단어)·태스크 가중 합성 활동량."""
    c = float(mi.commits or 0)
    lines = float(mi.lines_changed or 0)
    words = len((mi.self_report or "").split())
    tasks = float(mi.tasks_completed or 0)
    return c * 1.0 + lines / 50.0 + words * 0.35 + tasks * 0.4


def _collaboration_composite(mi: Any, degree: int) -> float:
    """PR·회의·동료 메모·네트워크 연결을 반영한 협업량."""
    pr = float(mi.pull_requests or 0)
    meet = float(mi.meetings_attended or 0)
    peer = len((mi.peer_notes or "").split())
    return pr * 2.5 + meet * 1.2 + peer * 0.12 + min(degree, 10) * 1.8


def _cramming_ratio(tl: list[TimelinePointOut]) -> float:
    """후반 20% 구간에 몰린 상대 비중(몰아치기)."""
    if len(tl) < 2:
        return 0.35
    shares = [float(p.share_percent) for p in tl]
    total = sum(shares) or 1.0
    k = max(1, int(math.ceil(0.2 * len(tl))))
    last_sum = sum(shares[-k:])
    return last_sum / total


def _consistency_score_from_cram(cr: float) -> float:
    """꾸준함 높음 ~80–100, 몰아치기 ~30–50."""
    return max(25.0, min(100.0, 100.0 - min(72.0, max(0.0, cr - 0.20) * 105.0)))


def _grade_ko(score: float) -> str:
    if score >= 80:
        return "핵심 기여"
    if score >= 50:
        return "일반"
    if score >= 30:
        return "낮음"
    return "⚠️ 무임승차 의심"


def _risk_level_from_count(k: int) -> str:
    if k >= 3:
        return "suspected"
    if k == 2:
        return "caution"
    return "normal"


def _basic_layer(mi: Any) -> tuple[bool, list[str]]:
    commits = int(mi.commits or 0)
    words = len((mi.self_report or "").split())
    pr = int(mi.pull_requests or 0)
    meet = int(mi.meetings_attended or 0)
    commits_low = commits <= 2
    doc_low = words < 22
    collab_low = (pr + meet) == 0
    low = commits_low and doc_low and collab_low
    reasons: list[str] = []
    if low:
        reasons = [
            "커밋 수 거의 없음",
            "문서·자기보고 기여 없음",
            "협업·리뷰·회의 참여 없음",
        ]
    else:
        if commits_low:
            reasons.append("커밋 수가 상대적으로 매우 적음")
        if doc_low:
            reasons.append("문서·자기보고 분량이 매우 적음")
        if collab_low:
            reasons.append("PR·회의 참여 입력이 없음")
    return low, reasons


def _advanced_patterns(
    req: TeamEvaluateRequest,
    mi: Any,
    tl: list[TimelinePointOut],
    name: str,
    edges: list[NetworkEdge],
    explicit_edges: bool,
) -> tuple[list[str], float]:
    flags: list[str] = []
    score = 0.0
    if len(tl) >= 2:
        shares = [float(p.share_percent) for p in tl]
        if shares[-1] - shares[0] >= 20.0 and shares[-1] >= max(shares) * 0.72:
            flags.append("후반·마지막 구간에 활동이 몰림(마지막 날 몰아치기 의심)")
            score += 28.0
        if len(shares) >= 3:
            mx = max(shares)
            if mx >= 52.0:
                hi = sum(1 for s in shares if s == mx)
                rest = [s for s in shares if s != mx]
                rest_avg = sum(rest) / len(rest) if rest else 0.0
                if hi == 1 and mx > rest_avg + 22.0:
                    flags.append("특정 기간에만 상대 기여가 집중됨")
                    score += 22.0
    pr = int(mi.pull_requests or 0)
    meet = int(mi.meetings_attended or 0)
    peer = (mi.peer_notes or "").strip()
    if pr == 0 and meet <= 1 and len(peer) < 8:
        flags.append("리뷰·회의·동료 메모 등 상호작용 흔적이 입력상 거의 없음")
        score += 18.0
    if explicit_edges and len(req.members) >= 3:
        deg = _undirected_degree(name, edges)
        if deg == 1:
            flags.append("협업 네트워크에서 연결이 매우 약함(간선 1개, 입력 기준)")
            score += 20.0
    return flags, min(100.0, score)


def _isolation_layer(
    mi: Any,
    member: MemberOut,
    name: str,
    edges: list[NetworkEdge],
    explicit_edges: bool,
) -> tuple[bool, list[str]]:
    if explicit_edges:
        deg = _undirected_degree(name, edges)
        if deg == 0:
            return True, [
                "입력한 협업 네트워크에서 다른 팀원과 연결된 간선이 없음",
                "다른 사람 코드·리뷰·피드백 상호작용이 없는 것으로 보임(입력 기준)",
            ]
    collab = float(member.dimensions.collaboration)
    pr = int(mi.pull_requests or 0)
    meet = int(mi.meetings_attended or 0)
    if not explicit_edges and collab < 28.0 and pr == 0 and meet <= 1:
        return True, [
            "협업 차원 점수가 낮고 PR·회의 입력이 거의 없어 ‘혼자만 작업’ 패턴으로 추정",
        ]
    return False, []


def _vs_team_layer(ci: float, team_mean: float) -> tuple[bool, float]:
    delta = round(ci - team_mean, 2)
    below = team_mean > 0 and ci < team_mean * 0.90
    return below, delta


def _build_rule_metrics(
    *,
    n_team: int,
    a_i: float,
    a_bar: float,
    c_i: float,
    c_bar: float,
    tl: list[TimelinePointOut],
    interaction_risk: bool,
) -> FreeriderRuleMetrics:
    """4지표 위험 + 혼합 점수 + 등급."""
    activity_score = min(100.0, (a_i / max(a_bar, 1e-6)) * 100.0) if n_team >= 1 else 0.0
    collaboration_score = min(100.0, (c_i / max(c_bar, 1e-6)) * 100.0) if n_team >= 1 else 0.0

    cr = _cramming_ratio(tl)
    consistency_score = _consistency_score_from_cram(cr)

    if n_team >= 2 and a_bar > 1e-9:
        activity_risk = a_i <= 0.30 * a_bar
    else:
        activity_risk = False

    if n_team >= 2 and c_bar > 1e-9:
        collaboration_risk = c_i <= 0.30 * c_bar
    else:
        collaboration_risk = False

    # 마지막 20% 기간에 작업의 약 80% 이상 몰림 → 위험
    consistency_risk = len(tl) >= 2 and cr >= 0.80

    blended = (
        activity_score * 0.4 + collaboration_score * 0.3 + consistency_score * 0.3
    )
    rule_conditions_met = sum(
        [activity_risk, collaboration_risk, consistency_risk, interaction_risk]
    )
    # 3개 이상 위험 → 무임승차 의심(페널티), 2개 주의, 1개 이하 정상
    suspected_by_rules = rule_conditions_met >= 3
    final_score = round(blended * 0.6, 1) if suspected_by_rules else round(blended, 1)
    grade = _grade_ko(final_score)
    risk_level = _risk_level_from_count(rule_conditions_met)

    lines: list[str] = []
    if activity_risk:
        lines.append(f"활동량: 팀 평균 대비 매우 낮음(합성 지표 {a_i:.1f} / 팀평균 {a_bar:.1f})")
    if collaboration_risk:
        lines.append(f"협업 참여: 팀 평균 대비 매우 낮음(합성 {c_i:.1f} / 팀평균 {c_bar:.1f})")
    if consistency_risk:
        lines.append(
            f"시간 패턴: 마지막 20% 구간에 전체의 약 {cr * 100:.0f}% 집중(80% 이상 몰아치기 기준 충족)"
        )
    if interaction_risk:
        lines.append("상호작용: 협업 네트워크에서 고립 또는 연결 매우 약함")
    if not lines:
        lines.append("Rule 4지표상 특별한 위험 신호가 적습니다(참고).")

    return FreeriderRuleMetrics(
        activity_score=round(activity_score, 1),
        collaboration_score=round(collaboration_score, 1),
        consistency_score=round(consistency_score, 1),
        blended_score=round(blended, 1),
        final_score=final_score,
        penalty_applied=suspected_by_rules,
        activity_risk=activity_risk,
        collaboration_risk=collaboration_risk,
        consistency_risk=consistency_risk,
        interaction_risk=interaction_risk,
        rule_conditions_met=rule_conditions_met,
        risk_level=risk_level,
        grade_ko=grade,
        analysis_lines=lines[:8],
    )


def compute_freerider_reports(
    req: TeamEvaluateRequest,
    members: list[MemberOut],
    timelines: list[list[TimelinePointOut]],
    net: NetworkGraph,
) -> list[FreeriderDetectionReport]:
    """팀원별 탐지 + Rule 혼합 점수."""
    n = len(members)
    cis = [m.contribution_index for m in members]
    team_mean = sum(cis) / n if n else 0.0
    explicit = _explicit_user_edges(req)
    edges = net.edges

    activities: list[float] = []
    collabs: list[float] = []
    for i in range(n):
        mi = req.members[i] if i < len(req.members) else req.members[-1]
        nm = members[i].name.strip()
        deg = _undirected_degree(nm, edges) if explicit else 0
        activities.append(_activity_composite(mi))
        collabs.append(_collaboration_composite(mi, deg))

    a_bar = sum(activities) / n if n else 0.0
    c_bar = sum(collabs) / n if n else 0.0

    out: list[FreeriderDetectionReport] = []
    for i, mem in enumerate(members):
        mi = req.members[i] if i < len(req.members) else req.members[-1]
        name = mem.name.strip()
        tl = timelines[i] if i < len(timelines) else []
        nm = name
        deg = _undirected_degree(nm, edges) if explicit else 0

        low, basic_rs = _basic_layer(mi)
        adv_flags, adv_score = _advanced_patterns(req, mi, tl, nm, edges, explicit)
        iso, iso_rs = _isolation_layer(mi, mem, nm, edges, explicit)
        below, delta = _vs_team_layer(mem.contribution_index, team_mean)

        interaction_risk = iso or (explicit and deg == 0)

        rm = _build_rule_metrics(
            n_team=n,
            a_i=activities[i],
            a_bar=a_bar,
            c_i=collabs[i],
            c_bar=c_bar,
            tl=tl,
            interaction_risk=interaction_risk,
        )

        out.append(
            FreeriderDetectionReport(
                basic_low_contribution=low,
                basic_reasons=basic_rs,
                advanced_pattern_flags=adv_flags,
                advanced_pattern_score=round(adv_score, 1),
                collaboration_isolated=iso,
                collaboration_isolation_reasons=iso_rs,
                below_team_average=below,
                team_mean_contribution=round(team_mean, 2),
                delta_vs_team_mean=delta,
                ai_detection_summary="",
                rule_metrics=rm,
            )
        )
    return out


def build_team_dashboard(members: list[MemberOut]) -> list[MemberDashboardRow]:
    """최종 Rule 점수 기준 순위·막대 비율."""
    if not members:
        return []
    scored = [(m.freerider_detection.rule_metrics.final_score, m) for m in members]
    mx = max(s for s, _ in scored) or 1.0
    sorted_m = sorted(scored, key=lambda x: x[0], reverse=True)
    rows: list[MemberDashboardRow] = []
    for rank, (fs, m) in enumerate(sorted_m, start=1):
        rm = m.freerider_detection.rule_metrics
        bar = min(100.0, (fs / mx) * 100.0)
        rows.append(
            MemberDashboardRow(
                member_name=m.name,
                rank=rank,
                final_rule_score=round(fs, 1),
                bar_fill_percent=round(bar, 1),
                grade_ko=rm.grade_ko,
                rule_conditions_met=rm.rule_conditions_met,
                risk_level=rm.risk_level,
                suspected_highlight=(
                    rm.risk_level == "suspected" or rm.final_score <= 30 or fs <= 30
                ),
            )
        )
    return rows


def freerider_detection_overview(members: list[MemberOut]) -> str:
    if not members:
        return ""
    n_b = sum(1 for m in members if m.freerider_detection.basic_low_contribution)
    n_adv = sum(1 for m in members if m.freerider_detection.advanced_pattern_flags)
    n_iso = sum(1 for m in members if m.freerider_detection.collaboration_isolated)
    n_below = sum(1 for m in members if m.freerider_detection.below_team_average)
    n_sus = sum(1 for m in members if m.freerider_detection.rule_metrics.risk_level == "suspected")
    n_cau = sum(1 for m in members if m.freerider_detection.rule_metrics.risk_level == "caution")
    return (
        f"4지표(활동·협업·시간·상호작용) 중 3개 이상 위험 → 의심 {n_sus}명, 2개 주의 {n_cau}명. "
        f"보조 신호: Low {n_b}명 · 패턴 {n_adv}명 · 고립 {n_iso}명 · 팀평균↓ {n_below}명. "
        "면담·로그로 확인하세요."
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


def freerider_ai_summaries(req: TeamEvaluateRequest, members: list[MemberOut], api_key: str) -> list[str]:
    payload = {
        "project": req.project_name,
        "tagline": PRODUCT_TAGLINE_KO,
        "members": [
            {
                "name": m.name,
                "contribution_index": m.contribution_index,
                "rule": {
                    "activity": m.freerider_detection.rule_metrics.activity_score,
                    "collaboration": m.freerider_detection.rule_metrics.collaboration_score,
                    "consistency": m.freerider_detection.rule_metrics.consistency_score,
                    "final_score": m.freerider_detection.rule_metrics.final_score,
                    "grade": m.freerider_detection.rule_metrics.grade_ko,
                    "risk_level": m.freerider_detection.rule_metrics.risk_level,
                    "conditions_met": m.freerider_detection.rule_metrics.rule_conditions_met,
                    "analysis_lines": m.freerider_detection.rule_metrics.analysis_lines[:6],
                },
            }
            for m in members
        ],
    }
    sys = """교육용 팀 분석가입니다. 중요: 탐지 여부·무임승차 Rule 점수·등급은 이미 코드(서버)로 확정되었습니다.
당신은 점수를 매기거나 바꾸지 말고, 위에 주어진 rule 지표와 analysis_lines만 근거로
교육자용 ‘설명·해석’만 한국어로 작성하세요. JSON만 출력.
{"summaries": [{"name": "이름", "text": "2~4문장. 협업 부족·후반 집중 등 패턴을 설명. 비난 금지, 면담·로그 확인 권장."}]}
모든 팀원에 대해 summaries를 채우세요."""

    client = get_openai_client(api_key)
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    data = _parse_json(res.choices[0].message.content or "{}")
    by_name = {
        str(x.get("name", "")).strip(): str(x.get("text", "")).strip()
        for x in (data.get("summaries") or [])
        if isinstance(x, dict)
    }
    out: list[str] = []
    for m in members:
        t = by_name.get(m.name.strip()) or by_name.get(m.name) or ""
        out.append(t[:1200])
    return out


def safe_freerider_ai_summaries(req: TeamEvaluateRequest, members: list[MemberOut], api_key: str) -> list[str]:
    try:
        return freerider_ai_summaries(req, members, api_key)
    except Exception:
        return ["" for _ in members]
