"""수상·기획 강화: 루브릭 4축, 평가 신뢰도, 팀 리스크, 설명→개선→예상 흐름 (규칙 기반)."""

from __future__ import annotations

import math
from typing import Any

from edu_tools.team import (
    EvaluationTrustBlock,
    ImprovementChainBlock,
    ImprovementChainItem,
    MemberOut,
    RubricMemberRow,
    RubricReport,
    TeamEvaluateRequest,
    TeamRiskBlock,
)
from edu_tools.team_advanced import NetworkGraph


def _gini(values: list[float]) -> float:
    v = sorted(float(x) for x in values if x >= 0)
    n = len(v)
    if n < 2:
        return 0.0
    s = sum(v)
    if s <= 1e-9:
        return 0.0
    num = sum((i + 1) * x for i, x in enumerate(v))
    return max(0.0, min(1.0, (2 * num) / (n * s) - (n + 1) / n))


def _data_sufficiency_score(req: TeamEvaluateRequest) -> float:
    """입력 필드가 얼마나 채워졌는지 0–100."""
    if not req.members:
        return 0.0
    filled = 0
    total = 0
    for m in req.members:
        total += 9
        if (m.commits or 0) > 0:
            filled += 1
        if (m.pull_requests or 0) > 0:
            filled += 1
        if (m.lines_changed or 0) > 0:
            filled += 1
        if (m.tasks_completed or 0) > 0:
            filled += 1
        if (m.meetings_attended or 0) > 0:
            filled += 1
        if (m.self_report or "").strip():
            filled += 2
        if (m.peer_notes or "").strip():
            filled += 1
        if m.timeline and len([t for t in m.timeline if (t.activity_score or 0) > 0]) >= 2:
            filled += 2
    return min(100.0, 100.0 * filled / max(total, 1))


def _edge_concentration_flag(net: NetworkGraph) -> tuple[float, str | None]:
    if not net.nodes or not net.edges:
        return 0.0, None
    idx = {nd.id: i for i, nd in enumerate(net.nodes)}
    wsum = [0.0] * len(net.nodes)
    for e in net.edges:
        if e.source in idx:
            wsum[idx[e.source]] += float(e.weight)
        if e.target in idx:
            wsum[idx[e.target]] += float(e.weight)
    total = sum(wsum) or 1.0
    mx = max(wsum) if wsum else 0.0
    share = mx / total
    if len(net.nodes) >= 3 and share >= 0.52:
        return share, "협업 네트워크 가중치가 특정 인원에게 집중되어 의존도가 높을 수 있습니다."
    return share, None


def build_rubric_report(members: list[MemberOut]) -> RubricReport:
    rows: list[RubricMemberRow] = []
    for m in members:
        rm = m.freerider_detection.rule_metrics
        dt = m.dimensions
        persistence = float(rm.consistency_score) if rm else 50.0
        problem_solving = (float(dt.technical) + float(dt.initiative)) / 2.0
        rows.append(
            RubricMemberRow(
                member_name=m.name,
                contribution=min(100.0, max(0.0, float(m.contribution_index))),
                collaboration=min(100.0, max(0.0, float(dt.collaboration))),
                persistence=min(100.0, max(0.0, persistence)),
                problem_solving=min(100.0, max(0.0, problem_solving)),
            )
        )
    if not rows:
        return RubricReport()

    def avg(getter: Any) -> float:
        return round(sum(getter(r) for r in rows) / len(rows), 1)

    team = RubricMemberRow(
        member_name="(팀 평균)",
        contribution=avg(lambda r: r.contribution),
        collaboration=avg(lambda r: r.collaboration),
        persistence=avg(lambda r: r.persistence),
        problem_solving=avg(lambda r: r.problem_solving),
    )
    note = (
        "4축은 모두 규칙·데이터 기반입니다. "
        "기여도=기여 지수, 협업=협업 차원, 지속성=시간 패턴(Rule 일관성), "
        "문제 해결=(기술+주도)/2."
    )
    return RubricReport(team_average=team, members=rows, criteria_note=note, ai_explanation="")


def build_evaluation_trust(
    req: TeamEvaluateRequest,
    members: list[MemberOut],
    net: NetworkGraph,
) -> EvaluationTrustBlock:
    data = _data_sufficiency_score(req)
    cis = [m.contribution_index for m in members]
    spread = (max(cis) - min(cis)) if cis else 0.0
    consistency = max(0.0, min(100.0, 100.0 - spread * 0.75))
    n_e = len(net.edges)
    n_m = max(len(members), 1)
    collab_data = min(100.0, 25.0 + n_e * 6.0 + (30.0 if n_e >= n_m else 0.0))

    score = 0.35 * data + 0.35 * consistency + 0.30 * collab_data
    score = round(min(100.0, max(0.0, score)), 1)
    if score >= 72:
        level = "높음"
    elif score >= 48:
        level = "중간"
    else:
        level = "낮음"

    factors = [
        f"데이터 충분성(입력·타임라인): 약 {data:.0f}/100",
        f"팀 내 기여 분산 일관성: 약 {consistency:.0f}/100",
        f"협업 간선·네트워크 정보: 약 {collab_data:.0f}/100",
    ]
    short = f"종합 {score:.0f}점 — 입력이 풍부하고 격차가 과도하지 않을수록 신뢰도가 올라갑니다."
    return EvaluationTrustBlock(
        score_0_100=score,
        level_ko=level,
        factors=factors,
        short_note=short,
    )


def build_team_risk(net: NetworkGraph, members: list[MemberOut]) -> TeamRiskBlock:
    flags: list[str] = []
    cis = [m.contribution_index for m in members]
    g = _gini(cis)
    if g >= 0.38 and len(cis) >= 3:
        flags.append("팀 내 기여도 격차가 커 편중 리스크가 있습니다.")

    _share, dep_msg = _edge_concentration_flag(net)
    if dep_msg:
        flags.append(dep_msg)

    avg_collab = sum(m.dimensions.collaboration for m in members) / max(len(members), 1)
    if avg_collab < 38 and len(members) >= 2:
        flags.append("팀 평균 협업 차원이 낮아 협업 부족 징후가 있습니다.")

    n_low = sum(1 for m in members if m.freerider_detection.rule_metrics.risk_level == "suspected")
    if n_low >= max(1, len(members) // 2):
        flags.append("무임승차 Rule 위험 인원이 다수입니다. 팀 전체 면담을 권장합니다.")

    summary = " ".join(flags) if flags else "팀 단위 특이 리스크 신호가 크지 않습니다(참고)."
    return TeamRiskBlock(flags=flags, summary_ko=summary, ai_team_risk="")


def build_improvement_chain(
    members: list[MemberOut],
    team_risk: TeamRiskBlock,
    trust: EvaluationTrustBlock,
) -> ImprovementChainBlock:
    items: list[ImprovementChainItem] = []

    if any("편중" in f or "집중" in f for f in team_risk.flags):
        items.append(
            ImprovementChainItem(
                problem="작업·협업 편중",
                explanation="기여 지수 또는 네트워크 가중치가 일부 인원에 쏠려 있습니다.",
                suggestion="역할 분배·페어 프로그래밍·주간 공유 회의를 도입해 부담을 분산합니다.",
                predicted_outcome="협업·지속성 루브릭 점수와 팀 신뢰도 지표가 함께 개선될 수 있습니다.",
            )
        )

    low_collab = [m for m in members if m.dimensions.collaboration < 40]
    if low_collab:
        items.append(
            ImprovementChainItem(
                problem="협업 참여 부족",
                explanation="일부 인원의 협업 차원·리뷰·회의 입력이 낮습니다.",
                suggestion="코드 리뷰·이슈 코멘트·짧은 동기화를 주 1회 이상 의무화합니다.",
                predicted_outcome="협업 참여 루브릭·네트워크 간선이 늘어날 것으로 기대됩니다.",
            )
        )

    if trust.level_ko == "낮음":
        items.append(
            ImprovementChainItem(
                problem="입력 데이터가 부족해 평가 신뢰도가 낮음",
                explanation="커밋·PR·주차별 활동·동료 메모 등이 비어 있는 항목이 있습니다.",
                suggestion="레포 링크·회의록·태스크 보드를 입력하고 주차별 활동을 채웁니다.",
                predicted_outcome="신뢰도 점수가 상승하고 설명 가능한 평가가 강화됩니다.",
            )
        )

    if not items:
        items.append(
            ImprovementChainItem(
                problem="지속적 개선",
                explanation="현재 리스크 신호가 제한적입니다.",
                suggestion="스프린트마다 기여·협업 지표를 팀과 공유하고 루브릭에 맞춰 자기 점검합니다.",
                predicted_outcome="팀 건강도·일관성 유지에 도움이 됩니다.",
            )
        )

    return ImprovementChainBlock(
        headline="문제 → 이유 → 개선 → 기대 효과 (규칙 기반 제안)",
        items=items[:5],
    )


def build_contest_layers(
    req: TeamEvaluateRequest,
    members: list[MemberOut],
    net: NetworkGraph,
) -> tuple[RubricReport, EvaluationTrustBlock, TeamRiskBlock, ImprovementChainBlock]:
    rubric = build_rubric_report(members)
    trust = build_evaluation_trust(req, members, net)
    risk = build_team_risk(net, members)
    chain = build_improvement_chain(members, risk, trust)
    return rubric, trust, risk, chain
