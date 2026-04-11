"""팀 프로젝트 기여도 평가 — 무임승차 의심, 타임라인, 팀원별 AI 피드백."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from learning_analysis.llm_clients import get_openai_client

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

MAX_TEAM_MEMBERS = 40


class TimelinePointIn(BaseModel):
    period_label: str = Field(..., min_length=1)
    activity_score: float = Field(..., ge=0, le=100, description="해당 기간 활동·기여 점수(주관)")


class MemberIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    role: str = Field("", max_length=200)
    commits: int | None = Field(None, ge=0)
    pull_requests: int | None = Field(None, ge=0)
    lines_changed: int | None = Field(None, ge=0)
    tasks_completed: int | None = Field(None, ge=0)
    meetings_attended: int | None = Field(None, ge=0)
    self_report: str = Field("", max_length=12000)
    peer_notes: str = Field("", max_length=6000)
    timeline: list[TimelinePointIn] = Field(default_factory=list)
    outcome_score: float | None = Field(
        None,
        ge=0,
        le=100,
        description="프로젝트/발표/동료 평가 등 결과 점수(선택, 기여 추정과 비교)",
    )
    instructor_freerider_override: Literal["none", "clear_suspicion", "confirm_suspicion"] = Field(
        "none",
        description="교육자: 자동 무임승차 의심 플래그를 화면 기준으로 덮어씀(none=미사용).",
    )
    instructor_override_note: str = Field(
        "",
        max_length=500,
        description="오버라이드 사유(감사·면담 기록용, 선택).",
    )


class TeamEvaluateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "project_name": "2025-1 소프트웨어공학 팀프로젝트",
                "project_description": "웹 기반 협업 도구 개발",
                "evaluation_criteria": "기술 완성도, 협업, 창의성",
                "members": [
                    {
                        "name": "홍길동",
                        "role": "백엔드",
                        "commits": 24,
                        "pull_requests": 5,
                        "lines_changed": 1200,
                        "tasks_completed": 6,
                        "meetings_attended": 4,
                        "self_report": "API와 DB 설계를 담당했습니다.",
                        "peer_notes": "",
                        "timeline": [],
                    },
                    {
                        "name": "김팀원",
                        "role": "프론트엔드",
                        "commits": 18,
                        "pull_requests": 4,
                        "lines_changed": 800,
                        "tasks_completed": 5,
                        "meetings_attended": 4,
                        "self_report": "UI 구현과 테스트를 맡았습니다.",
                        "peer_notes": "",
                        "timeline": [],
                    },
                ],
                "collaboration_edges": [],
            }
        }
    )

    project_name: str = Field(..., min_length=1, max_length=300)
    project_description: str = Field("", max_length=12000)
    evaluation_criteria: str = Field("", max_length=16000)
    members: list[MemberIn] = Field(..., min_length=1)
    collaboration_edges: list[CollaborationEdgeIn] = Field(
        default_factory=list,
        description="팀원 간 상호작용(리뷰·회의·페어 등). 없으면 기여도 기반 완전 그래프로 추정",
    )
    use_deep_learning: bool = Field(
        True,
        description="False면 Rule·LLM 경로만 쓰고 PyTorch/경량 ML 보조(dl_score 등)를 붙이지 않습니다.",
    )
    deep_learning_accumulate_samples: bool = Field(
        True,
        description="False면 이번 요청을 학습 풀에 넣지 않고 추론만. use_deep_learning=True일 때만 적용.",
    )

    @model_validator(mode="after")
    def _limit_team_size(self) -> TeamEvaluateRequest:
        if len(self.members) > MAX_TEAM_MEMBERS:
            raise ValueError(f"팀원은 최대 {MAX_TEAM_MEMBERS}명까지 입력할 수 있습니다.")
        return self

    @field_validator("collaboration_edges")
    @classmethod
    def _limit_edges(cls, v: list[CollaborationEdgeIn]) -> list[CollaborationEdgeIn]:
        if len(v) > 500:
            raise ValueError("협업 간선은 최대 500개까지 입력할 수 있습니다.")
        return v


class DimensionScores(BaseModel):
    technical: float
    collaboration: float
    initiative: float


class FreeriderRuleMetrics(BaseModel):
    """Rule 기반: 활동량·협업·일관성(시간 패턴)·상호작용 → 혼합 점수·등급."""

    activity_score: float = Field(0.0, ge=0, le=100)
    collaboration_score: float = Field(0.0, ge=0, le=100)
    consistency_score: float = Field(0.0, ge=0, le=100)
    blended_score: float = Field(0.0, ge=0, le=100, description="0.4·0.3·0.3 가중")
    final_score: float = Field(0.0, ge=0, le=100, description="무임승차 의심 시 ×0.6 보정 후")
    penalty_applied: bool = False
    activity_risk: bool = False
    collaboration_risk: bool = False
    consistency_risk: bool = False
    interaction_risk: bool = False
    rule_conditions_met: int = Field(0, ge=0, le=4)
    combined_risk_count: int = Field(
        0,
        ge=0,
        le=5,
        description="Rule 4지표 충족 개수 + DL 보조 가산(최대 5)",
    )
    dl_model_freerider_support: bool = Field(
        False,
        description="DL 혼합 점수가 팀 대비 유의미하게 낮아 무임승차 판정에 가산됨",
    )
    dl_model_note: str = Field("", description="DL 근거 한 줄(있을 때만)")
    risk_level: str = Field("", description="normal | caution | suspected")
    grade_ko: str = Field("", description="핵심 기여 | 일반 | 낮음 | 무임승차 의심")
    analysis_lines: list[str] = Field(default_factory=list, description="개인 상세용 요약 불릿")


class FreeriderDetectionReport(BaseModel):
    """무임승차 자동 탐지 4단계(수상용 구조): 1차 저기여 → 2차 패턴 → 3차 고립 → 4차 팀평균."""

    basic_low_contribution: bool = Field(False, description="1차: Low Contribution")
    basic_reasons: list[str] = Field(default_factory=list)
    advanced_pattern_flags: list[str] = Field(
        default_factory=list,
        description="2차: 비정상 참여 패턴(몰아치기·기간 집중·상호작용 부족 등)",
    )
    advanced_pattern_score: float = Field(0.0, ge=0, le=100)
    collaboration_isolated: bool = Field(False, description="3차: 협업 고립 사용자")
    collaboration_isolation_reasons: list[str] = Field(default_factory=list)
    below_team_average: bool = Field(False, description="4차: 팀 평균 대비 낮음")
    team_mean_contribution: float = Field(0.0, description="팀 기여 지수 평균")
    delta_vs_team_mean: float = Field(0.0, description="본인 기여 − 팀 평균")
    ai_detection_summary: str = Field(
        "",
        description="무임승차 탐지·Rule 점수는 코드로 확정, AI는 교육자용 설명·해석만",
    )
    rule_metrics: FreeriderRuleMetrics = Field(default_factory=FreeriderRuleMetrics)


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
    freerider_detection: FreeriderDetectionReport = Field(default_factory=FreeriderDetectionReport)
    dl_score: float | None = Field(
        None,
        ge=0,
        le=100,
        description="데이터 학습 보조 점수(PyTorch MLP 또는 경량 선형). 참고용.",
    )
    dl_confidence: float | None = Field(
        None,
        ge=0,
        le=100,
        description="DL 신뢰도(불확실도·입력 분산 반영).",
    )
    dl_blended_score: float | None = Field(
        None,
        ge=0,
        le=100,
        description="통합 엔진: rule_w·normalizedScore + dl_w·dl_score (dl_w는 신뢰도·불확실도·드리프트·검증 MAE로 동적).",
    )
    dl_top_factors: list[str] = Field(
        default_factory=list,
        description="DL 보조 점수에 영향 큰 요인(통합 피처 기준)",
    )
    instructor_freerider_override: str = Field(
        "none",
        description="요청에 담긴 교육자 무임승차 오버라이드(none|clear_suspicion|confirm_suspicion).",
    )
    instructor_override_note: str = Field("", description="요청에 담긴 오버라이드 사유 에코.")
    freerider_override_effect_ko: str = Field(
        "",
        description="오버라이드가 플래그·위험도에 미친 영향 한 줄(없으면 빈 문자열).",
    )


class MemberExplainFact(BaseModel):
    """규칙 기반 설명 카드 — 점수의 ‘왜’를 면담·검수용으로 제시."""

    member_name: str
    facts: list[str] = Field(default_factory=list, max_length=5)


class TeamRoleBalanceOut(BaseModel):
    """팀 전체 평균 역할 4유형 비중(0–100 스케일, 팀원 평균)."""

    dev: float = 0.0
    doc: float = 0.0
    leader: float = 0.0
    supporter: float = 0.0
    balance_hint: str = ""


class ReflectionKit(BaseModel):
    """교육자용 성찰·면담 키트(징계 문구 아님)."""

    team_storyline: str = ""
    teacher_questions: list[str] = Field(default_factory=list, max_length=12)
    encouragement_line: str = ""


class CreativeInsights(BaseModel):
    """창의성 모듈: 설명 가능성 + 팀 내러티브 + 면담 질문."""

    explain_facts: list[MemberExplainFact] = Field(default_factory=list)
    team_role_balance: TeamRoleBalanceOut = Field(default_factory=TeamRoleBalanceOut)
    reflection_kit: ReflectionKit = Field(default_factory=ReflectionKit)
    team_health_score: float = Field(
        0.0,
        ge=0,
        le=100,
        description="팀 협업·기여 균형 추정 지수(높을수록 상대적으로 안정, 참고용)",
    )
    team_health_hint: str = Field(
        "",
        description="건강도 점수에 대한 한 줄 해설(교육·면담 참고)",
    )


class PracticalToolkit(BaseModel):
    """기획·실무: 교육자 검수·운영 체크리스트."""

    teacher_checklist: list[str] = Field(
        default_factory=list,
        description="조교·교수가 결과 활용 전 확인할 수 있는 항목",
    )
    checklist_note: str = "체크리스트는 참고용이며, 기관 규정·개인정보 보호를 우선합니다."


class ModelEvalBundle(BaseModel):
    """단일 LLM(최신 ChatGPT·Gemini·Claude·Grok 등)별 팀 평가 결과."""

    key: str = Field(..., description="openai | gemini | claude | grok")
    label: str = Field(..., description="실제 모델명(환경변수 기준)")
    ok: bool
    error: str | None = None
    result: TeamEvaluateResponse | None = None


class MemberContributionCompare(BaseModel):
    member_name: str
    contribution_index_by_model: dict[str, float] = Field(
        ...,
        description="모델 key → 기여 지수(0–100)",
    )
    spread: float = Field(..., description="해당 팀원에 대해 모델 간 최대-최소")
    mean: float = Field(..., description="모델 간 평균 기여 지수")
    dimension_spread: dict[str, float] = Field(
        default_factory=dict,
        description="팀원별 technical·collaboration·initiative 모델 간 편차(최대−최소)",
    )


class DivergenceAxisSummary(BaseModel):
    """모델 간 의견이 갈리는 축(기여 지수·3차원)."""

    axis_id: str = Field(..., description="contribution_index | technical | collaboration | initiative")
    axis_label_ko: str
    mean_spread: float = Field(..., description="팀원 전체에 대한 평균 편차")
    max_spread_member: str = Field("", description="이 축에서 편차가 가장 큰 팀원")
    interpretation: str = Field("", description="한 줄 해석")


class OpinionDivergenceAnalysis(BaseModel):
    """핵심1: AI 간 의견 차이 — 어떤 기준(축)에서 갈렸는지."""

    primary_axes: list[DivergenceAxisSummary] = Field(default_factory=list)
    criteria_segments: list[str] = Field(
        default_factory=list,
        description="입력 평가 기준을 잘라낸 문구(비교·설명용)",
    )
    criteria_keyword_overlap_note: str = Field(
        "",
        description="루브릭 키워드가 모델 근거 문장에 얼마나 등장했는지 요약",
    )
    narrative: str = Field("", description="종합 내러티브")


class TrustScoreBlock(BaseModel):
    """핵심2: 신뢰도 — 일관성·루브릭 일치·설명 품질(0–100)."""

    consistency_0_100: float = Field(0, ge=0, le=100, description="모델 간 수치 일관성")
    rubric_alignment_0_100: float = Field(0, ge=0, le=100, description="입력 평가 기준과 근거 문장의 키워드 정합")
    explanation_quality_0_100: float = Field(0, ge=0, le=100, description="근거·주의 문장의 구체성·길이")
    overall_trust_0_100: float = Field(0, ge=0, le=100, description="가중 평균 종합")
    notes: list[str] = Field(default_factory=list, description="해석용 짧은 문장")


class ExplainabilityEntry(BaseModel):
    """핵심3: 설명 가능한 AI — 각 모델이 왜 그렇게 평가했는지."""

    model_key: str
    model_label: str
    member_name: str
    contribution_index: float
    technical: float
    collaboration: float
    initiative: float
    evidence_summary: str = Field("", description="모델이 제시한 근거")
    caveats: str = Field("", description="모델이 제시한 주의·한계")
    why_one_liner: str = Field(
        "",
        description="차원 점수+근거 앞부분을 묶은 한 줄 요약",
    )


class EvaluationPipelineStep(BaseModel):
    """AI 멀티평가 처리 순서(STEP 1–6)."""

    step: int = Field(..., ge=1, le=6)
    title_ko: str = Field(..., description="단계 제목(고정 라벨)")
    status: str = Field(
        ...,
        description="completed | skipped | partial",
    )
    detail: str = Field("", description="이번 요청에서의 실행 요약")


class TeamCompareResponse(BaseModel):
    """AI 멀티평가: 복수 모델 독립 평가 + 의견 차이·신뢰도·설명 가능성."""

    request_id: str = ""
    generated_at: str = ""
    processing_ms: float = 0.0
    product_mode: str = Field("ai_multi_eval", description="프론트·문서용 식별자")
    pipeline_steps: list[EvaluationPipelineStep] = Field(
        default_factory=list,
        description="STEP1 데이터 수집 → … → STEP6 최종 점수·피드백",
    )
    models: list[ModelEvalBundle] = Field(default_factory=list)
    member_comparison: list[MemberContributionCompare] = Field(default_factory=list)
    comparison_summary: str = ""
    divergence: OpinionDivergenceAnalysis | None = None
    trust_scores: TrustScoreBlock | None = None
    explainability: list[ExplainabilityEntry] = Field(default_factory=list)
    disclaimer: str = (
        "모델별 결과는 서로 다른 가중·해석을 가질 수 있습니다. 참고용이며 최종 판단은 교육자에게 있습니다."
    )


class MemberDashboardRow(BaseModel):
    member_name: str
    rank: int = Field(..., ge=1)
    final_rule_score: float = Field(0.0, ge=0, le=100)
    bar_fill_percent: float = Field(0.0, ge=0, le=100, description="막대 그래프 0–100")
    grade_ko: str = ""
    rule_conditions_met: int = 0
    risk_level: str = ""
    suspected_highlight: bool = False


class RubricMemberRow(BaseModel):
    """루브릭 4축(0–100) — 규칙·데이터 기반."""

    member_name: str
    contribution: float = Field(0, ge=0, le=100, description="기여도")
    collaboration: float = Field(0, ge=0, le=100, description="협업 참여")
    persistence: float = Field(0, ge=0, le=100, description="지속성·일관성")
    problem_solving: float = Field(0, ge=0, le=100, description="문제 해결·기술·주도")


class RubricReport(BaseModel):
    team_average: RubricMemberRow = Field(
        default_factory=lambda: RubricMemberRow(member_name="(팀 평균)"),
    )
    members: list[RubricMemberRow] = Field(default_factory=list)
    criteria_note: str = ""
    ai_explanation: str = Field("", description="선택: 루브릭 기준 설명(AI는 해석만)")


class EvaluationTrustBlock(BaseModel):
    """평가 결과 신뢰도 — 휴리스틱."""

    score_0_100: float = Field(0, ge=0, le=100)
    level_ko: str = Field("", description="높음 | 중간 | 낮음")
    factors: list[str] = Field(default_factory=list)
    short_note: str = ""


class TeamRiskBlock(BaseModel):
    """팀 단위 리스크."""

    flags: list[str] = Field(default_factory=list)
    summary_ko: str = ""
    ai_team_risk: str = Field("", description="선택: 팀 리스크 AI 한 문단")


class ImprovementChainItem(BaseModel):
    problem: str = ""
    explanation: str = ""
    suggestion: str = ""
    predicted_outcome: str = ""


class ImprovementChainBlock(BaseModel):
    headline: str = ""
    items: list[ImprovementChainItem] = Field(default_factory=list)


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
    creative_insights: CreativeInsights = Field(
        default_factory=CreativeInsights,
        description="규칙 기반 설명 카드·팀 역할 밸런스·면담 질문·스토리라인·팀 건강도",
    )
    practical_toolkit: PracticalToolkit = Field(
        default_factory=PracticalToolkit,
        description="교육자용 실무 체크리스트",
    )
    request_id: str = Field("", description="요청 추적용 UUID")
    generated_at: str = Field("", description="응답 생성 시각(UTC, ISO 8601)")
    processing_ms: float = Field(0.0, ge=0, description="서버 처리 시간(밀리초, 근사)")
    disclaimer: str = (
        "팀 프로젝트 기여도 자동 평가(보조) 결과입니다. 최종 성적·인사 판단을 대체하지 않습니다. "
        "무임승차·불일치·네트워크·이상 탐지는 추정이며 면담·추가 증거로 확인하세요."
    )
    freerider_detection_overview: str = Field(
        "",
        description="무임승차 4단계 탐지 팀 요약",
    )
    product_tagline_ko: str = Field(
        "",
        description="심사용: 계산=데이터·규칙, AI=해석·설명·개선 제안",
    )
    team_dashboard: list[MemberDashboardRow] = Field(
        default_factory=list,
        description="팀 대시보드: 순위·막대·등급 표시용",
    )
    rubric_report: RubricReport = Field(
        default_factory=RubricReport,
        description="루브릭 4축(기여·협업·지속성·문제해결) — 규칙 기반",
    )
    evaluation_trust: EvaluationTrustBlock = Field(
        default_factory=EvaluationTrustBlock,
        description="데이터 충분성·일관성·협업 정보 기반 신뢰도",
    )
    team_risk: TeamRiskBlock = Field(
        default_factory=TeamRiskBlock,
        description="팀 전체 리스크(편중·의존·협업 부족 등)",
    )
    improvement_chain: ImprovementChainBlock = Field(
        default_factory=ImprovementChainBlock,
        description="문제→설명→개선→기대효과",
    )
    dl_model_info: dict[str, Any] = Field(
        default_factory=dict,
        description="팀 통합 평가 엔진과 동일한 데이터 학습·PyTorch 메타(없으면 빈 객체)",
    )


def _attach_dl_to_members(
    req: TeamEvaluateRequest,
    members: list[MemberOut],
    request_id: str,
) -> tuple[list[MemberOut], dict[str, Any]]:
    """통합 리포트(`team_unified_eval`)와 동일 파이프라인으로 DL 보조 점수를 붙인다."""
    if not req.use_deep_learning:
        try:
            from edu_tools.team_unified_eval import _dl_model_info_disabled_by_client

            return members, _dl_model_info_disabled_by_client()
        except Exception:
            return members, {
                "model_name": "disabled",
                "enabled": False,
                "reason": "client_disabled",
            }

    try:
        from edu_tools.team_unified_eval import TeamUserIn, apply_dl_scores, run_score_engine
    except Exception:
        return members, {}

    rid = (request_id or "").strip() or str(uuid.uuid4())
    try:
        users: list[TeamUserIn] = []
        for m in req.members:
            meet = float(m.meetings_attended or 0)
            users.append(
                TeamUserIn(
                    name=m.name.strip(),
                    commits=int(m.commits or 0),
                    prs=int(m.pull_requests or 0),
                    lines=int(m.lines_changed or 0),
                    attendance=min(100.0, max(0.0, meet * 12.5)),
                    selfReport=m.self_report or "",
                )
            )
        scores, _trust = run_score_engine(users)
        dl_info = apply_dl_scores(
            users,
            scores,
            request_id=rid,
            learning=req.deep_learning_accumulate_samples,
        )
        by_name = {s.member_name.strip(): s for s in scores}
        out: list[MemberOut] = []
        for mem in members:
            s = by_name.get(mem.name.strip())
            if s is None:
                out.append(mem)
                continue
            out.append(
                mem.model_copy(
                    update={
                        "dl_score": s.dl_score,
                        "dl_confidence": s.dl_confidence,
                        "dl_blended_score": s.blendedScore,
                        "dl_top_factors": list(s.dl_top_factors or []),
                    }
                )
            )
        return out, dl_info
    except Exception:
        return members, {}


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

    by_member: list[dict[str, float]] = []
    for m in members:
        by_member.append({tp.period_label.strip(): tp.activity_score for tp in m.timeline})
    total_at: list[float] = []
    for j, lab in enumerate(all_periods):
        s = 0.0
        for mi2 in range(n):
            s += by_member[mi2].get(lab, 0.0)
        total_at.append(s)

    out: list[list[TimelinePointOut]] = []
    for mi, m in enumerate(members):
        row: list[TimelinePointOut] = []
        for j, lab in enumerate(all_periods):
            scores_at = by_member[mi].get(lab, 0.0)
            tot = total_at[j] or 1.0
            share = 100.0 * scores_at / tot
            row.append(
                TimelinePointOut(
                    period_label=lab,
                    share_percent=round(share, 1),
                    activity_score=round(scores_at, 1),
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


def _compute_team_health(
    n: int,
    spread: float,
    n_sus: int,
    n_mismatch: int,
    n_anom: int,
) -> tuple[float, str]:
    """팀 협업·기여 균형 추정(0–100, 참고용)."""
    raw = 100.0
    raw -= 10.0 * n_sus
    raw -= min(22.0, spread * 0.22)
    raw -= 5.0 * n_mismatch
    raw -= 3.0 * n_anom
    if n < 2:
        raw = min(raw, 95.0)
    score = max(0.0, min(100.0, round(raw, 1)))
    if score >= 78:
        hint = "팀 기여 분포와 신호가 비교적 안정적으로 보입니다. 면담에서 비가시 기여만 보완 확인하면 좋습니다."
    elif score >= 52:
        hint = "일부 편차·불일치·알림이 있습니다. 추가 증거·면담으로 교육적으로 보완하세요."
    else:
        hint = "편차·의심·불일치 신호가 많습니다. 팀 합의·역할 재정의·교육 개입을 검토하세요."
    return score, hint


def _build_practical_toolkit(
    members: list[MemberOut],
    mismatches: list[MismatchItem],
    anomaly_alerts: list[AnomalyAlert],
) -> PracticalToolkit:
    """기획·실무: 교육자 검수용 체크리스트."""
    items: list[str] = [
        "자동 평가 결과를 학생에게 공유하기 전, 교수·조교가 내용과 톤을 검토했는가?",
        "수업 목표·루브릭과 입력한 평가 기준이 일치하는지 확인했는가?",
        "수치·AI 문장을 단정으로 사용하지 않고, 면담·추가 증거로 보완할 계획이 있는가?",
        "결과 JSON·요약 복사본을 개인정보·저작권 정책에 맞게 보관·폐기할 것인가?",
    ]
    n_sus = sum(1 for m in members if m.free_rider_suspected)
    if n_sus:
        items.insert(
            0,
            f"무임승차 의심 플래그 {n_sus}건이 있습니다. 면담·활동 로그 요청을 기관 규정에 맞게 안내할 것인가?",
        )
    if mismatches:
        items.insert(
            min(1, len(items)),
            "기여 추정과 결과 점수(발표·동료평가 등) 불일치가 있습니다. 추가 자료 요청 여부를 확인했는가?",
        )
    if anomaly_alerts:
        items.insert(
            min(2, len(items)),
            f"고급 이상 알림이 {len(anomaly_alerts)}건 있습니다. 맥락을 면담에서 확인할 것인가?",
        )
    return PracticalToolkit(teacher_checklist=items[:12])


def _build_creative_insights(
    req: TeamEvaluateRequest,
    members: list[MemberOut],
    mismatches: list[MismatchItem],
    anomaly_alerts: list[AnomalyAlert],
) -> CreativeInsights:
    """규칙 기반 창의·설명 레이어. API 없이 항상 동작."""
    n = len(members)
    if n == 0:
        return CreativeInsights()

    cis = [m.contribution_index for m in members]
    avg_ci = sum(cis) / n
    top_i = max(range(n), key=lambda i: cis[i])
    bot_i = min(range(n), key=lambda i: cis[i])
    spread = max(cis) - min(cis)
    n_sus = sum(1 for m in members if m.free_rider_suspected)
    health_score, health_hint = _compute_team_health(
        n, spread, n_sus, len(mismatches), len(anomaly_alerts)
    )

    explain_facts: list[MemberExplainFact] = []
    for i, mem in enumerate(members):
        mreq = req.members[i] if i < len(req.members) else req.members[-1]
        facts: list[str] = []
        ci = mem.contribution_index
        if ci >= avg_ci * 1.12:
            facts.append(f"기여 지수가 팀 평균({avg_ci:.1f})보다 높은 편입니다.")
        elif ci <= avg_ci * 0.88:
            facts.append(
                f"기여 지수가 팀 평균({avg_ci:.1f})보다 낮습니다. 리뷰·기획 기여는 수치에 약하니 면담에서 확인하세요."
            )
        else:
            facts.append(f"기여 지수가 팀 평균({avg_ci:.1f}) 근처에 있습니다.")

        dt = mem.dimensions
        dims = [("기술", dt.technical), ("협업", dt.collaboration), ("주도", dt.initiative)]
        dom = max(dims, key=lambda x: x[1])
        facts.append(f"{dom[0]} 차원({dom[1]:.0f})이 상대적으로 두드러집니다.")

        c = float(mreq.commits or 0)
        t = float(mreq.tasks_completed or 0)
        pr = float(mreq.pull_requests or 0)
        lc = float(mreq.lines_changed or 0)
        meet = float(mreq.meetings_attended or 0)
        facts.append(f"입력 정량: 커밋 {int(c)} · PR {int(pr)} · 태스크 {int(t)} · 변경 라인 {int(lc)} · 회의 {int(meet)}.")
        explain_facts.append(MemberExplainFact(member_name=mem.name, facts=facts[:3]))

    rs_list = [m.role_scores or {} for m in members]

    def _avg_key(k: str) -> float:
        vals = [float(r.get(k, 0) or 0) for r in rs_list]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    dev = _avg_key("dev")
    doc = _avg_key("doc")
    leader = _avg_key("leader")
    supporter = _avg_key("supporter")
    role_pairs = [
        ("개발·구현 기여", dev),
        ("문서·정리", doc),
        ("리더·조율", leader),
        ("서포트·협업", supporter),
    ]
    mx = max(role_pairs, key=lambda x: x[1])
    mn = min(role_pairs, key=lambda x: x[1])
    balance_hint = (
        f"팀 평균 역할 분포에서 「{mx[0]}」 비중이 가장 높고, 「{mn[0]}」이 상대적으로 낮습니다. "
        "역할 공백은 면담에서 확인하면 좋습니다."
    )
    team_balance = TeamRoleBalanceOut(
        dev=dev, doc=doc, leader=leader, supporter=supporter, balance_hint=balance_hint
    )

    proj = req.project_name.strip() or "이 프로젝트"
    story = (
        f"「{proj}」팀은 {n}명으로 구성되었고, 기여 지수는 {min(cis):.1f}~{max(cis):.1f} 범위(편차 약 {spread:.1f})입니다. "
        f"상대적으로 높은 기여 지수는 {members[top_i].name} 팀원으로 추정됩니다. "
    )
    if n_sus:
        story += f"자동 규칙상 무임승차 의심 플래그는 {n_sus}건이며, 증거 확보 없이 판단하지 않도록 설계되었습니다."
    else:
        story += "현재 입력 기준으로는 무임승차 의심 플래그가 없습니다."
    if top_i != bot_i:
        story += f" {members[bot_i].name} 팀원은 기여 지수가 팀 내에서 상대적으로 낮으니, 비가시 기여 여부를 확인해 보세요."

    questions: list[str] = [
        "팀 목표와 개인 역할이 주차별로 어떻게 조정되었는지, 회의록·메신저 등으로 재현 가능한가요?",
        "코드 리뷰·기획·디자인처럼 수치에 잘 안 잡히는 기여는 어떤 방식으로 기록·제출하시겠습니까?",
        "이번 자동 평가 결과를 학생에게 공유할 때, 어떤 톤·형식(피드백 vs 순위)을 사용하시겠습니까?",
    ]
    if mismatches:
        questions.append(
            "기여 추정과 결과 점수(발표·동료평가 등)가 어긋난 학생에게는 어떤 추가 자료를 요청하시겠습니까?"
        )
    if n_sus:
        questions.append(
            "의심 플래그가 있는 학생에게는 활동 로그·PR·회의 참석을 어떻게 합리적으로 요청하시겠습니까?"
        )
    if n >= 2 and spread > 35:
        questions.append(
            "팀 내 기여 편차가 큽니다. ‘같은 노력’에 대한 팀 합의 기준이 있었는지 확인해 보시겠습니까?"
        )

    encouragement = (
        "팀 프로젝트는 학습 과정이며, 이 보고서는 대화의 출발점으로 활용해 주세요. "
        "교육자의 판단과 맥락이 항상 우선합니다."
    )
    reflection = ReflectionKit(
        team_storyline=story,
        teacher_questions=questions[:8],
        encouragement_line=encouragement,
    )

    return CreativeInsights(
        explain_facts=explain_facts,
        team_role_balance=team_balance,
        reflection_kit=reflection,
        team_health_score=health_score,
        team_health_hint=health_hint,
    )


def _template_feedback(m: MemberOut) -> str:
    return (
        f"{m.name}님의 기여 지수는 {m.contribution_index:.1f}입니다. "
        f"기술·협업·주도 차원은 각각 {m.dimensions.technical:.0f}, {m.dimensions.collaboration:.0f}, {m.dimensions.initiative:.0f} 수준으로 보입니다. "
        f"{'무임승차 의심 신호가 있어 면담·활동 로그 확인을 권장합니다. ' if m.free_rider_suspected else ''}"
        f"{m.evidence_summary[:120] if m.evidence_summary else '구체적 역할과 산출물을 문서화하면 평가 정당성이 높아집니다.'}"
    )


def _advanced_sync(
    req: TeamEvaluateRequest,
    enriched: list[MemberOut],
    suspected: list[bool],
) -> tuple[list[MemberOut], NetworkGraph, list[MismatchItem], list[AnomalyAlert], str]:
    """역할·불일치·네트워크·이상 탐지(휴리스틱만, 외부 API 없음)."""
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
    return out_members, net, mm, anom, summary


def _try_openai_enrich(
    req: TeamEvaluateRequest,
    out_members: list[MemberOut],
    summary: str,
    api_key: str,
) -> tuple[str, str]:
    """불일치 해설 보강. 실패 시 원 요약·휴리스틱 모드."""
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
        data = openai_enrich_advanced(project, members_payload, heuristic_bundle, api_key)
        comment = str(data.get("contribution_outcome_comment") or "").strip()
        if comment:
            return comment + "\n\n" + summary, "openai_enriched"
    except Exception:
        pass
    return summary, "heuristic"


def _safe_openai_feedbacks(req: TeamEvaluateRequest, enriched: list[MemberOut], api_key: str) -> list[str]:
    try:
        return _openai_feedbacks(req, enriched, api_key)
    except Exception:
        return [_template_feedback(m) for m in enriched]


def _openai_feedbacks(req: TeamEvaluateRequest, members: list[MemberOut], api_key: str) -> list[str]:
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

    client = get_openai_client(api_key)
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
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


def _sync_freerider_dl_layer(members: list[MemberOut]) -> list[MemberOut]:
    """DL 병합 후 Rule `risk_level`·무임승차 플래그·신호 목록 정합."""
    out: list[MemberOut] = []
    for m in members:
        rm = m.freerider_detection.rule_metrics
        suspected_now = rm.risk_level == "suspected"
        sigs = list(m.free_rider_signals)
        note = (rm.dl_model_note or "").strip()
        if rm.dl_model_freerider_support and note and note not in sigs:
            sigs.insert(0, note)
        fr_risk = float(m.free_rider_risk or 0.0)
        if rm.dl_model_freerider_support:
            fr_risk = max(fr_risk, 58.0)
        out.append(
            m.model_copy(
                update={
                    "free_rider_suspected": bool(m.free_rider_suspected or suspected_now),
                    "free_rider_risk": fr_risk,
                    "free_rider_signals": sigs[:14],
                }
            )
        )
    return out


def _apply_instructor_freerider_overrides(
    req: TeamEvaluateRequest, members: list[MemberOut]
) -> list[MemberOut]:
    """교육자 입력으로 화면용 무임승차 의심 플래그·위험도·신호를 조정. Rule 상세 리포트는 감사용으로 유지."""
    out: list[MemberOut] = []
    for i, m in enumerate(members):
        mi = req.members[i] if i < len(req.members) else req.members[-1]
        ov = mi.instructor_freerider_override
        note = (mi.instructor_override_note or "").strip()[:500]
        base_echo = {
            "instructor_freerider_override": ov,
            "instructor_override_note": note,
        }
        if ov == "none":
            out.append(
                m.model_copy(
                    update={
                        **base_echo,
                        "freerider_override_effect_ko": "",
                    }
                )
            )
            continue

        sigs = list(m.free_rider_signals)
        if ov == "clear_suspicion":
            suspected = False
            risk = min(float(m.free_rider_risk), 22.0)
            effect = (
                "교육자 입력에 따라 화면용 무임승차 의심을 해제했습니다. "
                "세부 Rule·DL 리포트는 자동 산출 그대로이며 면담·감사 시 함께 참고하세요."
            )
            prefix = "교육자 오버라이드(의심 해제)"
            if note:
                prefix += f": {note}"
            if not any(prefix in s for s in sigs):
                sigs.insert(0, prefix)
        else:
            suspected = True
            risk = max(float(m.free_rider_risk), 62.0)
            effect = (
                "교육자 입력에 따라 무임승차 의심을 유지·강조했습니다. "
                "최종 인사·성적 판단은 기관 규정에 따릅니다."
            )
            prefix = "교육자 오버라이드(의심 유지·강조)"
            if note:
                prefix += f": {note}"
            if not any(prefix in s for s in sigs):
                sigs.insert(0, prefix)

        out.append(
            m.model_copy(
                update={
                    **base_echo,
                    "free_rider_suspected": suspected,
                    "free_rider_risk": round(risk, 1),
                    "free_rider_signals": sigs[:14],
                    "freerider_override_effect_ko": effect,
                }
            )
        )
    return out


def _finalize_members(
    req: TeamEvaluateRequest,
    base: list[MemberOut],
    contribution_indices: list[float],
    mode: str,
    fairness_notes: str = "",
    *,
    enrich_openai: bool = True,
    request_id: str = "",
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

    out_members, net, mm, anom, co_summary = _advanced_sync(req, enriched, suspected)

    out_members, dl_model_info = _attach_dl_to_members(req, out_members, request_id)

    from edu_tools.team_freerider import (
        PRODUCT_TAGLINE_KO,
        build_team_dashboard,
        compute_freerider_reports,
        merge_dl_freerider_signals,
        freerider_detection_overview,
        safe_freerider_ai_summaries,
    )
    from edu_tools.team_contest_layers import build_contest_layers

    fr_reports = compute_freerider_reports(req, out_members, timelines, net)
    fr_reports = merge_dl_freerider_signals(out_members, fr_reports)
    out_members = [
        out_members[i].model_copy(update={"freerider_detection": fr_reports[i]})
        for i in range(len(out_members))
    ]
    out_members = _sync_freerider_dl_layer(out_members)
    out_members = _apply_instructor_freerider_overrides(req, out_members)
    n_sus_final = sum(1 for m in out_members if m.free_rider_suspected)
    n_ov = sum(1 for m in req.members if m.instructor_freerider_override != "none")
    fr_summary = (
        f"자동 탐지: 무임승차 의심 플래그 {n_sus_final}명. "
        "Rule·DL 보조·키워드 기반이며, 최종 판단은 교수·팀 면담으로 하세요."
    )
    if n_ov:
        fr_summary += f" 교육자 무임승차 오버라이드 반영 {n_ov}건."
    fr_overview = freerider_detection_overview(out_members)

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    adv_mode = "heuristic"

    if enrich_openai and key:
        # 피드백·불일치 해설·무임승차 4단계 AI 요약을 병렬 실행
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_fb = pool.submit(_safe_openai_feedbacks, req, enriched, key)
            fut_en = pool.submit(_try_openai_enrich, req, out_members, co_summary, key)
            fut_fr = pool.submit(safe_freerider_ai_summaries, req, out_members, key)
            fbs = fut_fb.result()
            co_summary, adv_mode = fut_en.result()
            fr_ai = fut_fr.result()
        out_members = [
            out_members[i].model_copy(
                update={
                    "ai_feedback": fbs[i],
                    "freerider_detection": out_members[i].freerider_detection.model_copy(
                        update={
                            "ai_detection_summary": fr_ai[i]
                            if i < len(fr_ai)
                            else ""
                        }
                    ),
                }
            )
            for i in range(len(out_members))
        ]
    else:
        out_members = [
            out_members[i].model_copy(update={"ai_feedback": _template_feedback(out_members[i])})
            for i in range(len(out_members))
        ]

    team_dashboard = build_team_dashboard(out_members)
    rubric_rep, trust_blk, risk_blk, improve_blk = build_contest_layers(req, out_members, net)

    creative = _build_creative_insights(req, out_members, mm, anom)
    practical = _build_practical_toolkit(out_members, mm, anom)

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
        creative_insights=creative,
        practical_toolkit=practical,
        freerider_detection_overview=fr_overview,
        product_tagline_ko=PRODUCT_TAGLINE_KO,
        team_dashboard=team_dashboard,
        rubric_report=rubric_rep,
        evaluation_trust=trust_blk,
        team_risk=risk_blk,
        improvement_chain=improve_blk,
        dl_model_info=dl_model_info,
    )


def _heuristic(req: TeamEvaluateRequest, request_id: str = "") -> TeamEvaluateResponse:
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

    return _finalize_members(req, out, contribution_indices, "heuristic", note, request_id=request_id)


def _openai_eval(req: TeamEvaluateRequest, api_key: str, request_id: str = "") -> TeamEvaluateResponse:
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

    client = get_openai_client(api_key)
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
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
            return _heuristic(req, request_id)
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
        return _heuristic(req, request_id)

    fn = str(data.get("fairness_notes", ""))[:2000]
    return _finalize_members(req, out, contribution_indices, "ai", fn, request_id=request_id)


def _stamp_team_response(resp: TeamEvaluateResponse, request_id: str, t0: float) -> TeamEvaluateResponse:
    ms = (time.perf_counter() - t0) * 1000
    return resp.model_copy(
        update={
            "request_id": request_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "processing_ms": round(ms, 2),
        }
    )


def _stamp_compare_response(resp: TeamCompareResponse, request_id: str, t0: float) -> TeamCompareResponse:
    ms = (time.perf_counter() - t0) * 1000
    return resp.model_copy(
        update={
            "request_id": request_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "processing_ms": round(ms, 2),
        }
    )


@router.post("/evaluate/compare", response_model=TeamCompareResponse)
async def evaluate_team_compare(body: TeamEvaluateRequest) -> TeamCompareResponse:
    """ChatGPT·Gemini·Claude·Grok 최신(기본) 모델 각각 독립 평가 후 기여 지수 비교."""
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    from edu_tools.team_multi_llm import run_team_compare

    return _stamp_compare_response(run_team_compare(body), request_id, t0)


@router.post("/evaluate", response_model=TeamEvaluateResponse)
async def evaluate_team(body: TeamEvaluateRequest) -> TeamEvaluateResponse:
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    has_llm = bool(
        (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
        or (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        or (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or "").strip()
    )
    if has_llm:
        try:
            from edu_tools.team_multi_llm import run_parallel_team_eval

            merged = run_parallel_team_eval(body, request_id=request_id)
            if merged is not None:
                return _stamp_team_response(merged, request_id, t0)
        except Exception:
            pass
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if key:
            try:
                return _stamp_team_response(_openai_eval(body, key, request_id), request_id, t0)
            except Exception:
                return _stamp_team_response(_heuristic(body, request_id), request_id, t0)
        return _stamp_team_response(_heuristic(body, request_id), request_id, t0)
    return _stamp_team_response(_heuristic(body, request_id), request_id, t0)
