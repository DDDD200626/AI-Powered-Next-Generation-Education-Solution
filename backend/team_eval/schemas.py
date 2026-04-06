"""요청·응답 스키마 — 팀 프로젝트 기여도 평가."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MemberInput(BaseModel):
    name: str = Field(..., min_length=1, description="이름 또는 식별자")
    role: str = Field("", description="역할 (예: 프론트, 백엔드, 기획)")
    commits: int | None = Field(None, ge=0, description="커밋 수 (Git 등, 선택)")
    pull_requests: int | None = Field(None, ge=0, description="PR/MR 수 (선택)")
    lines_changed: int | None = Field(None, ge=0, description="변경 라인 수 근사 (선택)")
    tasks_completed: int | None = Field(None, ge=0, description="완료한 태스크 수 (선택)")
    meetings_attended: int | None = Field(None, ge=0, description="참여한 회의/스탠드업 횟수 (선택)")
    self_report: str = Field("", description="본인 기여 서술")
    peer_notes: str = Field("", description="동료/팀 관점 메모 (익명 요약 등)")


class TeamEvaluateRequest(BaseModel):
    project_name: str = Field(..., min_length=1)
    project_description: str = Field("", description="과제 목표·범위")
    evaluation_criteria: str = Field(
        "",
        description="추가 평가 기준 (교수/조교 루브릭 요약 등)",
    )
    members: list[MemberInput] = Field(..., min_length=1)


class DimensionScores(BaseModel):
    technical: float = Field(..., ge=0, le=100, description="기술·구현 기여")
    collaboration: float = Field(..., ge=0, le=100, description="협업·커뮤니케이션")
    initiative: float = Field(..., ge=0, le=100, description="주도성·문제 해결")


class MemberEvaluation(BaseModel):
    name: str
    role: str = ""
    contribution_index: float = Field(..., ge=0, le=100, description="종합 기여 지수 0–100")
    dimensions: DimensionScores
    evidence_summary: str = Field("", description="근거 요약")
    caveats: str = Field("", description="지표 한계·주의점")


class TeamEvaluateResponse(BaseModel):
    mode: str = Field(..., description="'ai' | 'heuristic'")
    project_summary: str = Field("", description="프로젝트 맥락 요약")
    fairness_notes: str = Field("", description="공정성·한계 안내")
    members: list[MemberEvaluation]
    disclaimer: str = Field(
        "",
        description="자동 평가는 보조용이며 최종 성적의 유일한 근거가 되지 않습니다.",
    )
