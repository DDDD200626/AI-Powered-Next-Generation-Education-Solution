"""학습 과정 vs 시험 결과 불일치 분석 — 요청/응답 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LearningProcessInput(BaseModel):
    """학습 과정 지표 (가능한 항목만 채움)."""

    weekly_study_hours_self_report: float | None = Field(None, ge=0, description="자기보고 주간 학습시간")
    lms_video_watch_ratio: float | None = Field(None, ge=0, le=100, description="강의 시청률 %")
    quiz_average: float | None = Field(None, ge=0, le=100, description="캐퀴즈·소평균 %")
    assignment_on_time_ratio: float | None = Field(None, ge=0, le=100, description="과제 기한 내 제출 비율 %")
    discussion_or_forum_count: int | None = Field(None, ge=0, description="토론/포럼 참여 횟수")
    attendance_or_checkin_ratio: float | None = Field(None, ge=0, le=100, description="출석·출첵 비율 %")
    notes: str = Field("", description="학습 과정에 대한 자유 서술")


class ExamResultInput(BaseModel):
    """시험·평가 결과."""

    midterm_score: float | None = Field(None, ge=0, le=100, description="중간고사 %")
    final_or_recent_exam_score: float | None = Field(None, ge=0, le=100, description="기말/최근 시험 %")
    exam_time_anomaly_note: str = Field(
        "",
        description="시험 소요시간 이상, 제출 패턴 등 (있을 때만)",
    )
    notes: str = Field("", description="시험 관련 메모")


class AnalyzeRequest(BaseModel):
    course_name: str = Field("", description="과목명 (선택)")
    student_or_group_label: str = Field("", description="익명 라벨 등")
    learning: LearningProcessInput = Field(default_factory=LearningProcessInput)
    exam: ExamResultInput = Field(default_factory=ExamResultInput)
    context_for_educator: str = Field(
        "",
        description="교수·조교만 아는 맥락(시험 난이도, 오픈북 여부 등)",
    )


class ModelJudgment(BaseModel):
    provider: str
    model_label: str
    ok: bool = True
    error: str | None = None
    cheating_likelihood: float | None = Field(None, ge=0, le=100, description="부정행위 의심 0–100")
    learning_state_summary: str = Field("", description="학습 상태 요약")
    mismatch_analysis: str = Field("", description="과정-시험 불일치 해석")
    future_prediction: str = Field("", description="이후 학습·성취 위험 예측")
    confidence_note: str = Field("", description="불확실성·한계")


class AnalyzeResponse(BaseModel):
    providers_used: list[str] = Field(default_factory=list)
    providers_skipped: list[str] = Field(default_factory=list)
    judgments: list[ModelJudgment]
    consensus_cheating_avg: float | None = Field(None, description="모델 평균 부정행위 의심도")
    consensus_summary: str = Field("", description="종합 요약")
    disclaimer: str = Field("", description="법적·교육적 한계 안내")
    perf: dict[str, float] | None = Field(
        default=None,
        description="병목 분석(ms): llm_parallel_ms, local_ms, total_ms — 언어 변경 전에 확인",
    )


class LLMCompareRequest(BaseModel):
    """Gemini·ChatGPT·Claude·Grok 동일 프롬프트 병렬 분석."""

    prompt: str = Field(..., min_length=1, max_length=20000, description="분석할 본문·질문")
    system_hint: str = Field("", max_length=4000, description="추가 시스템 지시(선택)")
    task_title: str = Field("", max_length=200, description="작업 제목(선택)")


class LLMTextResult(BaseModel):
    provider: str
    model_label: str
    ok: bool = True
    text: str = ""
    error: str | None = None


class LLMCompareResponse(BaseModel):
    providers_used: list[str] = Field(default_factory=list)
    providers_skipped: list[str] = Field(default_factory=list)
    results: list[LLMTextResult]
    disclaimer: str = ""
    perf: dict[str, float] | None = Field(
        default=None,
        description="병목 분석(ms): llm_parallel_ms, local_ms, total_ms",
    )
