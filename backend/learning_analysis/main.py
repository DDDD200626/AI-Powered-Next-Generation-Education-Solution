"""학습 과정 vs 시험 불일치 · 다중 LLM 분석 API."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from edu_tools.at_risk import router as at_risk_router
from edu_tools.course_qa import router as course_qa_router
from edu_tools.discussion import router as discussion_router
from edu_tools.feedback import router as feedback_router
from edu_tools.rubric_align import router as rubric_align_router
from edu_tools.team import router as team_router
from learning_analysis.compare_freeform import compare_llm_async
from learning_analysis.pipeline import analyze_async, provider_keys_status
from learning_analysis.schemas import AnalyzeRequest, AnalyzeResponse, LLMCompareRequest, LLMCompareResponse


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").strip()
    if not raw:
        return ["http://127.0.0.1:5173", "http://localhost:5173"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="팀 프로젝트 기여도 자동 평가 시스템",
    description="핵심: POST /api/team/evaluate — 기여도·무임승차 의심·타임라인·팀원 피드백. 부가: 학습-시험 분석, 4모델 비교, 이탈·피드백·QnA·토론·루브릭",
    version="4.0.0",
)

app.include_router(team_router, prefix="/api/team", tags=["team"])
app.include_router(at_risk_router, prefix="/api/at-risk", tags=["at-risk"])
app.include_router(feedback_router, prefix="/api/feedback", tags=["feedback"])
app.include_router(course_qa_router, prefix="/api/course", tags=["course"])
app.include_router(discussion_router, prefix="/api/discussion", tags=["discussion"])
app.include_router(rubric_align_router, prefix="/api/rubric", tags=["rubric"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    st = provider_keys_status()
    return {"status": "ok", "providers": st}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def api_analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    return await analyze_async(body)


@app.post("/api/llm/compare", response_model=LLMCompareResponse)
async def api_llm_compare(body: LLMCompareRequest) -> LLMCompareResponse:
    return await compare_llm_async(body)
