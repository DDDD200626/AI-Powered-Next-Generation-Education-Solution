"""학습 과정 vs 시험 불일치 · 다중 LLM 분석 API."""

from __future__ import annotations

import os
import uuid
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request

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


_APP_DESC = """## 심사기준 정렬
- **기술적 완성도**: FastAPI·Vite, OpenAPI(`/docs`), **Docker**·docker-compose, **CI·pytest**, **`X-Request-ID`**, **`GET /api/capabilities`**(심사 4축 JSON).
- **AI 활용 능력 및 효율성**: 다중 LLM **asyncio 병렬**(`/api/analyze`), 4모델 병렬(`/api/llm/compare`), 팀 평가 **ThreadPoolExecutor 병렬**, 휴리스틱 폴백.
- **기획력 및 실무 접합성**: 팀 과제·조교 흐름, 부가 모듈(at-risk·피드백·Q&A·토론·루브릭), 결과 내보내기.
- **창의성**: 불일치·네트워크·역할·이상 탐지·**창의 인사이트**·가상 시뮬레이터. 저장소 **`docs/CONTEST_RUBRIC.md`** 공모전 대응서.

## 핵심 API
`POST /api/team/evaluate` — 기여도·무임승차 의심·타임라인·팀원 피드백·(고급) 네트워크·불일치·역할·이상 알림.

## 부가
학습–시험 분석, 4모델 비교, 이탈·피드백·QnA·토론·루브릭."""

app = FastAPI(
    title="팀 프로젝트 기여도 자동 평가 시스템",
    description=_APP_DESC,
    version="4.5.0",
)


def _capabilities_payload() -> dict[str, Any]:
    """공모전·심사위원용: 4개 심사기준과 근거를 기계 판독 가능한 형태로 제공."""
    return {
        "product": app.title,
        "version": app.version,
        "documentation": {
            "openapi": "/docs",
            "redoc": "/redoc",
            "contest_rubric_doc": "docs/CONTEST_RUBRIC.md (저장소)",
        },
        "rubric": {
            "technical_completeness": {
                "label_ko": "기술적 완성도",
                "summary": "FastAPI·Vite 분리, Pydantic, OpenAPI, CI/pytest, 요청 추적, Docker",
                "evidence": [
                    "backend/learning_analysis/main.py — 앱·미들웨어",
                    ".github/workflows/ci.yml — CI",
                    "backend/tests/ — pytest",
                    "docker-compose.yml, backend/Dockerfile, frontend/Dockerfile",
                ],
            },
            "ai_efficiency": {
                "label_ko": "AI 활용 능력 및 효율성",
                "summary": "다중 LLM 병렬, 팀 평가 병렬 생성, 키 없을 때 휴리스틱",
                "evidence": [
                    "learning_analysis/pipeline.py — asyncio.gather",
                    "learning_analysis/compare_freeform.py — 4모델 병렬",
                    "edu_tools/team.py — ThreadPoolExecutor",
                ],
            },
            "planning_practical": {
                "label_ko": "기획력 및 실무 접합성",
                "summary": "교육 팀 과제 입력·부가 도구·내보내기·면담 키트",
                "evidence": [
                    "POST /api/team/evaluate",
                    "POST /api/at-risk/evaluate, /api/feedback/draft, /api/course/ask, …",
                ],
            },
            "creativity": {
                "label_ko": "창의성",
                "summary": "기여–결과 불일치, 협업 네트워크, 역할·이상 탐지, 창의 인사이트, 시뮬레이터",
                "evidence": [
                    "edu_tools/team_advanced.py",
                    "creative_insights (team.py)",
                    "프론트 가상 시뮬레이터 (main.ts)",
                ],
            },
        },
        "ai_providers_configured": provider_keys_status(),
        "endpoints": {
            "capabilities": "GET /api/capabilities",
            "health": "GET /api/health",
            "version": "GET /api/version",
            "team_evaluate": "POST /api/team/evaluate",
            "analyze": "POST /api/analyze",
            "llm_compare": "POST /api/llm/compare",
        },
    }


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


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
app.add_middleware(GZipMiddleware, minimum_size=800)


@app.get("/api/health")
async def health():
    st = provider_keys_status()
    return {"status": "ok", "providers": st, "version": app.version}


@app.get("/api/version")
async def api_version():
    return {
        "name": app.title,
        "version": app.version,
        "openapi_docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/api/capabilities")
async def api_capabilities():
    """공모전 심사 4개 기준 요약·증빙 경로·엔드포인트 목록(JSON)."""
    return _capabilities_payload()


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def api_analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    return await analyze_async(body)


@app.post("/api/llm/compare", response_model=LLMCompareResponse)
async def api_llm_compare(body: LLMCompareRequest) -> LLMCompareResponse:
    return await compare_llm_async(body)
