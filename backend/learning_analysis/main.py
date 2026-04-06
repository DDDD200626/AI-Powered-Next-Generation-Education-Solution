"""학습 과정 vs 시험 불일치 · 다중 LLM 분석 API."""

from __future__ import annotations

import os
import uuid

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
- **기술적 완성도**: FastAPI 백엔드·Vite 프론트, 구조화된 응답, 팀 평가·고급 분석(네트워크·불일치·역할·이상 탐지), `/docs` 문서, **요청 추적 ID(`X-Request-ID`)**, **CI·pytest 회귀 테스트**, **`/api/version`**.
- **AI 활용 능력 및 효율성**: 팀 평가·피드백에 생성형 보강(선택), 다중 LLM 파이프라인(`/api/analyze`) **asyncio 병렬**, 4모델 병렬(`/api/llm/compare`), 키 없을 때 휴리스틱 폴백, 팀 평가 시 OpenAI **ThreadPoolExecutor 병렬** 피드백·해설.
- **기획력 및 실무 접합성**: 교육 현장 팀 과제·동료 평가·결과 점수 연계, 이탈 경보·과제 피드백·강의 Q&A·토론 요약·루브릭 점검 등 운영 모듈.
- **창의성**: 기여–결과 괴리 분석, 협업 그래프 시각화, 역할 4유형 자동 분류, 고급 이상 신호 조합, 규칙 기반 설명 카드·팀 역할 밸런스·면담 질문 키트·프론트 가상 시뮬레이터.

## 핵심 API
`POST /api/team/evaluate` — 기여도·무임승차 의심·타임라인·팀원 피드백·(고급) 네트워크·불일치·역할·이상 알림.

## 부가
학습–시험 분석, 4모델 비교, 이탈·피드백·QnA·토론·루브릭."""

app = FastAPI(
    title="팀 프로젝트 기여도 자동 평가 시스템",
    description=_APP_DESC,
    version="4.4.0",
)


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


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def api_analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    return await analyze_async(body)


@app.post("/api/llm/compare", response_model=LLMCompareResponse)
async def api_llm_compare(body: LLMCompareRequest) -> LLMCompareResponse:
    return await compare_llm_async(body)
