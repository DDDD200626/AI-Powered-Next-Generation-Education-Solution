"""학습 과정 vs 시험 불일치 · 다중 LLM 분석 API."""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from collections import deque
from typing import Any

_APP_START_MONO = time.monotonic()

# 최근 요청 처리 시간(프로파일·병목 확인용, 기본 최대 50건)
_PERF_LOCK = threading.Lock()
_PERF_RECENT: deque[dict[str, Any]] = deque(maxlen=50)


def _perf_ring_enabled() -> bool:
    return os.environ.get("PERF_RING_BUFFER", "1").strip().lower() not in ("0", "false", "no")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Response
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
    _default = (
        "http://127.0.0.1:5173,http://localhost:5173,"
        "http://127.0.0.1:4173,http://localhost:4173,"
        "http://127.0.0.1:8080,http://localhost:8080"
    )
    raw = os.environ.get("CORS_ORIGINS", _default).strip()
    if not raw:
        return [o.strip() for o in _default.split(",") if o.strip()]
    return [o.strip() for o in raw.split(",") if o.strip()]


_APP_DESC = """## 심사기준 정렬
- **기술적 완성도**: FastAPI·Vite, OpenAPI(`/docs`), **`GET /api/observability`·`/api/ready`·`/api/live`**, **Docker**·docker-compose, **CI·pytest**, **`X-Request-ID`**, **`GET /api/capabilities`**, **`POST /api/team/evaluate` OpenAPI 예시**.
- **AI 활용 능력 및 효율성**: 다중 LLM **asyncio 병렬**(`/api/analyze`), 4모델 병렬(`/api/llm/compare`), 팀 평가 **ThreadPoolExecutor 병렬**, **`OPENAI_TIMEOUT_SEC`**로 호출 상한, 휴리스틱 폴백.
- **기획력 및 실무 접합성**: 팀 과제·조교 흐름, 응답 **`practical_toolkit`(교육자 체크리스트)**, 부가 모듈, 결과 내보내기, **데모 데이터** 입력(프론트).
- **창의성**: 불일치·네트워크·역할·이상 탐지·**창의 인사이트**·**팀 협업 건강도**·가상 시뮬레이터. **`docs/CONTEST_RUBRIC.md`**.

## 핵심 API
`POST /api/team/evaluate` — 기여도·무임승차 의심·타임라인·팀원 피드백·(고급) 네트워크·불일치·역할·이상 알림.

## 부가
학습–시험 분석, 4모델 비교, 이탈·피드백·QnA·토론·루브릭."""

app = FastAPI(
    title="팀 프로젝트 기여도 자동 평가 시스템",
    description=_APP_DESC,
    version="4.7.4",
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
                    "GET /api/observability, /api/ready, /api/live, GET /api/perf/recent — 운영·병목",
                    ".github/workflows/ci.yml — CI",
                    "backend/tests/ — pytest",
                    "docker-compose.yml, backend/Dockerfile, frontend/Dockerfile",
                ],
            },
            "ai_efficiency": {
                "label_ko": "AI 활용 능력 및 효율성",
                "summary": "다중 LLM 병렬, 팀 평가 병렬 생성, OPENAI_TIMEOUT_SEC, 휴리스틱",
                "evidence": [
                    "learning_analysis/pipeline.py — asyncio.gather",
                    "learning_analysis/compare_freeform.py — 4모델 병렬",
                    "edu_tools/team.py, edu_tools/team_multi_llm.py — 4모델 병렬 팀 평가·ThreadPoolExecutor",
                    "edu_tools/team_advanced.py — openai_enrich_advanced",
                ],
            },
            "planning_practical": {
                "label_ko": "기획력 및 실무 접합성",
                "summary": "교육 팀 과제·practical_toolkit(교육자 체크리스트)·부가 도구·내보내기",
                "evidence": [
                    "POST /api/team/evaluate — practical_toolkit",
                    "POST /api/at-risk/evaluate, /api/feedback/draft, /api/course/ask, …",
                    "프론트 데모 데이터 입력 (main.ts applyDemoTeamData)",
                ],
            },
            "creativity": {
                "label_ko": "창의성",
                "summary": "불일치·네트워크·역할·이상 탐지·창의 인사이트·팀 건강도·시뮬레이터",
                "evidence": [
                    "edu_tools/team_advanced.py",
                    "creative_insights.team_health_score (team.py)",
                    "프론트 가상 시뮬레이터 (main.ts)",
                ],
            },
        },
        "ai_providers_configured": provider_keys_status(),
        "endpoints": {
            "capabilities": "GET /api/capabilities",
            "health": "GET /api/health",
            "version": "GET /api/version",
            "observability": "GET /api/observability",
            "ready": "GET /api/ready",
            "live": "GET /api/live",
            "openapi_json": "GET /openapi.json",
            "team_evaluate": "POST /api/team/evaluate",
            "analyze": "POST /api/analyze",
            "llm_compare": "POST /api/llm/compare",
            "perf_recent": "GET /api/perf/recent",
            "rubric_draft": "POST /api/rubric/draft",
        },
    }


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
    if _perf_ring_enabled():
        with _PERF_LOCK:
            _PERF_RECENT.append(
                {
                    "path": request.url.path,
                    "method": request.method,
                    "status": response.status_code,
                    "ms": round(ms, 2),
                    "request_id": rid,
                }
            )
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
async def health(response: Response):
    """프론트 상태 표시용: 제공자·버전·문서 경로를 한 번에 반환(요청 수 최소화)."""
    response.headers["Cache-Control"] = "private, max-age=5"
    st = provider_keys_status()
    return {
        "status": "ok",
        "providers": st,
        "version": app.version,
        "app_name": app.title,
        "openapi_docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/api/version")
async def api_version(response: Response):
    response.headers["Cache-Control"] = "private, max-age=60"
    return {
        "name": app.title,
        "version": app.version,
        "openapi_docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/api/capabilities")
async def api_capabilities(response: Response):
    """공모전 심사 4개 기준 요약·증빙 경로·엔드포인트 목록(JSON)."""
    response.headers["Cache-Control"] = "private, max-age=30"
    return _capabilities_payload()


@app.get("/api/observability")
async def api_observability(response: Response):
    """운영·관측성: 업타임·환경·런타임 메타 (모니터링·온콜·심사 근거)."""
    response.headers["Cache-Control"] = "no-store"
    commit = (os.environ.get("GIT_COMMIT") or os.environ.get("SOURCE_COMMIT") or "").strip() or None
    return {
        "service": app.title,
        "version": app.version,
        "uptime_seconds": round(time.monotonic() - _APP_START_MONO, 3),
        "environment": os.environ.get("APP_ENV", "development"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "commit": commit,
    }


@app.get("/api/ready")
async def api_ready(response: Response):
    """Kubernetes 등에서 사용하는 readiness(트래픽 수신 가능) 프로브."""
    response.headers["Cache-Control"] = "no-store"
    return {"status": "ready", "checks": {"app": "ok"}}


@app.get("/api/live")
async def api_live(response: Response):
    """Liveness 프로세스 생존 확인(재시작 판단용)."""
    response.headers["Cache-Control"] = "no-store"
    return {"status": "live"}


@app.get("/api/perf/recent")
async def api_perf_recent(response: Response):
    """최근 요청 처리 시간(링 버퍼). 언어 변경 전 병목(네트워크 vs 로컬) 확인용."""
    response.headers["Cache-Control"] = "no-store"
    with _PERF_LOCK:
        items = list(_PERF_RECENT)
    return {
        "hint": "헤더 X-Process-Time-Ms = 전체. POST /api/analyze·/api/llm/compare 응답의 perf.llm_parallel_ms ≈ 외부 LLM 대기.",
        "ring_buffer_enabled": _perf_ring_enabled(),
        "recent": items,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def api_analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    return await analyze_async(body)


@app.post("/api/llm/compare", response_model=LLMCompareResponse)
async def api_llm_compare(body: LLMCompareRequest) -> LLMCompareResponse:
    return await compare_llm_async(body)
