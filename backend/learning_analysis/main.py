"""학습 과정 vs 시험 불일치 · 다중 LLM 분석 API."""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from functools import partial
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

_APP_START_MONO = time.monotonic()

# 최근 요청 처리 시간(프로파일·병목 확인용, 기본 최대 50건)
_PERF_LOCK = threading.Lock()
_PERF_RECENT: deque[dict[str, Any]] = deque(maxlen=50)

_RL_LOCK = threading.Lock()
_RL_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def _perf_ring_enabled() -> bool:
    return os.environ.get("PERF_RING_BUFFER", "1").strip().lower() not in ("0", "false", "no")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request

from edu_tools.contest_judge_pack import contest_submission_pack
from edu_tools.dl_roadmap import roadmap_payload
from edu_tools.at_risk import router as at_risk_router
from edu_tools.course_qa import router as course_qa_router
from edu_tools.discussion import router as discussion_router
from edu_tools.feedback import router as feedback_router
from edu_tools.rubric_align import router as rubric_align_router
from edu_tools.team import router as team_router
from edu_tools.team_data_store import contribution_trends, db_profile, model_monitor
from edu_tools.team_ml_model import dataset_label_streaming_stats
from edu_tools.team_model_report import build_limits_report
from edu_tools.team_unified_eval import router as team_unified_router
from learning_analysis.compare_freeform import compare_llm_async
from learning_analysis.connection_dl_advise import router as connection_dl_router
from learning_analysis.pipeline import analyze_async, provider_keys_status
from learning_analysis.schemas import AnalyzeRequest, AnalyzeResponse, LLMCompareRequest, LLMCompareResponse


def _post_rate_limit_per_minute() -> int:
    raw = (os.environ.get("EDUSIGNAL_POST_RATE_LIMIT_PER_MINUTE") or "0").strip()
    try:
        n = int(raw)
    except ValueError:
        return 0
    return max(0, min(n, 600))


def _post_rate_limit_path(path: str) -> bool:
    """비용 큰 POST만 제한(기본 0=비활성)."""
    prefixes = (
        "/api/team/report",
        "/api/team/evaluate",
        "/api/analyze",
        "/api/llm/compare",
    )
    return any(path == p or path.startswith(f"{p}/") for p in prefixes)


def _rate_limit_client_key(request: Request) -> str:
    xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if xff:
        return xff[:200]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _post_rate_limit_should_block(request: Request) -> bool:
    limit = _post_rate_limit_per_minute()
    if limit <= 0 or request.method != "POST" or not _post_rate_limit_path(request.url.path):
        return False
    now = time.monotonic()
    key = _rate_limit_client_key(request)
    with _RL_LOCK:
        dq = _RL_BUCKETS[key]
        while dq and now - dq[0] > 60.0:
            dq.popleft()
        if len(dq) >= limit:
            return True
        dq.append(now)
    return False


def _apply_security_headers(response: Response) -> None:
    raw = (os.environ.get("EDUSIGNAL_SECURITY_HEADERS", "1") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    hsts = (os.environ.get("EDUSIGNAL_HSTS_MAX_AGE") or "").strip()
    if hsts.isdigit() and int(hsts) > 0:
        response.headers.setdefault(
            "Strict-Transport-Security",
            f"max-age={int(hsts)}; includeSubDomains",
        )


def _cors_origins() -> list[str]:
    _default = (
        "http://127.0.0.1:5173,http://localhost:5173,"
        "http://127.0.0.1:5174,http://localhost:5174,"
        "http://127.0.0.1:4173,http://localhost:4173,"
        "http://127.0.0.1:8080,http://localhost:8080,"
        "http://127.0.0.1:8000,http://localhost:8000"
    )
    raw = os.environ.get("CORS_ORIGINS", _default).strip()
    if not raw:
        return [o.strip() for o in _default.split(",") if o.strip()]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _cors_origin_regex() -> str | None:
    """LAN IP로 Vite 접속 시 Origin이 192.168.x.x:5173 등이 됨 — 직접 API 호출(CORS) 허용."""
    raw = os.environ.get("CORS_ORIGIN_REGEX", "").strip()
    if raw.lower() in ("0", "false", "no", "off"):
        return None
    if raw:
        return raw
    return (
        r"^http://(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r":(5173|5174|4173|8080|8000)$"
    )


_APP_DESC = """## 심사기준 정렬
- **기술적 완성도**: FastAPI·Vite, OpenAPI(`/docs`), **`GET /api/observability`·`/api/ready`·`/api/live`**, **Docker**·docker-compose, **CI·pytest**, **`X-Request-ID`**, **`GET /api/capabilities`**, **`POST /api/team/evaluate` OpenAPI 예시**.
- **AI 활용 능력 및 효율성**: 다중 LLM **asyncio 병렬**(`/api/analyze`), 4모델 병렬(`/api/llm/compare`), 팀 평가 **ThreadPoolExecutor 병렬**, **`OPENAI_TIMEOUT_SEC`**로 호출 상한, 휴리스틱 폴백.
- **기획력 및 실무 접합성**: 팀 과제·조교 흐름, 응답 **`practical_toolkit`(교육자 체크리스트)**, 부가 모듈, 결과 내보내기, **데모 데이터** 입력(프론트).
- **창의성**: 불일치·네트워크·역할·이상 탐지·**창의 인사이트**·**팀 협업 건강도**·가상 시뮬레이터. **`docs/CONTEST_RUBRIC.md`**.

## 핵심 API
`POST /api/team/evaluate` — 기여도·무임승차 의심·타임라인·팀원 피드백·(고급) 네트워크·불일치·역할·이상 알림.

## 부가
학습–시험 분석, 4모델 비교, 이탈·피드백·QnA·토론·루브릭."""


@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    from learning_analysis.edusignal_autoretrain import (
        start_edusignal_background_autoretrain,
        stop_edusignal_background_autoretrain,
    )

    await start_edusignal_background_autoretrain()
    yield
    await stop_edusignal_background_autoretrain()


app = FastAPI(
    title="팀 프로젝트 기여도 자동 평가 시스템",
    description=_APP_DESC,
    version="4.7.4",
    lifespan=_app_lifespan,
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
            "operations_privacy_doc": "docs/OPERATIONS_AND_LIMITS.md (저장소)",
            "completion_contract_doc": "docs/ABSOLUTE_COMPLETION.md (저장소) — 영구 완성 불가·태그 단위 완료 선언",
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
            "deep_learning_assist": {
                "label_ko": "딥러닝 기여도 보강(투명성)",
                "summary": "PyTorch 앙상블·불확실도·피처·블렌드·한계를 응답에 구조화",
                "evidence": [
                    "POST /api/team/report → dl_model_info.contest_transparency",
                    "edu_tools/team_unified_eval.py — _contest_transparency_pack",
                    "edu_tools/team_torch_model.py — 학습·추론·TEAM_TORCH_CONTEST_MAX(심사 최고 품질 프리셋)",
                    "learning_analysis/edusignal_autoretrain.py — 백그라운드 주기 재학습(새 샘플·드리프트 반영)",
                    "team_torch_model.meta.json → dl_quality_unified.contest_max_quality",
                    "dl_model_info.prior_calibration · TEAM_WEB_PRIORS_* (승인 JSON 벤치마크 보정)",
                    "프론트 팀 리포트 — 외부·벤치마크 priors 보정 패널",
                ],
            },
        },
        "ai_providers_configured": provider_keys_status(),
        "dl_roadmap": roadmap_payload(),
        "contest_submission_pack": contest_submission_pack(),
        "endpoints": {
            "capabilities": "GET /api/capabilities",
            "health": "GET /api/health",
            "version": "GET /api/version",
            "observability": "GET /api/observability",
            "ready": "GET /api/ready",
            "live": "GET /api/live",
            "openapi_json": "GET /openapi.json",
            "team_evaluate": "POST /api/team/evaluate",
            "team_report": "POST /api/team/report",
            "team_evaluate_compare": "POST /api/team/evaluate/compare",
            "analyze": "POST /api/analyze",
            "llm_compare": "POST /api/llm/compare",
            "perf_recent": "GET /api/perf/recent",
            "rubric_draft": "POST /api/rubric/draft",
            "team_model_limits_report": "GET /api/team/model/limits-report",
            "team_dataset_label_summary": "GET /api/team/data/dataset-label-summary",
            "team_training_backup": "POST /api/team/data/backup-artifacts",
            "team_benchmark_narrow": "POST /api/team/benchmark-narrow",
            "connection_advise": "POST /api/connection/advise",
        },
    }


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    if _post_rate_limit_should_block(request):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        resp = JSONResponse(
            status_code=429,
            content={
                "detail": "요청 한도를 초과했습니다. 잠시 후 다시 시도하세요.",
                "retry_after_sec": 60,
            },
        )
        resp.headers["X-Request-ID"] = rid
        resp.headers["Retry-After"] = "60"
        _apply_security_headers(resp)
        return resp
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
    _apply_security_headers(response)
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
app.include_router(team_unified_router, prefix="/api/team", tags=["team"])
app.include_router(at_risk_router, prefix="/api/at-risk", tags=["at-risk"])
app.include_router(feedback_router, prefix="/api/feedback", tags=["feedback"])
app.include_router(course_qa_router, prefix="/api/course", tags=["course"])
app.include_router(discussion_router, prefix="/api/discussion", tags=["discussion"])
app.include_router(rubric_align_router, prefix="/api/rubric", tags=["rubric"])
app.include_router(connection_dl_router, prefix="/api/connection", tags=["connection"])

_cors_kw: dict[str, Any] = {
    "allow_origins": _cors_origins(),
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
_rx = _cors_origin_regex()
if _rx:
    _cors_kw["allow_origin_regex"] = _rx
app.add_middleware(CORSMiddleware, **_cors_kw)
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


@app.get("/api/team/data/profile")
async def api_team_data_profile(response: Response):
    """팀 기여 데이터셋/모델런 로그 규모 확인용."""
    response.headers["Cache-Control"] = "private, max-age=5"
    return {"status": "ok", "profile": db_profile()}


@app.get("/api/team/data/trends")
async def api_team_data_trends(response: Response, days: int = 30, member_name: str = ""):
    """기간별 개인 기여도 추세(룰/ML/혼합) 조회."""
    response.headers["Cache-Control"] = "private, max-age=5"
    return {
        "status": "ok",
        "trends": contribution_trends(days=days, member_name=member_name or None),
    }


@app.post("/api/team/data/backup-artifacts")
async def api_team_training_backup(response: Response):
    """학습 데이터·선형·PyTorch 가중치를 `edu_tools/data/backups/`(또는 TEAM_DATA_BACKUP_ROOT)에 타임스탬프 폴더로 복사."""
    from edu_tools.team_training_backup import backup_team_training_artifacts

    response.headers["Cache-Control"] = "no-store"
    out = await asyncio.to_thread(partial(backup_team_training_artifacts, reason="api"))
    return {"status": "ok", **out}


@app.get("/api/team/data/dataset-label-summary")
async def api_team_dataset_label_summary(response: Response):
    """team_ml_dataset.jsonl 라벨(y) 분포 요약 — 수집·스케일 점검."""
    response.headers["Cache-Control"] = "private, max-age=15"
    # CI/로컬 환경 차이(데이터 파일 부재 등)에서도 응답 스키마를 고정한다.
    raw = dataset_label_streaming_stats()
    stats = dict(raw) if isinstance(raw, dict) else {}
    stats.setdefault("lines_scanned", 0)
    stats.setdefault("y_present", 0)
    stats.setdefault("dataset_path", "")
    return {"status": "ok", "stats": stats}


@app.get("/api/team/model/monitor")
async def api_team_model_monitor(response: Response, days: int = 30):
    """DL 보조 모델 운영 모니터링 요약."""
    response.headers["Cache-Control"] = "private, max-age=5"
    return {"status": "ok", "monitor": model_monitor(window_days=days)}


@app.get("/api/team/model/limits-report")
async def api_team_model_limits_report(response: Response, days: int = 30):
    """한계 1(정보·라벨)·한계 2(측정 품질) 요약 — 심사·운영 증빙용."""
    response.headers["Cache-Control"] = "private, max-age=10"
    return {"status": "ok", "report": build_limits_report(window_days=days)}


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


# --- SPA: Vite 빌드(frontend/dist)를 API와 동일 출처(예: :8000)로 제공 — 로컬은 npm start 한 주소만 열면 됨 ---
def _frontend_dist_dir() -> Path | None:
    raw = os.environ.get("FRONTEND_DIST", "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_dir() and (p / "index.html").is_file() else None
    # backend/learning_analysis/main.py → 저장소 루트
    repo = Path(__file__).resolve().parent.parent.parent
    p = repo / "frontend" / "dist"
    return p if p.is_dir() and (p / "index.html").is_file() else None


def _safe_file_in_dist(dist: Path, relative: str) -> Path | None:
    if not relative or relative.startswith("/"):
        return None
    dist_r = dist.resolve()
    candidate = (dist_r / relative).resolve()
    try:
        candidate.relative_to(dist_r)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


_spa_dist = _frontend_dist_dir()
if _spa_dist is not None:
    from starlette.staticfiles import StaticFiles

    _spa_assets = _spa_dist / "assets"
    if _spa_assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_spa_assets)), name="spa_assets")

    @app.get("/", include_in_schema=False)
    async def spa_root() -> FileResponse:
        return FileResponse(_spa_dist / "index.html")

    _spa_502 = _spa_dist / "502.html"
    if _spa_502.is_file():

        @app.get("/502.html", include_in_schema=False)
        async def spa_502() -> FileResponse:
            return FileResponse(_spa_502)

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def spa_history_fallback(spa_path: str) -> FileResponse:
        # /docs, /openapi.json, /api/* 등은 위에서 이미 처리됨. 여기까지 온 비파일 경로는 SPA로 넘김.
        if spa_path.startswith("api") or spa_path in ("docs", "redoc", "openapi.json"):
            raise HTTPException(status_code=404)
        hit = _safe_file_in_dist(_spa_dist, spa_path)
        if hit is not None:
            return FileResponse(hit)
        return FileResponse(_spa_dist / "index.html")
