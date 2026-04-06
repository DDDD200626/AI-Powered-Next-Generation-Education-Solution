"""
ClassPulse 백엔드 — FastAPI JSON API 전용 (Jinja/Streamlit 없음).
프론트엔드는 별도 `frontend/` (Vite)에서 호출합니다.

로컬 백엔드: uvicorn classpulse.api_app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from classpulse.comprehension import assess_comprehension
from classpulse.insights import (
    default_integrated_learner_rows,
    default_sample_metrics,
    integrated_df_is_valid,
    integrated_summary_stats,
    metrics_df_is_valid,
    metrics_summary_stats,
    parse_csv_text,
    summarize_for_admin,
    summarize_integrated_dashboard,
)
from classpulse.integrity import (
    answer_corpus_max_similarity,
    assess_integrity_llm,
    heuristic_integrity_signals,
)
from classpulse.rag import build_index, format_context_for_prompt, generate_answer, load_corpus, retrieve
from classpulse.teacher_feedback import generate_feedback_draft


def _api_key() -> str:
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def _openai_error_message(err: BaseException) -> str:
    try:
        from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
    except ImportError:
        return str(err)
    if isinstance(err, AuthenticationError):
        return (
            "API 키가 잘못되었거나 OpenAI 키가 아닙니다. 서버의 OPENAI_API_KEY(또는 배포 플랫폼 Secrets)를 확인하세요."
        )
    if isinstance(err, RateLimitError):
        return "요청 한도에 걸렸습니다. 잠시 후 다시 시도하세요."
    if isinstance(err, APIConnectionError):
        return "OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인하세요."
    if isinstance(err, APIStatusError):
        return f"OpenAI API 오류: {getattr(err, 'message', err)}"
    return str(err)


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").strip()
    if not raw:
        return ["http://127.0.0.1:5173", "http://localhost:5173"]
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks = load_corpus()
    vectorizer, matrix = build_index(chunks)
    app.state.chunks = chunks
    app.state.vectorizer = vectorizer
    app.state.matrix = matrix
    yield


app = FastAPI(
    title="ClassPulse API",
    description="학습 이해도·무결성 보조·교육 운영 — 백엔드 JSON API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TopKMixin(BaseModel):
    top_k: int = Field(4, ge=2, le=8)


class RetrieveBody(TopKMixin):
    query: str


class LearnBody(TopKMixin):
    query: str


class ComprehensionBody(TopKMixin):
    question: str
    answer: str


class IntegrityBody(BaseModel):
    assignment_prompt: str = ""
    submission: str


class TeacherBody(BaseModel):
    rubric: str
    submission: str


class CsvOrSampleBody(BaseModel):
    csv_text: str = ""


@app.get("/api/health")
async def health():
    return {"status": "ok", "openai_configured": bool(_api_key())}


@app.post("/api/retrieve")
async def api_retrieve(request: Request, body: RetrieveBody):
    chunks = request.app.state.chunks
    vectorizer = request.app.state.vectorizer
    matrix = request.app.state.matrix
    hits = retrieve(body.query.strip(), chunks, vectorizer, matrix, top_k=body.top_k)
    return {
        "hits": [
            {"chunk_id": ch.chunk_id, "source": ch.source, "text": ch.text, "score": round(score, 4)}
            for ch, score in hits
        ]
    }


@app.post("/api/learn")
async def api_learn(request: Request, body: LearnBody):
    chunks = request.app.state.chunks
    vectorizer = request.app.state.vectorizer
    matrix = request.app.state.matrix
    key = _api_key()
    hits = retrieve(body.query.strip(), chunks, vectorizer, matrix, top_k=body.top_k)
    out: dict = {
        "hits": [
            {"chunk_id": ch.chunk_id, "source": ch.source, "text": ch.text, "score": round(score, 4)}
            for ch, score in hits
        ],
        "answer": None,
        "answer_error": None,
    }
    if not key:
        out["answer_error"] = "서버에 OPENAI_API_KEY가 설정되지 않아 답변을 생성할 수 없습니다."
        return out
    try:
        out["answer"] = generate_answer(body.query.strip(), hits, key)
    except Exception as e:
        out["answer_error"] = _openai_error_message(e)
    return out


@app.post("/api/comprehension")
async def api_comprehension(request: Request, body: ComprehensionBody):
    if not body.question.strip() or not body.answer.strip():
        raise HTTPException(status_code=400, detail="question과 answer는 필수입니다.")
    chunks = request.app.state.chunks
    vectorizer = request.app.state.vectorizer
    matrix = request.app.state.matrix
    key = _api_key()
    q_for_ret = f"{body.question.strip()}\n\n학생 답변 요약 키워드: {body.answer.strip()[:400]}"
    hits = retrieve(q_for_ret, chunks, vectorizer, matrix, top_k=body.top_k)
    result: dict = {
        "hits": [
            {"chunk_id": ch.chunk_id, "source": ch.source, "text": ch.text, "score": round(score, 4)}
            for ch, score in hits
        ],
        "report": None,
        "report_error": None,
    }
    if not key:
        result["report_error"] = "서버에 OPENAI_API_KEY가 설정되지 않았습니다."
        return result
    try:
        result["report"] = assess_comprehension(body.question.strip(), body.answer.strip(), hits, key)
    except Exception as e:
        result["report_error"] = _openai_error_message(e)
    return result


@app.post("/api/integrity")
async def api_integrity(request: Request, body: IntegrityBody):
    if not body.submission.strip():
        raise HTTPException(status_code=400, detail="submission은 필수입니다.")
    chunks = request.app.state.chunks
    vectorizer = request.app.state.vectorizer
    matrix = request.app.state.matrix
    key = _api_key()
    sig = heuristic_integrity_signals(body.submission)
    sim, src = answer_corpus_max_similarity(body.submission, chunks, vectorizer, matrix)
    hits_i = retrieve(body.submission.strip()[:2000], chunks, vectorizer, matrix, top_k=3)
    excerpt = format_context_for_prompt(hits_i) if hits_i else ""
    out: dict = {
        "heuristic": sig,
        "max_similarity": round(sim, 4),
        "max_similarity_source": src,
        "corpus_excerpt_preview": excerpt[:4000] if excerpt else "",
        "narrative": None,
        "narrative_error": None,
    }
    if not key:
        out["narrative_error"] = "서버에 OPENAI_API_KEY가 설정되지 않았습니다."
        return out
    try:
        out["narrative"] = assess_integrity_llm(
            body.assignment_prompt,
            body.submission,
            key,
            corpus_excerpt=excerpt or None,
        )
    except Exception as e:
        out["narrative_error"] = _openai_error_message(e)
    return out


@app.post("/api/teacher-feedback")
async def api_teacher_feedback(body: TeacherBody):
    if not body.submission.strip():
        raise HTTPException(status_code=400, detail="submission은 필수입니다.")
    key = _api_key()
    if not key:
        raise HTTPException(status_code=503, detail="서버에 OPENAI_API_KEY가 설정되지 않았습니다.")
    try:
        draft = generate_feedback_draft(body.submission.strip(), body.rubric, key)
        return {"draft": draft}
    except Exception as e:
        raise HTTPException(status_code=502, detail=_openai_error_message(e)) from e


def _df_from_dashboard_body(body: CsvOrSampleBody) -> tuple[pd.DataFrame, str | None]:
    raw = (body.csv_text or "").strip()
    if raw:
        df = parse_csv_text(raw)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="CSV 파싱에 실패했습니다.")
        if not integrated_df_is_valid(df):
            raise HTTPException(
                status_code=400,
                detail="필수 열: student_id, comprehension_score, integrity_risk_band",
            )
        return df, None
    return default_integrated_learner_rows(), "샘플 데이터를 사용합니다."


def _df_from_ops_body(body: CsvOrSampleBody) -> tuple[pd.DataFrame, str | None]:
    raw = (body.csv_text or "").strip()
    if raw:
        df = parse_csv_text(raw)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="CSV 파싱에 실패했습니다.")
        if not metrics_df_is_valid(df):
            raise HTTPException(
                status_code=400,
                detail="필수 열: assignment_submit_rate, late_submissions, forum_questions",
            )
        return df, None
    return default_sample_metrics(), "샘플 데이터를 사용합니다."


@app.post("/api/dashboard/preview")
async def api_dashboard_preview(body: CsvOrSampleBody):
    df, note = _df_from_dashboard_body(body)
    stats = integrated_summary_stats(df)
    return {
        "rows": df.fillna("").to_dict(orient="records"),
        "columns": list(df.columns),
        "stats": stats,
        "note": note,
    }


@app.post("/api/dashboard/summarize")
async def api_dashboard_summarize(body: CsvOrSampleBody):
    key = _api_key()
    if not key:
        raise HTTPException(status_code=503, detail="서버에 OPENAI_API_KEY가 설정되지 않았습니다.")
    df, _note = _df_from_dashboard_body(body)
    try:
        summary = summarize_integrated_dashboard(df, key)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=502, detail=_openai_error_message(e)) from e


@app.post("/api/ops/preview")
async def api_ops_preview(body: CsvOrSampleBody):
    df, note = _df_from_ops_body(body)
    stats = metrics_summary_stats(df)
    return {
        "rows": df.fillna("").to_dict(orient="records"),
        "columns": list(df.columns),
        "stats": stats,
        "note": note,
    }


@app.post("/api/ops/summarize")
async def api_ops_summarize(body: CsvOrSampleBody):
    key = _api_key()
    if not key:
        raise HTTPException(status_code=503, detail="서버에 OPENAI_API_KEY가 설정되지 않았습니다.")
    df, _note = _df_from_ops_body(body)
    try:
        summary = summarize_for_admin(df, key)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=502, detail=_openai_error_message(e)) from e


# API만 쓸 때는 루트 JSON, 통합 배포 시 FRONTEND_DIST 에 Vite 빌드(dist) 경로
_frontend_dist = os.environ.get("FRONTEND_DIST", "").strip()
if _frontend_dist and Path(_frontend_dist).resolve().is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(Path(_frontend_dist).resolve()), html=True),
        name="frontend",
    )
else:

    @app.get("/")
    async def root():
        return {
            "service": "ClassPulse API",
            "health": "/api/health",
            "docs": "/docs",
            "hint": "프론트: frontend/ 에서 npm run dev — 또는 FRONTEND_DIST 로 빌드 정적 서빙",
        }
