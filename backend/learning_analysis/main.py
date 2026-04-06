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

from learning_analysis.pipeline import analyze_async, provider_keys_status
from learning_analysis.schemas import AnalyzeRequest, AnalyzeResponse


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").strip()
    if not raw:
        return ["http://127.0.0.1:5173", "http://localhost:5173"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="Learning-Exam Consistency Analyzer",
    description="Gemini, ChatGPT, Claude, Grok 병렬 분석 — 부정행위 의심·학습 상태·미래 예측(보조)",
    version="2.0.0",
)

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
