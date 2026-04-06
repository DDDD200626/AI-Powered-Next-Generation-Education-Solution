"""팀 기여도 평가 API."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from team_eval.evaluator import evaluate
from team_eval.schemas import TeamEvaluateRequest, TeamEvaluateResponse


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").strip()
    if not raw:
        return ["http://127.0.0.1:5173", "http://localhost:5173"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="Team Contribution Evaluator",
    description="AI·휴리스틱 기반 팀 프로젝트 기여도 자동 평가",
    version="1.0.0",
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
    has_key = bool((os.environ.get("OPENAI_API_KEY") or "").strip())
    return {"status": "ok", "openai_configured": has_key}


@app.post("/api/evaluate", response_model=TeamEvaluateResponse)
async def api_evaluate(body: TeamEvaluateRequest) -> TeamEvaluateResponse:
    return evaluate(body)
