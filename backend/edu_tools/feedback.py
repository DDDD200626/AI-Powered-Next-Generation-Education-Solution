"""과제 피드백 초안 생성 — 루브릭·제출물 기반."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from learning_analysis.llm_clients import get_openai_client

router = APIRouter()


class FeedbackRequest(BaseModel):
    rubric: str = Field(..., min_length=1, description="채점 기준·루브릭")
    assignment_prompt: str = Field("", description="과제 설명")
    submission: str = Field(..., min_length=1, description="학생 제출물")


class FeedbackResponse(BaseModel):
    mode: str
    draft_feedback: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    disclaimer: str = "교사 검수 후 전달하세요. AI 초안은 오류가 있을 수 있습니다."


@router.post("/draft", response_model=FeedbackResponse)
async def draft_feedback(body: FeedbackRequest) -> FeedbackResponse:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="피드백 초안 생성에는 OPENAI_API_KEY가 필요합니다.",
        )
    client = get_openai_client(key)
    sys = """당신은 대학 조교입니다. 루브릭에 맞춰 과제 피드백 초안을 한국어로 작성합니다.
JSON만 출력하세요.
{
  "draft_feedback": "전체 피드백 문단",
  "strengths": ["강점 2~4개"],
  "improvements": ["개선점 2~4개"]
}
비난하지 말고 구체적으로 작성하세요."""

    user = f"루브릭:\n{body.rubric}\n\n과제:\n{body.assignment_prompt}\n\n제출:\n{body.submission}"
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.35,
    )
    import json
    import re

    text = (res.choices[0].message.content or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    data = json.loads(text)
    return FeedbackResponse(
        mode="ai",
        draft_feedback=str(data.get("draft_feedback", ""))[:8000],
        strengths=[str(x) for x in (data.get("strengths") or [])][:8],
        improvements=[str(x) for x in (data.get("improvements") or [])][:8],
    )
