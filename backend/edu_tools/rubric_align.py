"""채점 근거와 루브릭 정합성 점검 — 공정성·설명 가능성 보조."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class RubricAlignRequest(BaseModel):
    rubric: str = Field(..., min_length=1)
    grader_rationale: str = Field(..., min_length=1, description="채점자가 남긴 코멘트·근거")
    student_work_excerpt: str = Field("", description="학생 답안 일부(선택)")


class RubricAlignResponse(BaseModel):
    mode: str
    alignment_score: float = Field(..., ge=0, le=100, description="루브릭과 근거의 정합성 0~100")
    matched_rubric_points: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    suggestions: str = ""
    disclaimer: str = "자동 점검이며, 이의·재채점은 기관 규정에 따릅니다."


def _tokenize(s: str) -> set[str]:
    s = re.sub(r"[^\w\s가-힣]", " ", s.lower())
    return {w for w in s.split() if len(w) >= 2}


def _heuristic(req: RubricAlignRequest) -> RubricAlignResponse:
    rt = _tokenize(req.rubric)
    gt = _tokenize(req.grader_rationale)
    if not rt or not gt:
        return RubricAlignResponse(
            mode="heuristic",
            alignment_score=40.0,
            matched_rubric_points=[],
            gaps=["루브릭 또는 채점 근거가 너무 짧아 비교가 어렵습니다."],
            suggestions="루브릭 항목명·배점을 근거에 반복해 언급했는지 확인하세요.",
        )
    inter = len(rt & gt)
    union = len(rt | gt)
    jacc = inter / union if union else 0.0
    score = 35.0 + 65.0 * jacc
    # 보너스: 숫자/배점 언급
    if re.search(r"\d+\s*점|\d+%|배점", req.grader_rationale):
        score = min(100.0, score + 8.0)
    return RubricAlignResponse(
        mode="heuristic",
        alignment_score=round(score, 1),
        matched_rubric_points=[f"공통 어휘 약 {inter}개 (루브릭 대비 단순 매칭)"],
        gaps=["휴리스틱은 단어 겹침만 봅니다. 논리적 합치는 OpenAI 검토를 권장합니다."],
        suggestions="각 루브릭 항목별로 어떤 근거로 점수를 부여했는지 한 줄씩 매핑해 보세요.",
    )


def _openai_align(req: RubricAlignRequest, api_key: str) -> RubricAlignResponse:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    sys = """교육 평가 감사 조교입니다. 채점 근거가 루브릭과 얼마나 맞는지 JSON만 출력하세요.
{
  "alignment_score": 0-100,
  "matched_rubric_points": ["잘 연결된 루브릭 요소"],
  "gaps": ["근거가 부족하거나 불명확한 부분"],
  "suggestions": "채점자에게 도움이 되는 한국어 문단"
}
학생을 비난하지 말고, 공정성·설명 가능성 관점에서 작성하세요."""

    user = json.dumps(
        {
            "rubric": req.rubric[:12000],
            "grader_rationale": req.grader_rationale[:8000],
            "student_work_excerpt": (req.student_work_excerpt or "")[:6000],
        },
        ensure_ascii=False,
    )
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = (res.choices[0].message.content or "{}").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        raw = m.group(0)
    data: dict[str, Any] = json.loads(raw)
    return RubricAlignResponse(
        mode="ai",
        alignment_score=max(0, min(100, float(data.get("alignment_score", 0)))),
        matched_rubric_points=[str(x) for x in (data.get("matched_rubric_points") or [])][:12],
        gaps=[str(x) for x in (data.get("gaps") or [])][:12],
        suggestions=str(data.get("suggestions", ""))[:2000],
    )


@router.post("/check", response_model=RubricAlignResponse)
async def check_rubric_align(body: RubricAlignRequest) -> RubricAlignResponse:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            return _openai_align(body, key)
        except Exception:
            return _heuristic(body)
    return _heuristic(body)
