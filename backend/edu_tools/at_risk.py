"""학습 이탈·위험 조기 신호 — 주차별 참여 지표."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class WeekPoint(BaseModel):
    week_label: str = Field(..., description="예: 3주차")
    engagement: float = Field(..., ge=0, le=100, description="참여·과제·출석 등 종합 0-100")
    assessment_score: float | None = Field(None, ge=0, le=100, description="소평가·퀴즈 평균 등")


class AtRiskRequest(BaseModel):
    course_name: str = ""
    student_label: str = ""
    weeks: list[WeekPoint] = Field(..., min_length=1)
    notes: str = ""


class AtRiskResponse(BaseModel):
    mode: str
    dropout_risk: float = Field(..., ge=0, le=100, description="이탈·위험 지수 0-100")
    trend_summary: str = ""
    signals: list[str] = Field(default_factory=list)
    intervention_suggestions: str = ""
    disclaimer: str = "조기 경보는 보조입니다. 실제 상담·지원은 기관 절차에 따릅니다."


def _parse_json(text: str) -> dict[str, Any]:
    t = text.strip()
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    return json.loads(t)


def _heuristic(req: AtRiskRequest) -> AtRiskResponse:
    ws = sorted(req.weeks, key=lambda x: x.week_label)
    eng = [w.engagement for w in ws]
    risk = 25.0
    signals: list[str] = []

    if len(eng) >= 2:
        drop = eng[0] - eng[-1]
        if drop > 25:
            risk += min(40.0, drop * 0.8)
            signals.append(f"참여 지표가 {drop:.0f}p 이상 하락했습니다.")
        if eng[-1] < 40:
            risk += 25.0
            signals.append("최근 주차 참여가 40 미만입니다.")

    avg = sum(eng) / len(eng)
    if avg < 45:
        risk += 15.0
        signals.append("전체 평균 참여가 낮습니다.")

    risk = max(0.0, min(100.0, risk + (50 - avg) * 0.2))

    if not signals:
        signals.append("급격한 하락 패턴은 없습니다. 지속 관찰을 권장합니다.")

    return AtRiskResponse(
        mode="heuristic",
        dropout_risk=round(risk, 1),
        trend_summary=f"주차 수 {len(ws)}. 최근 참여 {eng[-1]:.1f}, 평균 {avg:.1f}.",
        signals=signals,
        intervention_suggestions="학습 코치·조교 면담, 과제 분할·마일스톤 안내를 검토하세요.",
    )


def _openai_risk(req: AtRiskRequest, api_key: str) -> AtRiskResponse:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    sys = """학습 이탈 조기 경보용 JSON만 출력하세요.
{
  "dropout_risk": 0-100,
  "trend_summary": "한국어",
  "signals": ["문자열 배열"],
  "intervention_suggestions": "한국어"
}
단정하지 말고 교육적 관점에서 작성하세요."""

    user = json.dumps(
        {
            "course_name": req.course_name,
            "student_label": req.student_label,
            "weeks": [w.model_dump() for w in req.weeks],
            "notes": req.notes,
        },
        ensure_ascii=False,
    )
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.25,
    )
    data = _parse_json(res.choices[0].message.content or "{}")
    return AtRiskResponse(
        mode="ai",
        dropout_risk=max(0, min(100, float(data.get("dropout_risk", 40)))),
        trend_summary=str(data.get("trend_summary", ""))[:2000],
        signals=[str(x) for x in (data.get("signals") or [])][:12],
        intervention_suggestions=str(data.get("intervention_suggestions", ""))[:2000],
    )


@router.post("/evaluate", response_model=AtRiskResponse)
async def evaluate_at_risk(body: AtRiskRequest) -> AtRiskResponse:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            return _openai_risk(body, key)
        except Exception:
            return _heuristic(body)
    return _heuristic(body)
