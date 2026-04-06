"""팀 프로젝트 기여도 평가 — OpenAI 또는 휴리스틱."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class MemberIn(BaseModel):
    name: str = Field(..., min_length=1)
    role: str = ""
    commits: int | None = Field(None, ge=0)
    pull_requests: int | None = Field(None, ge=0)
    lines_changed: int | None = Field(None, ge=0)
    tasks_completed: int | None = Field(None, ge=0)
    meetings_attended: int | None = Field(None, ge=0)
    self_report: str = ""
    peer_notes: str = ""


class TeamEvaluateRequest(BaseModel):
    project_name: str = Field(..., min_length=1)
    project_description: str = ""
    evaluation_criteria: str = ""
    members: list[MemberIn] = Field(..., min_length=1)


class DimensionScores(BaseModel):
    technical: float
    collaboration: float
    initiative: float


class MemberOut(BaseModel):
    name: str
    role: str = ""
    contribution_index: float
    dimensions: DimensionScores
    evidence_summary: str = ""
    caveats: str = ""


class TeamEvaluateResponse(BaseModel):
    mode: str
    members: list[MemberOut]
    fairness_notes: str = ""
    disclaimer: str = (
        "자동 추정이며 최종 성적·인사 판단을 대체하지 않습니다. 팀 내 갈등 시 인간 중재가 필요합니다."
    )


def _parse_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        t = m.group(0)
    return json.loads(t)


def _heuristic(req: TeamEvaluateRequest) -> TeamEvaluateResponse:
    members = req.members
    n = len(members)

    def norm(vals: list[float]) -> list[float]:
        s = sum(vals) or 1.0
        return [v / s for v in vals]

    commits = [float(m.commits or 0) for m in members]
    tasks = [float(m.tasks_completed or 0) for m in members]
    lines = [float(m.lines_changed or 0) for m in members]
    prs = [float(m.pull_requests or 0) for m in members]
    meet = [float(m.meetings_attended or 0) for m in members]
    words = [max(0, len((m.self_report or "").split())) for m in members]

    nc, nt, nl, npr, nw, nmeet = (
        norm(commits),
        norm(tasks),
        norm(lines),
        norm(prs),
        norm([float(w) for w in words]),
        norm(meet),
    )

    raw_scores: list[float] = []
    for i in range(n):
        blend = 0.22 * nc[i] + 0.18 * nt[i] + 0.18 * nl[i] + 0.12 * npr[i] + 0.30 * nw[i]
        raw_scores.append(blend)
    raw_sum = sum(raw_scores) or 1.0

    out: list[MemberOut] = []
    for i, m in enumerate(members):
        idx = 100.0 * (raw_scores[i] / raw_sum)
        q = nc[i] + nl[i] + npr[i] + 1e-6
        tech = min(100.0, 30.0 + 70.0 * (0.45 * nc[i] + 0.35 * nl[i] + 0.20 * npr[i]) / q)
        collab = min(100.0, 35.0 + 45.0 * nw[i] + 20.0 * nmeet[i])
        init_ = min(100.0, 32.0 + 40.0 * nt[i] + 28.0 * npr[i])
        out.append(
            MemberOut(
                name=m.name,
                role=m.role,
                contribution_index=round(idx, 1),
                dimensions=DimensionScores(
                    technical=round(tech, 1),
                    collaboration=round(collab, 1),
                    initiative=round(init_, 1),
                ),
                evidence_summary="정량·자기서술 가중 합성입니다. 문서·리뷰 기여는 수치에 약합니다.",
                caveats="OpenAI 키가 없어 휴리스틱만 사용했습니다.",
            )
        )

    note = "정량 지표는 참고용입니다."
    if n > 1 and commits and max(commits) > 0 and max(commits) / (sum(commits) / n) > 2.5:
        note += " 커밋 편차가 큽니다. 리뷰·기획 기여를 확인하세요."

    return TeamEvaluateResponse(mode="heuristic", members=out, fairness_notes=note)


def _openai_eval(req: TeamEvaluateRequest, api_key: str) -> TeamEvaluateResponse:
    from openai import OpenAI

    payload = {
        "project_name": req.project_name,
        "project_description": req.project_description,
        "evaluation_criteria": req.evaluation_criteria,
        "members": [m.model_dump() for m in req.members],
    }
    sys = """교육용 팀 기여도 조교입니다. JSON만 출력하세요.
{
  "fairness_notes": "문자열",
  "members": [
    {
      "name": "이름",
      "role": "역할",
      "contribution_index": 0-100,
      "dimensions": {"technical":0-100,"collaboration":0-100,"initiative":0-100},
      "evidence_summary": "근거",
      "caveats": "주의"
    }
  ]
}
커밋 수만으로 순위 매기지 말고 self_report·peer_notes로 비가시 기여를 반영하세요."""

    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.25,
    )
    data = _parse_json(res.choices[0].message.content or "{}")
    rows = {str(r.get("name", "")).strip(): r for r in (data.get("members") or []) if isinstance(r, dict)}
    out: list[MemberOut] = []
    for m in req.members:
        raw = rows.get(m.name.strip()) or rows.get(m.name)
        if raw is None and len(data.get("members") or []) == len(req.members):
            raw = (data.get("members") or [])[len(out)]
        if not isinstance(raw, dict):
            return _heuristic(req)
        d = raw.get("dimensions") or {}
        out.append(
            MemberOut(
                name=str(raw.get("name") or m.name),
                role=str(raw.get("role") or m.role),
                contribution_index=max(0, min(100, float(raw.get("contribution_index", 50)))),
                dimensions=DimensionScores(
                    technical=max(0, min(100, float(d.get("technical", 50)))),
                    collaboration=max(0, min(100, float(d.get("collaboration", 50)))),
                    initiative=max(0, min(100, float(d.get("initiative", 50)))),
                ),
                evidence_summary=str(raw.get("evidence_summary", ""))[:2000],
                caveats=str(raw.get("caveats", ""))[:1000],
            )
        )
    if len(out) != len(req.members):
        return _heuristic(req)
    return TeamEvaluateResponse(
        mode="ai",
        members=out,
        fairness_notes=str(data.get("fairness_notes", ""))[:2000],
    )


@router.post("/evaluate", response_model=TeamEvaluateResponse)
async def evaluate_team(body: TeamEvaluateRequest) -> TeamEvaluateResponse:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            return _openai_eval(body, key)
        except Exception:
            return _heuristic(body)
    return _heuristic(body)
