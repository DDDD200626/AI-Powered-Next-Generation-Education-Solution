"""채점 근거와 루브릭 정합성 점검 — 공정성·설명 가능성 보조."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from learning_analysis.llm_clients import get_openai_client

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
    client = get_openai_client(api_key)
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


class RubricCriterionOut(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field("", description="평가 기준 설명")
    weight_percent: float = Field(..., ge=0, le=100)


class RubricGenerateRequest(BaseModel):
    learning_objectives: str = Field(..., min_length=10, max_length=12000, description="학습 목표·성과 목표 문장")
    course_name: str = Field("", max_length=200)
    assignment_type: str = Field("", max_length=120, description="예: 팀 프로젝트, 보고서, 시험")
    max_criteria: int = Field(5, ge=3, le=8, description="루브릭 항목 수")


class RubricGenerateResponse(BaseModel):
    mode: str
    rubric_markdown: str
    criteria: list[RubricCriterionOut]
    disclaimer: str = "교수·기관 검수 후 사용하세요. 배점·항목은 과목에 맞게 수정해야 합니다."


def _normalize_criteria_weights(items: list[tuple[str, str]]) -> list[RubricCriterionOut]:
    n = len(items)
    if n == 0:
        return []
    base = round(100.0 / n, 2)
    weights = [base] * n
    diff = round(100.0 - sum(weights), 2)
    weights[-1] = round(weights[-1] + diff, 2)
    return [
        RubricCriterionOut(name=name, description=desc, weight_percent=w)
        for (name, desc), w in zip(items, weights, strict=True)
    ]


def _split_learning_goals(text: str, max_n: int) -> list[str]:
    raw = re.split(r"[\n;]+|•|·", text)
    goals: list[str] = []
    for line in raw:
        s = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if len(s) >= 4:
            goals.append(s[:500])
    extras = [
        "학습 목표에 대한 이해·설명",
        "개념·도구를 과제 맥락에 적용",
        "논리적 구성·근거 제시",
        "협업·피드백 반영 (해당 시)",
        "형식·제출·윤리 기준 준수",
        "문제 정의·대안 비교",
        "결과의 타당성·한계 인식",
    ]
    for e in extras:
        if len(goals) >= max_n:
            break
        if e not in goals:
            goals.append(e)
    return goals[:max_n]


def _heuristic_generate(req: RubricGenerateRequest) -> RubricGenerateResponse:
    goals = _split_learning_goals(req.learning_objectives, req.max_criteria)
    pairs: list[tuple[str, str]] = []
    for i, g in enumerate(goals):
        pairs.append(
            (
                f"항목 {i + 1}: {g[:80]}{'…' if len(g) > 80 else ''}",
                f"학습 목표에 명시된 내용을 반영했는지, 수행 수준(미흡·보통·우수)을 구분할 수 있는 근거를 둡니다. 원문: {g}",
            )
        )
    criteria = _normalize_criteria_weights(pairs)
    course = (req.course_name or "과목").strip()
    atype = (req.assignment_type or "과제").strip()
    lines = [
        f"# {course} — 평가 루브릭 (초안, {atype})",
        "",
        f"| 항목 | 기준 요약 | 배점 |",
        f"| --- | --- | --- |",
    ]
    for c in criteria:
        lines.append(f"| {c.name} | {c.description[:200]}{'…' if len(c.description) > 200 else ''} | {c.weight_percent:.0f}% |")
    lines.extend(
        [
            "",
            "## 사용 안내",
            "- 휴리스틱 초안입니다. 항목명·배점·설명은 반드시 수정하세요.",
            "- OpenAI 키가 있으면 서술 품질이 나은 초안이 생성됩니다.",
        ]
    )
    return RubricGenerateResponse(
        mode="heuristic",
        rubric_markdown="\n".join(lines),
        criteria=criteria,
    )


def _openai_generate(req: RubricGenerateRequest, api_key: str) -> RubricGenerateResponse:
    client = get_openai_client(api_key)
    sys = """교육 평가 설계 조교입니다. 학습 목표를 바탕으로 평가 루브릭 초안을 JSON만 출력하세요.
{
  "criteria": [
    {"name": "짧은 항목 제목(한국어)", "description": "상·중·하를 구분할 수 있는 평가 기준 설명", "weight_percent": 0-100}
  ],
  "rubric_markdown": "마크다운 표 또는 목록으로 정리한 루브릭 전문(한국어)"
}
- criteria 개수는 사용자가 요청한 범위 내(보통 3~8개).
- weight_percent 합이 정확히 100이 되도록 하세요.
- 비난·차별적 표현이 없게 하세요."""

    user = json.dumps(
        {
            "course_name": req.course_name,
            "assignment_type": req.assignment_type,
            "max_criteria": req.max_criteria,
            "learning_objectives": req.learning_objectives[:12000],
        },
        ensure_ascii=False,
    )
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.35,
    )
    raw = (res.choices[0].message.content or "{}").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        raw = m.group(0)
    data: dict[str, Any] = json.loads(raw)
    crit_raw = data.get("criteria") or []
    out: list[RubricCriterionOut] = []
    for x in crit_raw[: req.max_criteria]:
        if not isinstance(x, dict):
            continue
        name = str(x.get("name", "")).strip()
        if not name:
            continue
        desc = str(x.get("description", "")).strip()[:2000]
        try:
            w = float(x.get("weight_percent", 0))
        except (TypeError, ValueError):
            w = 0.0
        w = max(0.0, min(100.0, w))
        out.append(RubricCriterionOut(name=name[:200], description=desc, weight_percent=w))
    if len(out) < 3:
        return _heuristic_generate(req)
    total = sum(c.weight_percent for c in out)
    if total <= 0:
        merged = [(c.name, c.description) for c in out]
        out = _normalize_criteria_weights(merged)
    else:
        scale = 100.0 / total
        out = [
            RubricCriterionOut(
                name=c.name,
                description=c.description,
                weight_percent=round(c.weight_percent * scale, 2),
            )
            for c in out
        ]
        fix = 100.0 - sum(c.weight_percent for c in out)
        if out and abs(fix) > 0.001:
            last = out[-1]
            out[-1] = RubricCriterionOut(
                name=last.name,
                description=last.description,
                weight_percent=round(last.weight_percent + fix, 2),
            )
    md = str(data.get("rubric_markdown", "")).strip()[:20000]
    if not md:
        md = _heuristic_generate(req).rubric_markdown
    return RubricGenerateResponse(mode="ai", rubric_markdown=md, criteria=out)


@router.post("/draft", response_model=RubricGenerateResponse)
async def draft_rubric_from_objectives(body: RubricGenerateRequest) -> RubricGenerateResponse:
    """학습 목표·과제 유형을 넣으면 루브릭 초안(항목·배점·마크다운)을 생성합니다."""
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            return _openai_generate(body, key)
        except Exception:
            return _heuristic_generate(body)
    return _heuristic_generate(body)


@router.post("/check", response_model=RubricAlignResponse)
async def check_rubric_align(body: RubricAlignRequest) -> RubricAlignResponse:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            return _openai_align(body, key)
        except Exception:
            return _heuristic(body)
    return _heuristic(body)
