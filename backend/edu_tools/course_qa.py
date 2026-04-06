"""강의 안내·실라버스 기반 질의 응답 초안 — 반복 질문·문서 탐색 부담 완화."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class CourseAskRequest(BaseModel):
    course_name: str = ""
    syllabus_text: str = Field(..., min_length=20, description="강의계획서·안내 전문(일부)")
    question: str = Field(..., min_length=2)


class CourseAskResponse(BaseModel):
    mode: str
    answer_draft: str
    citations: list[str] = Field(default_factory=list)
    caveats: str = (
        "제공한 안내 문구만 근거로 합니다. 공지·정책은 최종적으로 LMS·학과 기준을 확인하세요."
    )


def _tokenize(s: str) -> set[str]:
    s = re.sub(r"[^\w\s가-힣]", " ", s.lower())
    return {w for w in s.split() if len(w) >= 2}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def _heuristic(req: CourseAskRequest) -> CourseAskResponse:
    q_tokens = _tokenize(req.question)
    if not q_tokens:
        return CourseAskResponse(
            mode="heuristic",
            answer_draft="질문에서 검색할 단어가 부족합니다. 핵심 키워드를 넣어 다시 시도하세요.",
            citations=[],
        )
    scored: list[tuple[float, str]] = []
    for sent in _sentences(req.syllabus_text):
        st = _tokenize(sent)
        if not st:
            continue
        overlap = len(q_tokens & st) / max(1, len(q_tokens))
        scored.append((overlap, sent))
    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:4] if s]
    if not top:
        top = _sentences(req.syllabus_text)[:3]
    body = (
        "아래는 안내 문구에서 질문과 단어가 겹치는 문장을 발췌한 것입니다. "
        "OpenAI 키가 없어 키워드 매칭만 사용했습니다.\n\n"
        + "\n".join(f"· {t}" for t in top[:3])
    )
    return CourseAskResponse(mode="heuristic", answer_draft=body, citations=top[:5])


def _openai_ask(req: CourseAskRequest, api_key: str) -> CourseAskResponse:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    sys = """당신은 대학 조교입니다. 제공된 강의 안내(실라버스 일부)만 근거로 학생 질문에 답하는 초안을 씁니다.
JSON만 출력하세요.
{
  "answer_draft": "한국어, 2~5문단 이내",
  "citations": ["안내에서 직접 인용한 짧은 문장들"]
}
안내에 없는 내용은 추측하지 말고 '안내에 명시되지 않음 — 담당 교수에게 확인'이라고 하세요."""

    user = json.dumps(
        {
            "course_name": req.course_name,
            "syllabus_excerpt": req.syllabus_text[:12000],
            "question": req.question,
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
    return CourseAskResponse(
        mode="ai",
        answer_draft=str(data.get("answer_draft", ""))[:8000],
        citations=[str(x) for x in (data.get("citations") or [])][:12],
    )


@router.post("/ask", response_model=CourseAskResponse)
async def course_ask(body: CourseAskRequest) -> CourseAskResponse:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            return _openai_ask(body, key)
        except Exception:
            return _heuristic(body)
    return _heuristic(body)
