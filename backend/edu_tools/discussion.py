"""토론·포럼 스레드 요약 — 대규모 수업에서 논의 파악 부담 완화."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from learning_analysis.llm_clients import get_openai_client

router = APIRouter()


class PostIn(BaseModel):
    author_label: str = Field("", description="익명 라벨")
    text: str = Field(..., min_length=1)


class DiscussionSynthesizeRequest(BaseModel):
    thread_title: str = ""
    posts: list[PostIn] = Field(..., min_length=1)


class DiscussionSynthesizeResponse(BaseModel):
    mode: str
    summary: str
    themes: list[str] = Field(default_factory=list)
    participation_notes: str = ""
    suggested_followups: list[str] = Field(default_factory=list)
    disclaimer: str = "요약은 참고용이며, 평가·갈등 조정은 인간이 합니다."


def _heuristic(req: DiscussionSynthesizeRequest) -> DiscussionSynthesizeResponse:
    by_author: dict[str, int] = {}
    all_text: list[str] = []
    for p in req.posts:
        label = (p.author_label or "익명").strip() or "익명"
        by_author[label] = by_author.get(label, 0) + len(p.text)
        all_text.append(p.text)
    blob = " ".join(all_text)
    tokens = re.findall(r"[가-힣]{2,}|[a-zA-Z]{3,}", blob.lower())
    common = [w for w, _ in Counter(tokens).most_common(8)]
    themes = [f"자주 등장한 표현: {', '.join(common[:5])}"] if common else ["주제 키워드를 추출하지 못했습니다."]
    parts = [f"{k}: 약 {v}자" for k, v in sorted(by_author.items(), key=lambda x: -x[1])[:6]]
    return DiscussionSynthesizeResponse(
        mode="heuristic",
        summary=f"게시 {len(req.posts)}건. 참여량(글자 수 기준): " + " · ".join(parts),
        themes=themes,
        participation_notes="OpenAI 키가 없어 키워드 빈도만 집계했습니다. 논지·갈등은 직접 확인하세요.",
        suggested_followups=["핵심 쟁점을 한 문장으로 정리해 보시겠어요?", "아직 답변이 없는 질문이 있는지 확인해 주세요."],
    )


def _openai_syn(req: DiscussionSynthesizeRequest, api_key: str) -> DiscussionSynthesizeResponse:
    client = get_openai_client(api_key)
    sys = """교육용 토론 스레드 요약 조교입니다. JSON만 출력하세요.
{
  "summary": "한국어 3~6문장",
  "themes": ["주제·쟁점 2~5개"],
  "participation_notes": "참여 양상(한쪽 편중, 침묵 등) 참고 메모",
  "suggested_followups": ["교수/조교가 던질 만한 후속 질문 2~4개"]
}
비난·실명 추정은 피하고 교육적으로 작성하세요."""

    user = json.dumps(
        {
            "thread_title": req.thread_title,
            "posts": [{"author": p.author_label, "text": p.text[:4000]} for p in req.posts],
        },
        ensure_ascii=False,
    )
    res = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.25,
    )
    raw = (res.choices[0].message.content or "{}").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        raw = m.group(0)
    data: dict[str, Any] = json.loads(raw)
    return DiscussionSynthesizeResponse(
        mode="ai",
        summary=str(data.get("summary", ""))[:6000],
        themes=[str(x) for x in (data.get("themes") or [])][:10],
        participation_notes=str(data.get("participation_notes", ""))[:2000],
        suggested_followups=[str(x) for x in (data.get("suggested_followups") or [])][:8],
    )


@router.post("/synthesize", response_model=DiscussionSynthesizeResponse)
async def synthesize_discussion(body: DiscussionSynthesizeRequest) -> DiscussionSynthesizeResponse:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            return _openai_syn(body, key)
        except Exception:
            return _heuristic(body)
    return _heuristic(body)
