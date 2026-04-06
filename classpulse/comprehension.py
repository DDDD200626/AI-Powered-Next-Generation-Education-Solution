"""강의 자료(RAG) 맥락을 바탕으로 한 학습 이해도 추적용 AI 평가."""

from __future__ import annotations

from classpulse.rag import Chunk, format_context_for_prompt


def assess_comprehension(
    question: str,
    student_answer: str,
    retrieved: list[tuple[Chunk, float]],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    from openai import OpenAI

    if not api_key.strip():
        return ""
    if not question.strip() or not student_answer.strip():
        return ""
    if not retrieved:
        ctx = "(검색된 강의 구절 없음 — 평가는 제한적입니다.)"
    else:
        ctx = format_context_for_prompt(retrieved)

    client = OpenAI(api_key=api_key)
    system = (
        "당신은 교육 평가 보조 도구입니다. 아래 [강의 자료 발췌]만 근거로 학생 답변의 이해도를 평가하세요. "
        "단정적 진단(‘부정행위’ 등)은 하지 말고 학습 관점에서만 서술하세요.\n\n"
        "출력 형식(한국어, 반드시 이 순서와 소제목 사용):\n"
        "## 이해도 점수\n"
        "0~100 숫자 한 줄, 그 다음 한 줄에 근거를 1~2문장.\n"
        "## 수준\n"
        "상/중/하 중 하나와 이유 1문장.\n"
        "## 자료와의 정합성\n"
        "자료와 일치·부분 일치·벗어남 중 어디에 가까운지와 근거.\n"
        "## 보완 학습 포인트\n"
        "불릿 2~4개.\n"
        "## 교사 검토 메모\n"
        "한 문장: 본 평가는 AI 보조이며 최종 판단은 교사에게 있음을 명시."
    )
    user = (
        f"[강의 자료 발췌]\n{ctx}\n\n"
        f"[평가 질문]\n{question.strip()}\n\n"
        f"[학생 답변]\n{student_answer.strip()}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.25,
    )
    return (resp.choices[0].message.content or "").strip()
