"""Rubric-grounded feedback draft for instructors."""

from __future__ import annotations


def generate_feedback_draft(
    submission: str,
    rubric: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    from openai import OpenAI

    if not api_key.strip():
        return ""
    client = OpenAI(api_key=api_key)
    system = (
        "당신은 교육자입니다. 제공된 루브릭 항목에 맞춰 학생 제출물에 대한 피드백 **초안**을 한국어로 작성하세요. "
        "구조: (1) 잘된 점 1~2문장 (2) 개선이 필요한 점 2~3문장, 루브릭 항목명을 언급 "
        "(3) 다음에 할 구체적 액션 1문장. "
        "말미에 '본 문구는 AI 초안이며 교수가 검토·수정 후 전달해야 합니다' 한 줄을 넣으세요."
    )
    user = f"루브릭:\n{rubric}\n\n학생 제출:\n{submission}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()
