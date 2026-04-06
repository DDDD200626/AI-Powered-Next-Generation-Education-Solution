"""부정행위 ‘판정’이 아닌, 제출물·답변의 무결성 보조 신호(휴리스틱 + LLM 설명)."""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from classpulse.rag import Chunk


def heuristic_integrity_signals(text: str) -> dict[str, float | int | str]:
    """교육 현장 참고용 휴리스틱. 단독으로 부정행위를 의미하지 않습니다."""
    t = (text or "").strip()
    if not t:
        return {
            "char_count": 0,
            "word_count": 0,
            "unique_word_ratio": 0.0,
            "longest_token_chars": 0,
            "multi_space_runs": 0,
            "note": "내용이 비어 있습니다.",
        }

    words = re.findall(r"\S+", t)
    word_tokens = re.findall(r"[\w가-힣]+", t.lower())
    uniq_ratio = (len(set(word_tokens)) / len(word_tokens)) if word_tokens else 0.0
    tokens = re.findall(r"\S+", t)
    longest = max((len(x) for x in tokens), default=0)
    multi_space = len(re.findall(r" {2,}", t)) + len(re.findall(r"\t", t))

    note_parts: list[str] = []
    if len(word_tokens) < 8:
        note_parts.append("매우 짧은 답변입니다.")
    if uniq_ratio < 0.35 and len(word_tokens) >= 20:
        note_parts.append("어휘 다양도가 낮습니다(반복·템플릿 가능성 참고).")
    if longest > 80:
        note_parts.append("끊김 없이 긴 토큰이 있습니다(붙여넣기 흔적 참고).")

    return {
        "char_count": len(t),
        "word_count": len(words),
        "unique_word_ratio": round(uniq_ratio, 3),
        "longest_token_chars": longest,
        "multi_space_runs": multi_space,
        "note": " ".join(note_parts) if note_parts else "특이 패턴 없음(휴리스틱 기준).",
    }


def answer_corpus_max_similarity(
    answer: str,
    chunks: list[Chunk],
    vectorizer,
    matrix,
) -> tuple[float, str | None]:
    """답변과 코퍼스 청크 간 최대 코사인 유사도(‘자료 복붙’ 의심 참고용)."""
    if not answer.strip() or not chunks:
        return 0.0, None
    av = vectorizer.transform([answer.strip()])
    sims = cosine_similarity(av, matrix).flatten()
    i = int(np.argmax(sims))
    return float(sims[i]), chunks[i].source if 0 <= i < len(chunks) else None


def assess_integrity_llm(
    assignment_prompt: str,
    student_submission: str,
    api_key: str,
    corpus_excerpt: str | None = None,
    model: str = "gpt-4o-mini",
) -> str:
    """LLM으로 ‘의심 요인·교육적 후속’만 제안. 법적·징계적 단정 금지."""
    from openai import OpenAI

    if not api_key.strip():
        return ""
    client = OpenAI(api_key=api_key)
    system = (
        "당신은 학습 무결성(academic integrity) 지원 분석가입니다. "
        "절대 ‘부정행위 확정’ ‘적발’ 같은 표현을 쓰지 마세요. "
        "오직 관찰 가능한 패턴, 자기검열·재작성 권장, 교사가 추가로 볼 포인트를 제안합니다.\n\n"
        "출력 형식(한국어):\n"
        "## 요약\n"
        "2~3문장.\n"
        "## 정성적 신호(참고)\n"
        "불릿 3~5개. 각 항목은 ‘가능성’ ‘참고’ 어휘를 사용.\n"
        "## 교육적 권장\n"
        "불릿 2~4개(면담 질문, 추가 과제, 오리지널리티 설명 요청 등).\n"
        "## 한계\n"
        "한 문장: 휴리스틱·단일 제출만으로는 판단 불가함을 명시."
    )
    extra = ""
    if corpus_excerpt and corpus_excerpt.strip():
        extra = f"\n\n[참고: 강의 자료 발췌(있을 경우)]\n{corpus_excerpt.strip()}"
    user = (
        f"[과제·시험 지문]\n{assignment_prompt.strip() or '(미입력)'}\n\n"
        f"[학생 제출]\n{student_submission.strip()}"
        f"{extra}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()
