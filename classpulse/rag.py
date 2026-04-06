"""Corpus loading, TF-IDF retrieval, and optional OpenAI answer generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    chunk_id: int
    source: str
    text: str


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts if parts else [text.strip()]


def _chunk_paragraph(paragraph: str, max_chars: int = 480, overlap: int = 80) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    out: list[str] = []
    start = 0
    while start < len(paragraph):
        end = min(start + max_chars, len(paragraph))
        piece = paragraph[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(paragraph):
            break
        start = max(0, end - overlap)
    return out


def load_corpus(corpus_dir: Path | None = None) -> list[Chunk]:
    base = corpus_dir or Path(__file__).resolve().parent / "corpus"
    chunks: list[Chunk] = []
    idx = 0
    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for para in _split_paragraphs(text):
            for piece in _chunk_paragraph(para):
                chunks.append(Chunk(chunk_id=idx, source=path.name, text=piece))
                idx += 1
    return chunks


def build_index(chunks: list[Chunk]) -> tuple[TfidfVectorizer, np.ndarray]:
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(max_features=8192, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve(
    query: str,
    chunks: list[Chunk],
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = 4,
) -> list[tuple[Chunk, float]]:
    if not query.strip() or not chunks:
        return []
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, matrix).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(chunks[int(i)], float(sims[int(i)])) for i in order if sims[int(i)] > 0]


def format_context_for_prompt(retrieved: list[tuple[Chunk, float]]) -> str:
    lines: list[str] = []
    for n, (ch, _) in enumerate(retrieved, start=1):
        lines.append(f"[{n}] (출처: {ch.source})\n{ch.text}")
    return "\n\n".join(lines)


def generate_answer(
    query: str,
    retrieved: list[tuple[Chunk, float]],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    from openai import OpenAI

    if not api_key.strip():
        return ""
    if not retrieved:
        return "검색된 강의 자료 구절이 없습니다. 질문을 바꾸거나 핵심 키워드를 넣어 보세요."

    context = format_context_for_prompt(retrieved)
    client = OpenAI(api_key=api_key)
    system = (
        "당신은 대학(또는 기관) 강의 보조입니다. 오직 제공된 [번호] 자료만 근거로 답하세요. "
        "자료에 없는 내용은 추측하지 말고 '제공된 자료에는 없습니다'라고 하세요. "
        "답변 끝에 사용한 근거 번호를 [1], [2] 형식으로 나열하세요. 한국어로 간결하게 답하세요."
    )
    user = f"자료:\n{context}\n\n질문: {query}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()
