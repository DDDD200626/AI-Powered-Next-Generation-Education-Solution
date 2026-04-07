"""
ClassPulse AI — 경량 교육 보조 데모 (Streamlit)
실행(저장소 루트에서): streamlit run classpulse/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from classpulse.insights import (
    default_sample_metrics,
    metrics_df_is_valid,
    metrics_summary_stats,
    parse_uploaded_csv,
    summarize_for_admin,
)
from classpulse.rag import build_index, generate_answer, load_corpus, retrieve
from classpulse.teacher_feedback import generate_feedback_draft


def _show_openai_error(err: BaseException) -> None:
    try:
        from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
    except ImportError:
        st.error(f"OpenAI 호출 오류: {err}")
        return
    if isinstance(err, AuthenticationError):
        st.error(
            "**API 키가 잘못되었거나 OpenAI 키가 아닙니다.** "
            "https://platform.openai.com/account/api-keys 에서 **`sk-`로 시작하는** 키를 새로 만들어 "
            "`.env`의 `OPENAI_API_KEY` 또는 사이드바 입력란에 넣으세요. "
            "다른 서비스 토큰은 사용할 수 없습니다."
        )
    elif isinstance(err, RateLimitError):
        st.error("요청 한도에 걸렸습니다. 잠시 후 다시 시도하세요.")
    elif isinstance(err, APIConnectionError):
        st.error("OpenAI 서버에 연결할 수 없습니다. 네트워크·방화벽을 확인하세요.")
    elif isinstance(err, APIStatusError):
        st.error(f"OpenAI API 오류: {getattr(err, 'message', err)}")
    else:
        st.error(f"오류: {err}")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
        load_dotenv()
    except ImportError:
        pass


def _api_key() -> str:
    _load_dotenv()
    env_k = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if env_k:
        return env_k
    try:
        return (st.secrets.get("OPENAI_API_KEY") or "").strip()
    except Exception:
        return ""


@st.cache_resource
def _rag_index():
    chunks = load_corpus()
    vectorizer, matrix = build_index(chunks)
    return chunks, vectorizer, matrix


def main() -> None:
    st.set_page_config(
        page_title="ClassPulse AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ClassPulse AI")
    st.caption(
        "교육 현장 보조 레이어 — 근거 기반 학습 도우미 · 루브릭 피드백 초안 · 운영 지표 요약 (LMS 대체 아님)"
    )

    key = _api_key()
    with st.sidebar:
        st.subheader("OpenAI API")
        if key:
            st.success("API 키: 환경변수 또는 Streamlit Secrets에서 로드됨")
        else:
            st.warning("키가 없으면 **검색 근거만** 표시되고, 생성 응답은 비활성입니다.")
        with st.expander("로컬/심사용 키 입력 (세션만)", expanded=False):
            session_key = st.text_input(
                "OPENAI_API_KEY",
                type="password",
                help="브라우저 세션에만 유지됩니다. 저장소에 커밋하지 마세요.",
                key="manual_openai_key",
            )
            if session_key.strip():
                key = session_key.strip()
        st.caption("OpenAI 키는 **sk-** 로 시작합니다.")
        st.caption("배포 시 Streamlit Secrets에 `OPENAI_API_KEY`를 등록하세요.")

    chunks, vectorizer, matrix = _rag_index()

    tab_learn, tab_teacher, tab_ops = st.tabs(
        ["수강생 · 근거 기반 학습 도우미", "교강사 · 피드백 초안", "운영자 · 지표 요약"]
    )

    with tab_learn:
        st.markdown(
            "강의 자료(`classpulse/corpus/*.md`) 범위에서만 검색한 뒤, API 키가 있으면 LLM이 출처 번호를 붙여 답합니다."
        )
        q = st.text_input("질문", placeholder="예: RAG가 교실에서 어떤 페인을 줄여 주나요?")
        top_k = st.slider("검색 구절 수", 2, 8, 4)
        if st.button("질문하기", type="primary") and q.strip():
            hits = retrieve(q.strip(), chunks, vectorizer, matrix, top_k=top_k)
            st.subheader("검색된 근거")
            if not hits:
                st.info("관련 구절을 찾지 못했습니다. 다른 표현으로 질문해 보세요.")
            else:
                for ch, score in hits:
                    with st.expander(f"[{ch.chunk_id}] {ch.source} · 유사도 {score:.3f}"):
                        st.write(ch.text)
            if key:
                ans = ""
                ok = True
                with st.spinner("답변 생성 중…"):
                    try:
                        ans = generate_answer(q.strip(), hits, key)
                    except Exception as e:
                        _show_openai_error(e)
                        ok = False
                st.subheader("AI 답변 (자료 근거)")
                if ok:
                    st.write(ans or "(빈 응답)")
            else:
                st.info("OpenAI API 키를 설정하면 위 근거를 바탕으로 한 답변이 생성됩니다.")

    with tab_teacher:
        st.markdown("루브릭과 학생 제출을 넣으면 **초안** 피드백을 생성합니다. 제출 전 교사 검수가 전제입니다.")
        rubric = st.text_area(
            "루브릭 / 평가 기준",
            height=160,
            value=(
                "문제 정의 25점, AI 설계 25점, 구현 30점, 윤리·한계 20점. "
                "각 항목은 구체적 근거가 있어야 합니다."
            ),
        )
        submission = st.text_area("학생 제출(요약 또는 본문)", height=220, placeholder="프로젝트 설명, 보고서 일부 등")
        if st.button("피드백 초안 생성", type="primary"):
            if not key:
                st.error("OpenAI API 키가 필요합니다.")
            elif not submission.strip():
                st.warning("학생 제출 내용을 입력하세요.")
            else:
                draft = ""
                ok = True
                with st.spinner("초안 작성 중…"):
                    try:
                        draft = generate_feedback_draft(submission.strip(), rubric, key)
                    except Exception as e:
                        _show_openai_error(e)
                        ok = False
                st.subheader("피드백 초안")
                if ok:
                    st.write(draft or "(빈 응답)")

    with tab_ops:
        st.markdown("주차별 제출률·지각·질문 수 샘플을 요약합니다. CSV를 올리면 동일 형식 컬럼을 사용하세요.")
        up = st.file_uploader("운영 CSV (선택)", type=["csv"])
        df = parse_uploaded_csv(up) if up else None
        if df is None or not metrics_df_is_valid(df):
            if up is not None and df is not None and not df.empty:
                st.warning(
                    "CSV에 필요한 열이 없습니다. 예: assignment_submit_rate, late_submissions, forum_questions"
                )
            df = default_sample_metrics()
            if up is None:
                st.info(
                    "샘플 데이터를 표시합니다. 컬럼 예: week, enrolled, assignment_submit_rate, late_submissions, forum_questions"
                )
        st.dataframe(df, use_container_width=True)
        stats = metrics_summary_stats(df)
        st.json(stats)
        if st.button("운영 요약 생성", type="primary"):
            if not key:
                st.error("OpenAI API 키가 필요합니다.")
            else:
                summary = ""
                ok = True
                with st.spinner("요약 중…"):
                    try:
                        summary = summarize_for_admin(df, key)
                    except Exception as e:
                        _show_openai_error(e)
                        ok = False
                st.subheader("AI 운영 요약")
                if ok:
                    st.write(summary or "(빈 응답)")


if __name__ == "__main__":
    main()
