"""Sample operations metrics and NL summary for administrators."""

from __future__ import annotations

import io

import pandas as pd


def default_sample_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week": [1, 2, 3, 4, 5],
            "enrolled": [120, 118, 115, 112, 110],
            "assignment_submit_rate": [0.97, 0.91, 0.88, 0.85, 0.82],
            "late_submissions": [3, 8, 12, 15, 18],
            "forum_questions": [5, 12, 18, 22, 28],
        }
    )


REQUIRED_METRIC_COLS = (
    "assignment_submit_rate",
    "late_submissions",
    "forum_questions",
)


def metrics_df_is_valid(df: pd.DataFrame) -> bool:
    return bool(not df.empty and all(c in df.columns for c in REQUIRED_METRIC_COLS))


def metrics_summary_stats(df: pd.DataFrame) -> dict[str, float | int]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    return {
        "weeks": int(len(df)),
        "last_submit_rate": float(last["assignment_submit_rate"]),
        "submit_rate_delta": float(last["assignment_submit_rate"] - prev["assignment_submit_rate"]),
        "last_late": int(last["late_submissions"]),
        "late_delta": int(last["late_submissions"] - prev["late_submissions"]),
        "last_forum_q": int(last["forum_questions"]),
    }


def summarize_for_admin(df: pd.DataFrame, api_key: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI

    if not api_key.strip():
        return ""
    stats = metrics_summary_stats(df)
    table = df.to_csv(index=False)
    client = OpenAI(api_key=api_key)
    system = (
        "당신은 교육 운영 담당자에게 보고하는 분석가입니다. 표 데이터와 요약 수치만 근거로 "
        "3~5문장 한국어로 핵심 이슈, 추세, 권장 조치(조기 개입·공지·오피스아워 등)를 제안하세요. "
        "수치는 과장하지 말고 표에 있는 범위 안에서만 말하세요."
    )
    user = f"주차별 지표(CSV):\n{table}\n\n파생 수치: {stats}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.35,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_csv_text(text: str) -> pd.DataFrame | None:
    if not (text or "").strip():
        return None
    try:
        return pd.read_csv(io.StringIO(text.strip()))
    except Exception:
        return None


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    try:
        raw = uploaded_file.read()
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None


# --- 통합 대시보드(이해도 + 무결성 신호 요약) ---

INTEGRATED_REQUIRED = (
    "student_id",
    "comprehension_score",
    "integrity_risk_band",
)


def default_integrated_learner_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["S001", "S002", "S003", "S004", "S005"],
            "comprehension_score": [82, 64, 71, 45, 88],
            "integrity_risk_band": ["low", "medium", "low", "high", "low"],
            "last_activity_days_ago": [1, 3, 2, 14, 1],
            "quiz_attempts": [3, 2, 4, 1, 5],
            "notes": [
                "자료 정합성 양호",
                "답변 짧음·재시도 권장",
                "",
                "휴리스틱·LLM 플래그 다수(교사 확인)",
                "",
            ],
        }
    )


def integrated_df_is_valid(df: pd.DataFrame) -> bool:
    return bool(not df.empty and all(c in df.columns for c in INTEGRATED_REQUIRED))


def integrated_summary_stats(df: pd.DataFrame) -> dict[str, float | int | str]:
    last_band = df["integrity_risk_band"].astype(str).str.lower()
    high_n = int((last_band == "high").sum())
    med_n = int((last_band == "medium").sum())
    low_n = int((last_band == "low").sum())
    scores = pd.to_numeric(df["comprehension_score"], errors="coerce")
    return {
        "learners": int(len(df)),
        "comprehension_mean": float(scores.mean()) if scores.notna().any() else 0.0,
        "integrity_high": high_n,
        "integrity_medium": med_n,
        "integrity_low": low_n,
    }


def summarize_integrated_dashboard(df: pd.DataFrame, api_key: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI

    if not api_key.strip():
        return ""
    stats = integrated_summary_stats(df)
    table = df.to_csv(index=False)
    client = OpenAI(api_key=api_key)
    system = (
        "당신은 교육 담당자에게 보고하는 데이터 분석가입니다. "
        "표의 student_id·comprehension_score·integrity_risk_band 등만 근거로 "
        "4~7문장 한국어로 (1) 전체 이해도 추세 (2) 무결성 리스크 구간별 개입 우선순위 "
        "(3) 면담·추가 평가·학습 지원 권장을 제안하세요. "
        "개인을 비난하거나 부정행위를 단정하지 마세요. "
        "수치는 표에 있는 범위에서만 언급하세요."
    )
    user = f"학습자 요약 표(CSV):\n{table}\n\n파생 수치: {stats}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.35,
    )
    return (resp.choices[0].message.content or "").strip()
