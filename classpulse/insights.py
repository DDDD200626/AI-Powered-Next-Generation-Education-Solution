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


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    try:
        raw = uploaded_file.read()
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None
