"""LLM 공통 지시 — JSON만 출력."""

from __future__ import annotations

import json

from learning_analysis.schemas import AnalyzeRequest


def build_user_payload(req: AnalyzeRequest) -> str:
    d = {
        "course_name": req.course_name,
        "student_or_group_label": req.student_or_group_label,
        "learning": req.learning.model_dump(),
        "exam": req.exam.model_dump(),
        "context_for_educator": req.context_for_educator,
    }
    return json.dumps(d, ensure_ascii=False, indent=2)


SYSTEM_KO = """당신은 교육 데이터를 다루는 분석가입니다. 입력은 학습 과정 지표와 시험 결과입니다.
다음을 수행하세요.
1) 학습 과정과 시험 결과의 **불일치**를 분석합니다. (예: 과정 지표는 낮은데 시험만 매우 높음 → 부정행위 **가능성** 검토, 또는 역으로 학습 부진 신호)
2) **부정행위 여부를 단정하지 말고**, 의심 수준을 0–100 숫자로 표현합니다.
3) **학습 상태**(이해도·참여·습관)를 교육적 관점에서 요약합니다.
4) **미래 예측**: 이 패턴이 이어질 때 다음 시험·학습에 대한 위험·개선 방향을 짧게 예측합니다.

반드시 **JSON 한 개만** 출력하세요. 키는 정확히 다음과 같습니다.
{
  "cheating_likelihood": 숫자 0-100,
  "learning_state_summary": "문자열",
  "mismatch_analysis": "문자열",
  "future_prediction": "문자열",
  "confidence_note": "데이터 한계·불확실성"
}

한국어로 작성하세요. 특정 개인을 비하하거나 범죄 단정을 하지 마세요."""


def user_message(req: AnalyzeRequest) -> str:
    return f"아래 데이터를 분석하세요.\n\n{build_user_payload(req)}"
