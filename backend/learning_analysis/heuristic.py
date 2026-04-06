"""API 키가 하나도 없을 때 최소 규칙 기반 추정(보조)."""

from __future__ import annotations

from learning_analysis.schemas import AnalyzeRequest, ModelJudgment


def heuristic_judgment(req: AnalyzeRequest) -> ModelJudgment:
    lp = req.learning
    ex = req.exam
    q = lp.quiz_average
    mid = ex.midterm_score
    fin = ex.final_or_recent_exam_score
    exam_hi = None
    if mid is not None and fin is not None:
        exam_hi = max(mid, fin)
    elif mid is not None:
        exam_hi = mid
    elif fin is not None:
        exam_hi = fin

    learn_lo = None
    vals = [
        v
        for v in [
            lp.lms_video_watch_ratio,
            lp.quiz_average,
            lp.assignment_on_time_ratio,
            lp.attendance_or_checkin_ratio,
        ]
        if v is not None
    ]
    if vals:
        learn_lo = sum(vals) / len(vals)

    cheat = 15.0
    gap_note = ""
    if learn_lo is not None and exam_hi is not None:
        gap = exam_hi - learn_lo
        if gap > 35:
            cheat = min(85.0, 40.0 + gap * 0.6)
            gap_note = (
                f"학습 과정 평균({learn_lo:.1f}) 대비 시험({exam_hi:.1f}) 차이가 큽니다. "
                "부정행위를 의미하지 않으며, 데이터 오류·편중·오픈북 등 다른 설명이 있을 수 있습니다."
            )
        elif gap < -25:
            cheat = max(5.0, 25.0 + gap * 0.2)
            gap_note = "시험이 과정 대비 낮게 나온 패턴입니다. 학습 지원·시험 환경을 점검할 가치가 있습니다."
        else:
            gap_note = "과정과 시험의 대략적 방향이 일치하는 편입니다."

    learn_state = "정량 입력이 부족해 학습 상태를 세밀히 말하기 어렵습니다."
    if learn_lo is not None:
        learn_state = f"과정 지표 평균 약 {learn_lo:.1f}/100 수준으로 보입니다."

    future = (
        "API 키를 설정하면 Gemini·ChatGPT·Claude·Grok 다중 모델로 "
        "불일치·학습·미래 위험을 더 정교히 분석할 수 있습니다."
    )

    return ModelJudgment(
        provider="heuristic",
        model_label="rule-based",
        ok=True,
        cheating_likelihood=round(cheat, 1),
        learning_state_summary=learn_state,
        mismatch_analysis=gap_note or "입력된 수치로는 불일치 신호가 약합니다.",
        future_prediction=future,
        confidence_note="규칙 기반 추정입니다. 교육적 판단·징계 결정에 단독으로 사용하지 마세요.",
    )
