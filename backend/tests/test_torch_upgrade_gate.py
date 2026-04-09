from __future__ import annotations

from edu_tools.team_torch_model import _promotion_gate_decision


def test_promotion_gate_accepts_when_candidate_is_not_worse() -> None:
    prev = {"validation_mae_mean": 8.4, "holdout_time_mae": 9.1}
    cand = {"validation_mae_mean": 8.2, "holdout_time_mae": 8.9}
    ok, reasons = _promotion_gate_decision(prev, cand)
    assert ok is True
    assert "metrics_ok" in reasons


def test_promotion_gate_rejects_when_candidate_regresses() -> None:
    prev = {"validation_mae_mean": 7.0, "holdout_time_mae": 8.0}
    cand = {"validation_mae_mean": 8.2, "holdout_time_mae": 9.1}
    ok, reasons = _promotion_gate_decision(prev, cand)
    assert ok is False
    assert "validation_mae_regressed" in reasons or "holdout_mae_regressed" in reasons
