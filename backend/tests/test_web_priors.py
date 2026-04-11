from __future__ import annotations

from edu_tools.team_web_priors import (
    _default_priors,
    compute_external_benchmark_score,
    compute_prior_score_delta,
    get_web_priors,
)


def test_compute_prior_matches_legacy_three_term_when_lines_self_zero() -> None:
    priors = _default_priors()
    adj = {
        "prior_strength": 1.0,
        "gap_weights": {"commit": 5.0, "pr": 4.0, "lines": 0.0, "attendance": 3.0, "self_report": 0.0},
        "max_delta": 100.0,
    }
    import math

    x = [0.8, 0.5, 0.4, 0.72, 0.3] + [0.0] * 29
    d_new, _ = compute_prior_score_delta(x, priors, adj)
    cg = min(1.0, max(0.0, x[0] / math.log1p(30))) - (priors["commit_expectation"] / 100.0)
    pg = min(1.0, max(0.0, x[1] / math.log1p(12))) - (priors["pr_expectation"] / 100.0)
    ag = x[3] - (priors["attendance_expectation"] / 100.0)
    d_old = cg * 5.0 + pg * 4.0 + ag * 3.0
    assert abs(d_new - d_old) < 1e-6


def test_get_web_priors_returns_adjustment() -> None:
    p, m = get_web_priors()
    assert "prior_adjustment" in m
    assert "gap_weights" in m["prior_adjustment"]


def test_external_benchmark_score_in_range() -> None:
    priors = _default_priors()
    x = [0.5, 0.4, 0.3, 0.72, 0.25] + [0.0] * 29
    s, d = compute_external_benchmark_score(x, priors)
    assert 0.0 <= s <= 100.0
    assert "mean_abs_gap" in d
