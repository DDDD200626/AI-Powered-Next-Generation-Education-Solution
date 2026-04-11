from __future__ import annotations

from edu_tools.team_research_evidence import compute_research_z_alignment


def test_research_z_alignment_perfect_match_high_score() -> None:
    profile = {
        "normative_on_features": {
            "log_commits": {"mean": 2.0, "std": 1.0},
            "log_prs": {"mean": 1.0, "std": 1.0},
            "log_lines": {"mean": 5.0, "std": 1.0},
            "attendance_norm": {"mean": 0.8, "std": 0.1},
            "log_self_report_words": {"mean": 3.0, "std": 1.0},
        },
        "integration": {"z_trim": 2.5},
    }
    x = [2.0, 1.0, 5.0, 0.8, 3.0] + [0.0] * 29
    s, d = compute_research_z_alignment(x, profile)
    assert s >= 99.0
    assert d.get("n_used") == 5


def test_research_z_alignment_far_from_norm_low_score() -> None:
    profile = {
        "normative_on_features": {
            "log_commits": {"mean": 0.0, "std": 0.01},
        },
        "integration": {"z_trim": 1.0},
    }
    x = [5.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 29
    s, d = compute_research_z_alignment(x, profile)
    assert s < 50.0
    assert "log_commits" in d.get("per_feature", {})
