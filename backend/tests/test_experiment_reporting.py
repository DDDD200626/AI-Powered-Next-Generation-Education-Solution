from __future__ import annotations

from edu_tools.experiment_reporting import collect_dl_experiment_context


def test_collect_dl_experiment_context_schema() -> None:
    ctx = collect_dl_experiment_context()
    assert ctx.get("schema") == "dl_experiment_context_v1"
    assert "captured_at_utc" in ctx
    assert "dataset_path" in ctx
    assert "metrics_summary" in ctx
    assert isinstance(ctx.get("env_team_torch_semantic_train"), dict)
