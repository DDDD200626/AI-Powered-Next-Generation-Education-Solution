"""API 스모크·회귀 테스트 — CI에서 기술적 완성도 검증."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from learning_analysis.main import app

client = TestClient(app)


def test_health_ok_and_request_id_header() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "providers" in body
    assert body.get("version")
    assert body.get("app_name")
    assert body.get("openapi_docs") == "/docs"
    assert r.headers.get("X-Request-ID")
    assert r.headers.get("X-Process-Time-Ms")


def test_version_endpoint() -> None:
    r = client.get("/api/version")
    assert r.status_code == 200
    data = r.json()
    assert data["name"]
    assert data["version"]
    assert "/docs" in data.get("openapi_docs", "")


def test_perf_recent() -> None:
    client.get("/api/health")
    r = client.get("/api/perf/recent")
    assert r.status_code == 200
    data = r.json()
    assert "recent" in data
    assert isinstance(data["recent"], list)
    assert data.get("ring_buffer_enabled") is True
    assert r.headers.get("Cache-Control", "").startswith("no-store")


def test_analyze_includes_perf_breakdown() -> None:
    body = {
        "course_name": "테스트",
        "student_or_group_label": "익명",
        "learning": {},
        "exam": {},
        "context_for_educator": "",
    }
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "perf" in data and data["perf"]
    assert "llm_parallel_ms" in data["perf"]
    assert "local_ms" in data["perf"]
    assert "total_ms" in data["perf"]


def test_observability_ready_live() -> None:
    r = client.get("/api/observability")
    assert r.status_code == 200
    data = r.json()
    assert data.get("service")
    assert data.get("version")
    assert "uptime_seconds" in data
    assert data.get("environment")
    assert r.headers.get("Cache-Control", "").startswith("no-store")
    r2 = client.get("/api/ready")
    assert r2.status_code == 200
    assert r2.json().get("status") == "ready"
    r3 = client.get("/api/live")
    assert r3.status_code == 200
    assert r3.json().get("status") == "live"


def test_capabilities_contest_rubric() -> None:
    r = client.get("/api/capabilities")
    assert r.status_code == 200
    data = r.json()
    assert data["version"]
    rub = data["rubric"]
    assert "technical_completeness" in rub
    assert "ai_efficiency" in rub
    assert "planning_practical" in rub
    assert "creativity" in rub
    assert data["endpoints"]["team_evaluate"] == "POST /api/team/evaluate"
    assert data["endpoints"].get("observability") == "GET /api/observability"
    assert data["endpoints"].get("ready") == "GET /api/ready"
    assert data["endpoints"].get("perf_recent") == "GET /api/perf/recent"
    assert r.headers.get("X-Request-ID")


def test_rubric_draft_heuristic_returns_criteria() -> None:
    r = client.post(
        "/api/rubric/draft",
        json={
            "learning_objectives": "REST API를 설계하고 문서화할 수 있다.\n보안·오류 처리를 적용한다.",
            "course_name": "웹서비스",
            "assignment_type": "팀 프로젝트",
            "max_criteria": 4,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("mode") in ("heuristic", "ai")
    assert len(data.get("criteria") or []) >= 3
    assert data.get("rubric_markdown")
    assert "disclaimer" in data


def test_team_evaluate_heuristic_has_creative_insights() -> None:
    body = {
        "project_name": "테스트 프로젝트",
        "project_description": "",
        "evaluation_criteria": "",
        "members": [
            {
                "name": "김개발",
                "role": "백엔드",
                "commits": 12,
                "pull_requests": 3,
                "lines_changed": 400,
                "tasks_completed": 4,
                "meetings_attended": 3,
                "self_report": "구현과 리뷰를 맡았습니다.",
                "peer_notes": "",
                "timeline": [],
            },
            {
                "name": "이문서",
                "role": "기획",
                "commits": 2,
                "pull_requests": 1,
                "lines_changed": 50,
                "tasks_completed": 5,
                "meetings_attended": 4,
                "self_report": "문서와 회의 진행.",
                "peer_notes": "",
                "timeline": [],
            },
        ],
        "collaboration_edges": [],
    }
    r = client.post("/api/team/evaluate", json=body)
    assert r.status_code == 200
    data = r.json()
    assert len(data["members"]) == 2
    assert data.get("freerider_detection_overview")
    assert "product_tagline_ko" in data
    assert isinstance(data.get("team_dashboard"), list)
    assert len(data["team_dashboard"]) == 2
    assert "member_name" in data["team_dashboard"][0]
    assert "freerider_detection" in data["members"][0]
    fd0 = data["members"][0]["freerider_detection"]
    assert "basic_low_contribution" in fd0
    assert "rule_metrics" in fd0
    assert fd0["rule_metrics"] is not None
    assert "rubric_report" in data
    assert data["rubric_report"]["members"]
    assert "evaluation_trust" in data
    assert data["evaluation_trust"]["level_ko"]
    assert "team_risk" in data
    assert "improvement_chain" in data
    assert data["improvement_chain"]["items"]
    assert "creative_insights" in data
    ci = data["creative_insights"]
    assert ci["reflection_kit"]["team_storyline"]
    assert len(ci["explain_facts"]) == 2
    assert "team_health_score" in ci
    assert 0 <= ci["team_health_score"] <= 100
    assert ci.get("team_health_hint")
    assert "practical_toolkit" in data
    pt = data["practical_toolkit"]
    assert len(pt["teacher_checklist"]) >= 3
    assert "request_id" in data
    assert r.headers.get("X-Request-ID")


def test_team_evaluate_compare_no_api_keys_returns_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """키가 없을 때도 비교 엔드포인트는 200 + 모델별 오류 메시지로 응답."""
    for k in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    body = {
        "project_name": "비교 테스트",
        "project_description": "",
        "evaluation_criteria": "",
        "members": [
            {
                "name": "A",
                "role": "",
                "commits": 5,
                "pull_requests": 1,
                "lines_changed": 100,
                "tasks_completed": 2,
                "meetings_attended": 2,
                "self_report": "",
                "peer_notes": "",
                "timeline": [],
            },
        ],
        "collaboration_edges": [],
    }
    r = client.post("/api/team/evaluate/compare", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert len(data["models"]) >= 1
    assert data.get("comparison_summary")
    assert data.get("product_mode") == "ai_multi_eval"
    assert len(data.get("pipeline_steps") or []) == 6
    assert data.get("divergence") is None
    assert data.get("trust_scores") is None
    assert data.get("explainability") == []
    assert "request_id" in data
    assert r.headers.get("X-Request-ID")
