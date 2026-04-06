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
    assert r.headers.get("X-Request-ID")


def test_version_endpoint() -> None:
    r = client.get("/api/version")
    assert r.status_code == 200
    data = r.json()
    assert data["name"]
    assert data["version"]
    assert "/docs" in data.get("openapi_docs", "")


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
    assert "creative_insights" in data
    ci = data["creative_insights"]
    assert ci["reflection_kit"]["team_storyline"]
    assert len(ci["explain_facts"]) == 2
    assert "request_id" in data
    assert r.headers.get("X-Request-ID")
