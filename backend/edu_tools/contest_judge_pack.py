"""
공모전 심사 가산점용: 심사위원이 API·JSON 경로만으로 근거를 따라가게 하는 메타 패키지.
"""

from __future__ import annotations

from typing import Any


def contest_submission_pack() -> dict[str, Any]:
    """GET /api/capabilities 에 포함 — 기계 판독·인쇄 제출용 증빙 포인터."""
    return {
        "label_ko": "심사 즉시 검증 패키지",
        "summary_ko": (
            "심사기준별로 ‘어떤 엔드포인트·JSON 필드’를 보면 되는지 고정 경로로 제시합니다. "
            "코드 저장소와 함께 제출 시 재현·투명성 심사에 유리합니다."
        ),
        "verification_order": [
            {
                "step": 1,
                "method": "GET",
                "path": "/api/capabilities",
                "confirms_ko": "4+1 심사축(rubric)·dl_roadmap·엔드포인트 목록·본 패키지",
            },
            {
                "step": 2,
                "method": "GET",
                "path": "/api/observability",
                "confirms_ko": "업타임·환경·관측성(기술적 완성도·운영)",
            },
            {
                "step": 3,
                "method": "GET",
                "path": "/api/health",
                "confirms_ko": "AI 제공자 설정 여부(ai_efficiency·비용 대응)",
            },
            {
                "step": 4,
                "method": "POST",
                "path": "/api/team/report",
                "confirms_ko": "dl_model_info·contest_transparency·품질 메타(딥러닝 투명성)",
                "body_min_ko": "JSON teamData에 팀원 1명 이상(커밋·PR 등 필드)",
            },
        ],
        "json_paths": {
            "rubric_machine_readable": "capabilities.rubric.*",
            "dl_roadmap_phases": "capabilities.dl_roadmap.phases_ko",
            "dl_quality_flat": "response.dl_model_info.quality",
            "dl_quality_unified": "response.dl_model_info.quality.dl_quality_unified",
            "dl_contest_transparency": "response.dl_model_info.contest_transparency",
            "dl_blend_formula": "response.dl_model_info.contest_transparency.blend_formula",
            "dl_limitations_ko": "response.dl_model_info.contest_transparency.limitations_ko",
            "dl_quality_snapshot": "response.dl_model_info.contest_transparency.quality_snapshot",
            "dl_prior_calibration": "response.dl_model_info.prior_calibration",
            "web_priors_meta": "response.dl_model_info.web_priors",
            "benchmark_inference": "response.dl_model_info.benchmark_inference",
            "request_tracing": "HTTP 응답 헤더 X-Request-ID, X-Process-Time-Ms",
            "dataset_label_summary": "response.stats (GET /api/team/data/dataset-label-summary)",
        },
        "repository_evidence_ko": [
            "docs/CONTEST_RUBRIC.md — 심사 대응서(근거 경로)",
            "docs/OPERATIONS_AND_LIMITS.md — 운영·보안·개인정보·한계",
            "docs/ABSOLUTE_COMPLETION.md — 완성 선언의 의미(태그·재현)",
            "backend/tests/ — pytest 회귀",
            ".github/workflows/ci.yml — CI(있는 경우)",
        ],
    }
