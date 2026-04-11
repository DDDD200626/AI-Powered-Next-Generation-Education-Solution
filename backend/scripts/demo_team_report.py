#!/usr/bin/env python3
"""샘플 팀 데이터로 POST /api/team/report 를 호출하고 JSON·요약을 남깁니다.

사용:
  cd backend && python scripts/demo_team_report.py
  cd backend && python scripts/demo_team_report.py uneven
  cd backend && python scripts/demo_team_report.py small

시나리오:
  default — 4인, 역할 분담이 비교적 고른 팀 (기본, 저장: demo_team_report_response.json)
  uneven  — 3인, 한 명은 커밋·PR 많음 / 한 명은 서술·출석 위주로 격차 큼
  small   — 2인, 빠른 확인용

환경:
  TEAM_REPORT_DEMO_URL — 비우면 먼저 HTTP(127.0.0.1:8010) 시도 후 실패 시 TestClient(서버 불필요).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

SCENARIOS: dict[str, dict] = {
    "default": {
        "project_name": "데모: 팀 기여도 평가 (샘플 입력)",
        "use_deep_learning": True,
        "deep_learning_accumulate_samples": False,
        "teamData": [
            {
                "name": "김민지",
                "commits": 42,
                "prs": 8,
                "lines": 3200,
                "attendance": 95,
                "selfReport": (
                    "백엔드 API와 DB 스키마 설계를 맡았습니다. "
                    "스프린트마다 페어 프로그래밍으로 리뷰를 주도했고, 장애 대응 시 온콜에 참여했습니다."
                ),
            },
            {
                "name": "이준호",
                "commits": 28,
                "prs": 5,
                "lines": 2100,
                "attendance": 88,
                "selfReport": (
                    "프론트엔드 화면과 컴포넌트 라이브러리를 구축했습니다. "
                    "접근성 점검과 디자인 시스템 문서 초안을 작성했습니다."
                ),
            },
            {
                "name": "박서연",
                "commits": 15,
                "prs": 3,
                "lines": 890,
                "attendance": 92,
                "selfReport": "CI 테스트 보강과 배포 파이프라인 점검, 릴리즈 노트 정리를 주로 맡았습니다.",
            },
            {
                "name": "최도윤",
                "commits": 8,
                "prs": 1,
                "lines": 220,
                "attendance": 75,
                "selfReport": "문서화와 회의록 정리, 이슈 트리아지를 도왔습니다.",
            },
        ],
    },
    "uneven": {
        "project_name": "데모: 격차 큰 팀 (한 명 집중 기여 + 기획·서술형 멤버)",
        "use_deep_learning": True,
        "deep_learning_accumulate_samples": False,
        "teamData": [
            {
                "name": "강태양",
                "commits": 120,
                "prs": 22,
                "lines": 12000,
                "attendance": 98,
                "selfReport": (
                    "모놀리스 분해·마이그레이션 리드를 맡았고, 온콜·장애 복구를 수차례 수행했습니다. "
                    "코드 오너십과 리뷰 정책을 제안했습니다."
                ),
            },
            {
                "name": "한지우",
                "commits": 3,
                "prs": 0,
                "lines": 40,
                "attendance": 85,
                "selfReport": (
                    "요구사항 정리·스토리보드·사용자 인터뷰 노트를 작성했습니다. "
                    "회의 진행과 외부 팀과의 커뮤니케이션을 주로 담당했습니다."
                ),
            },
            {
                "name": "오하람",
                "commits": 18,
                "prs": 4,
                "lines": 650,
                "attendance": 90,
                "selfReport": "테스트 케이스 보강과 버그 재현, 릴리즈 체크리스트를 맡았습니다.",
            },
        ],
    },
    "small": {
        "project_name": "데모: 2인 퀵 테스트",
        "use_deep_learning": True,
        "deep_learning_accumulate_samples": False,
        "teamData": [
            {
                "name": "A",
                "commits": 20,
                "prs": 4,
                "lines": 800,
                "attendance": 90,
                "selfReport": "백엔드 API",
            },
            {
                "name": "B",
                "commits": 5,
                "prs": 1,
                "lines": 100,
                "attendance": 70,
                "selfReport": "문서",
            },
        ],
    },
}


def _out_path(scenario: str) -> Path:
    if scenario == "default":
        return BACKEND_ROOT / "scripts" / "demo_team_report_response.json"
    return BACKEND_ROOT / "scripts" / f"demo_team_report_response_{scenario}.json"


def _post_http(url: str, payload: dict) -> dict:
    import httpx

    r = httpx.post(f"{url.rstrip('/')}/api/team/report", json=payload, timeout=180.0)
    r.raise_for_status()
    return r.json()


def _post_testclient(payload: dict) -> dict:
    from fastapi.testclient import TestClient

    from learning_analysis.main import app

    with TestClient(app) as client:
        r = client.post("/api/team/report", json=payload)
        r.raise_for_status()
        return r.json()


def main() -> int:
    argv = [a.strip().lower() for a in sys.argv[1:] if a.strip()]
    scenario = "default"
    if argv:
        if argv[0] in SCENARIOS:
            scenario = argv[0]
        else:
            print(f"알 수 없는 시나리오: {argv[0]!r}. 사용 가능: {', '.join(SCENARIOS)}", file=sys.stderr)
            return 2

    payload = SCENARIOS[scenario]
    out_path = _out_path(scenario)

    data: dict | None = None
    mode = ""
    demo_url = (os.environ.get("TEAM_REPORT_DEMO_URL") or "").strip()

    if demo_url:
        try:
            data = _post_http(demo_url, payload)
            mode = f"http:{demo_url}"
        except Exception as e:
            print(f"[HTTP 실패] {e}", file=sys.stderr)
            data = None

    if data is None and not demo_url:
        for base in ("http://127.0.0.1:8010", "http://127.0.0.1:8000"):
            try:
                data = _post_http(base, payload)
                mode = f"http:{base}"
                break
            except Exception:
                continue

    if data is None:
        data = _post_testclient(payload)
        mode = "TestClient(in-process)"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 팀 기여도 평가 데모 결과 ===")
    print(f"시나리오: {scenario}")
    print(f"호출 방식: {mode}")
    print(f"저장 파일: {out_path}")
    print()
    scores = data.get("scores") or []
    for s in scores:
        name = s.get("member_name", "?")
        raw = s.get("rawScore")
        norm = s.get("normalizedScore")
        rank = s.get("rank")
        blend = s.get("blendedScore")
        dl = s.get("dl_score")
        dl_c = s.get("dl_confidence")
        line = f"  {name}: raw={raw} · 정규화={norm} · 순위={rank}"
        if blend is not None:
            line += f" · 혼합={blend}"
        if dl is not None:
            line += f" · DL={dl}"
        if dl_c is not None:
            line += f" · DL신뢰도={dl_c}"
        print(line)

    log = data.get("evaluation_log") or {}
    if log.get("request_id"):
        print(f"\nrequest_id: {log.get('request_id')}")

    dl_info = data.get("dl_model_info") or {}
    if isinstance(dl_info, dict):
        en = dl_info.get("enabled")
        mn = dl_info.get("model_name")
        print(f"\ndl_model_info: enabled={en} · model_name={mn}")

    print("\n전체 JSON은 위 저장 파일을 열어보면 됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
