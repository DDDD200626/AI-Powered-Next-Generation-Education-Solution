# 공모전 제출 패키지

이 문서는 `팀 프로젝트 기여도 자동 평가 시스템`의 심사 제출용 운영 패키지입니다.  
핵심 기능 설명이 아니라, **발표/검증/비상 대응**을 빠르게 수행하기 위한 실전 문서입니다.

## 구성

- `docs/DEMO_RUNBOOK_3MIN.md` — 3분 시연 스크립트
- `docs/SUBMISSION_CHECKLIST.md` — 제출 전 확인 체크리스트
- `docs/FALLBACK_PLAN.md` — API 키/네트워크 이슈 시 플랜B

## 제출 시 같이 제시할 근거

- API 문서: `/docs`
- 심사 기준 정렬: `docs/CONTEST_RUBRIC.md`
- 기여도 평가 실행 결과:
  - 팀 리포트 화면
  - `심사 제출 패키지` 버튼으로 내보낸 JSON
  - 추세 그래프(30/60/90일, 멤버별)

## 권장 제출 캡처

- 팀 리포트 상단(정규화·DL·혼합 점수)
- 심사기준 점수 요약(기술/AI/실무/창의)
- 개인 기여도 추세 그래프
- Evidence JSON 일부(`judge_criteria_scores`, `judge_criteria_rationale`)

