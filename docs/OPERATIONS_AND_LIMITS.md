# 운영·보안·개인정보·한계 (운영자·심사 참고)

본 문서는 제품을 **실제 기관에 붙일 때** 점검할 항목과, 코드에 이미 반영된 설정을 정리합니다. 법률 자문을 대체하지 않습니다.

**“완성”이라는 말의 정의**는 `docs/ABSOLUTE_COMPLETION.md` 에서 다룹니다(영구·절대 완성 불가, 태그 단위 완료).

---

## 1. 보안 (백엔드)

| 항목 | 동작 | 환경 변수 |
|------|------|-----------|
| 응답 보안 헤더 | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin` 기본 적용 | `EDUSIGNAL_SECURITY_HEADERS=0` 으로 끔 |
| HSTS | HTTPS 리버스 프록시 뒤에서만 설정 권장 | `EDUSIGNAL_HSTS_MAX_AGE` (초 단위 숫자, 예: `31536000`) |
| CORS | 기본은 로컬·LAN Vite Origin + `CORS_ORIGINS` 목록 | `CORS_ORIGINS`, `CORS_ORIGIN_REGEX` |
| 무거운 POST 요율 제한 | IP(또는 `X-Forwarded-For` 첫 값)당 분당 N회, **기본 비활성** | `EDUSIGNAL_POST_RATE_LIMIT_PER_MINUTE` (0이면 미사용). 대상: `/api/team/report`, `/api/team/evaluate`, `/api/analyze`, `/api/llm/compare` |

요청 추적: `X-Request-ID`, `X-Process-Time-Ms` — `GET /api/perf/recent`, `GET /api/observability` 와 함께 운영 근거로 사용합니다.

---

## 2. 개인정보·데이터 보관

- 팀 평가 입력·로그는 **기관의 개인정보 처리방침·보존 기간**을 따릅니다. 저장소 기본 경로는 `edu_tools/data/` 및 백업 루트(`TEAM_DATA_BACKUP_ROOT`)입니다.
- **학생에게 자동 결과를 그대로 공개하지 말 것** — 응답 `disclaimer`·`practical_toolkit`에 교육자 검수 전제가 있습니다.
- 교육자 무임승차 오버라이드는 **면담·감사 기록**과 함께 쓰도록 설계되었습니다(API·프론트 필드).

---

## 3. 이의제기·공정성

- 자동 탐지(무임승차·불일치·DL)는 **추정**입니다. 이의가 있으면 **추가 증거(커밋 로그, 회의록, 역할 분담표)**와 **교육자 최종 판정**을 우선합니다.
- `POST /api/team/evaluate` 의 `instructor_freerider_override` / `instructor_override_note` 로 화면 플래그를 정정할 수 있습니다(Rule 상세 리포트는 참고용으로 유지).

---

## 4. 의존성·성능

- Gemini 호출은 **`google-genai`** (`from google import genai`) SDK를 사용합니다. 구 `google-generativeai` 패키지는 의존성에서 제거되었습니다.
- PyTorch·대용량 JSONL은 환경 변수로 상한·리저버를 조절합니다(`TEAM_TRAIN_*`, `TEAM_DATA_*` 등 — 코드·`GET /api/capabilities` 참고).

---

## 5. 접근성 (프론트)

- 본문으로 이동하는 **스킵 링크**, 주 메뉴 `aria-current="page"` 를 제공합니다. 완전한 WCAG 준수를 위해서는 주기적인 수동 감사가 필요합니다.

---

## 6. 알려진 한계

- 분당 요율 제한은 **단일 프로세스 메모리** 기준이며, 수평 확장 시 공유 저장소 없이는 노드별로 따로 적용됩니다.
- 보안 헤더만으로 애플리케이션 취약점이 사라지지는 않습니다. TLS 종료, 패치, 비밀 관리는 인프라에서 별도로 다루어야 합니다.

---

*실제 운영 시 기관 규정·개인정보 보호·장애 대응 절차를 우선합니다.*
