# 팀 프로젝트 기여도 자동 평가 시스템

팀 단위 과제에서 **정량·서술·주차별 활동**을 입력하면 **기여 지수**, **무임승차 의심**, **기여도 타임라인**, **팀원별 AI 피드백**을 자동 생성하는 **교육 보조** 웹·API입니다. 출력은 **참고용**이며 최종 평가·징계를 대체하지 않습니다.

## 심사기준 부합 (요약)

| 기준 | 본 솔루션에서의 근거 |
|------|----------------------|
| **기술적 완성도** | FastAPI·Vite 분리, Pydantic 스키마, OpenAPI(`/docs`), **GitHub Actions CI**(import·**pytest**·프론트 빌드), 응답 **`X-Request-ID`**·`GET /api/version`, 팀 평가 고급 모듈(불일치·네트워크·역할·이상 탐지) |
| **AI 활용 능력 및 효율성** | `/api/analyze` **asyncio 병렬** 다중 LLM, `/api/llm/compare` 4모델 병렬, 팀 평가 시 OpenAI **ThreadPoolExecutor 병렬**(피드백·불일치 해설), 키 없을 때 **휴리스틱 폴백**으로 비용·환경에 맞춘 단계적 활용 |
| **기획력 및 실무 접합성** | 팀 과제 운영(지표·동료 메모·결과 점수·협업 간선), 교육 부가 도구(이탈 경보·과제 피드백·강의 Q&A·토론·루브릭)로 **수업·조교 업무 흐름**과 연결 |
| **창의성** | 기여–결과 **불일치**, **협업 네트워크**, 역할 4유형, **고급 이상 탐지**, **규칙 기반 창의 인사이트**(스토리라인·역할 레이더·면담 질문·설명 카드)·**가상 시뮬레이터** |

**핵심 API:** `POST /api/team/evaluate`  
동일 저장소에 **부가 모듈**(과정–시험 분석, 4AI 비교, 이탈 경보, 피드백 초안 등)이 포함되어 있으며, 웹에서는 **「부가 도구」** 메뉴에서 열 수 있습니다.

## 구현된 기능 (요약)

| 영역 | 설명 |
|------|------|
| **팀 기여도 (핵심)** | `POST /api/team/evaluate` — 기여 지수, 무임승차 의심, **기여 유형(개발·문서·리더·서포터)**, **기여–결과 불일치**, **협업 네트워크 그래프**, **이상 탐지(고급)**, 타임라인, 팀원별 피드백 |
| **과정 vs 시험** | `POST /api/analyze` — 다중 LLM 구조화 분석 |
| **4AI 자유 비교** | `POST /api/llm/compare` — Gemini·ChatGPT·Claude·Grok 병렬 |
| **이탈·피드백·QnA·토론·루브릭** | 각 `/api/...` (README 하단 백엔드 절 참고) |

## 한 번에 실행 (백엔드 + 프론트)

저장소 **루트**에서 (Python 의존성은 먼저 한 번 설치):

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env
cd ..
npm install
npm run dev
```

- **API**: `http://127.0.0.1:8000` — 문서: `/docs`  
- **웹**: `http://127.0.0.1:5173` — 기본 화면이 **팀 기여도 자동 평가**입니다.  
- `VITE_API_BASE` 없이 개발하면 Vite가 `/api`를 백엔드로 프록시합니다.

## 백엔드만 실행할 때

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env
uvicorn learning_analysis.main:app --reload --host 127.0.0.1 --port 8000
```

- `GET /api/health` — 제공자 키 설정 여부  
- `POST /api/team/evaluate` — 팀 기여도 자동 평가  
- `POST /api/analyze` — 학습–시험 불일치 분석  
- `POST /api/llm/compare` — 4모델 자유 응답 비교  
- `POST /api/at-risk/evaluate`, `/api/feedback/draft`, `/api/course/ask`, `/api/discussion/synthesize`, `/api/rubric/check` — 부가 도구  

### 환경 변수 요약

| 키 | 용도 |
|----|------|
| `OPENAI_API_KEY` | 팀 평가·피드백 등 품질 향상에 권장 |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Gemini |
| `ANTHROPIC_API_KEY` | Claude |
| `XAI_API_KEY` / `GROK_API_KEY` | Grok |

## 프론트엔드만 실행할 때

백엔드를 먼저 `127.0.0.1:8000` 에 띄운 뒤:

```bash
cd frontend
npm install
npm run dev
```

배포 시 다른 도메인만 쓰면 `frontend/.env.example` 참고해 `VITE_API_BASE` 설정.

### `POST /api/team/evaluate` 요청·응답 (고급 필드)

- **멤버별 `outcome_score`** (선택, 0–100): 발표·동료 평가 등 **결과 점수**. 있으면 추정 **기여 지수**와 비교해 불일치 목록(`mismatches`)·요약(`contribution_outcome_summary`)을 채웁니다.
- **`collaboration_edges`** (선택): `[{ "source": "이름", "target": "이름", "weight": 0–100 }]` — 팀원 간 상호작용. 없으면 서버가 기여 지수로 **완전 그래프**를 추정합니다.
- **응답**: `collaboration_network`(노드 좌표·간선), `anomaly_alerts`(예: `FREE_RIDER`, `NETWORK_ISOLATE`, `CONTRIBUTION_OUTCOME_GAP`), 멤버별 `contribution_type_label`·`role_scores`, `advanced_mode`(`heuristic` | `openai_enriched`).

## 테스트 (기술적 완성도)

```bash
cd backend
pip install -r requirements.txt
python -m pytest tests/ -q
```

## 성능·실무 팁

- **요청 추적**: 모든 API 응답에 **`X-Request-ID`** 헤더가 붙습니다(클라이언트가 `X-Request-ID`를 보내면 그 값을 유지).
- **버전**: `GET /api/version` — 앱 이름·버전·문서 경로. `GET /api/health` 응답에도 `version` 필드가 포함됩니다.
- **응답 메타**: 팀 평가 응답에 `request_id`, `generated_at`, `processing_ms`가 포함되어 기록·재현에 활용할 수 있습니다.
- **백엔드**: OpenAI 키가 있을 때 팀원 피드백 생성과 고급 해설 보강을 **병렬**로 호출해 지연을 줄입니다. 큰 페이로드는 **GZip**으로 압축됩니다.
- **프론트**: 평가 입력은 브라우저에 **자동 임시 저장**되며, 결과는 **JSON 내보내기·요약 복사·인쇄**로 보관할 수 있습니다.

## 주의

교육 **보조**용입니다. 자동 평가 결과는 **교수·조교 검토** 후 활용하고, 갈등·이의는 **기관 절차**를 따릅니다.
