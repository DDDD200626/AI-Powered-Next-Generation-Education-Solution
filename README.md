# EduSignal — 교육 현장 페인 포인트 × AI

교육 현장의 반복 과제(과정–시험, 팀, 이탈, 피드백, 반복 질문, 대규모 토론, 채점 공정성)를 **보조**하기 위한 API와 웹 UI입니다. 출력은 **참고용**이며 징계·최종 성적을 대체하지 않습니다.

## 구현된 기능

| 영역 | 설명 |
|------|------|
| **과정 vs 시험** | Gemini, ChatGPT, Claude, Grok 병렬 (`POST /api/analyze`). 키가 없으면 휴리스틱. |
| **팀 기여도** | 정량·서술·주차별 활동 → 기여 지수, **무임승차 의심**, **타임라인**, **팀원별 AI 피드백** (`POST /api/team/evaluate`). |
| **이탈·위험** | 주차별 참여 → 위험 지수 (`POST /api/at-risk/evaluate`). |
| **과제 피드백 초안** | 루브릭·제출물 (`POST /api/feedback/draft`). **OpenAI 필수**. |
| **강의 안내 Q&A** | 실라버스 발췌 + 질문 → 답 초안·인용 (`POST /api/course/ask`). |
| **토론 요약** | 게시글 목록 → 주제·후속 질문 (`POST /api/discussion/synthesize`). |
| **루브릭 정합** | 루브릭 + 채점 근거 → 정합 점수·격차 (`POST /api/rubric/check`). |

프론트엔드: **허브**와 상단 메뉴에서 각 도구로 이동 · **안내**에 운영 주의사항.

## 한 번에 실행 (백엔드 + 프론트 연결)

저장소 **루트**에서 (Python 의존성은 먼저 한 번 설치):

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env
cd ..
npm install
npm run dev
```

- **API**: `http://127.0.0.1:8000` — FastAPI (`/docs` 로 스키마 확인)  
- **웹 UI**: `http://127.0.0.1:5173` — Vite가 **`/api` 요청을 백엔드로 프록시**합니다.  
- 프론트는 `fetch("/api/...")` 만 사용하면 되며, 별도 `VITE_API_BASE` 없이 개발하면 됩니다.

## 백엔드만 실행할 때

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env
# GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY 등 설정
uvicorn learning_analysis.main:app --reload --host 127.0.0.1 --port 8000
```

- `GET /api/health` — 제공자 키 설정 여부  
- `POST /api/analyze` — `learning_analysis.schemas.AnalyzeRequest`  
- `POST /api/team/evaluate` — 팀 기여도  
- `POST /api/at-risk/evaluate` — 이탈 조기 경보  
- `POST /api/feedback/draft` — 피드백 초안 (OpenAI 필요)  
- `POST /api/course/ask` — 강의 안내 기반 Q&A 초안  
- `POST /api/discussion/synthesize` — 토론 스레드 요약  
- `POST /api/rubric/check` — 루브릭·채점 근거 정합 점검  
- 문서: `http://127.0.0.1:8000/docs`

### 환경 변수 요약

| 키 | 용도 |
|----|------|
| `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY` | Gemini (`GEMINI_MODEL`, 기본 `gemini-2.0-flash`) |
| `OPENAI_API_KEY` | ChatGPT — 팀·이탈·피드백·강의Q·토론·루브릭 보조, 과정–시험 파이프라인 |
| `ANTHROPIC_API_KEY` | Claude (`ANTHROPIC_MODEL`) |
| `XAI_API_KEY` 또는 `GROK_API_KEY` | Grok (`GROK_MODEL`, `XAI_BASE_URL`) |

## 프론트엔드만 실행할 때

백엔드를 **먼저** `127.0.0.1:8000` 에 띄운 뒤:

```bash
cd frontend
npm install
npm run dev
```

**프로덕션 빌드 미리보기** (`npm run preview`)도 `vite.config.ts`에 동일한 `/api` 프록시가 있어, 백엔드가 켜져 있으면 같은 방식으로 연결됩니다.

정적 파일을 **다른 도메인**에만 올릴 때는 빌드 전에 `frontend/.env.production` 등에 `VITE_API_BASE=https://백엔드-주소` 를 넣으세요 (`frontend/.env.example` 참고).

## 주의

교육 **보조**용입니다. 부정행위 판정·징계는 인간의 심사와 절차에 따르며, 모델 출력에는 오류·편향이 있을 수 있습니다.
