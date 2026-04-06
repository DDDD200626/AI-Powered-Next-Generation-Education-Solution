# EduSignal — 학습 과정 vs 시험 결과 불일치 분석 (다중 AI)

과정 지표(LMS·캐퀴즈·과제·출석 등)와 시험 점수를 입력하면 **Google Gemini**, **OpenAI(ChatGPT)**, **Anthropic Claude**, **xAI Grok**이 병렬로 응답하여 다음을 **보조적으로** 제시합니다.

## 사이트

프론트엔드는 **홈 · 분석 · 안내** 화면과 GitHub 링크가 있는 단일 페이지 앱입니다. 로컬에서 백엔드와 함께 띄우면 API 키 연결 상태가 상단에 표시됩니다.

- 과정–시험 **불일치** 해석  
- **부정행위 의심도**(0–100, 단정 아님)  
- **학습 상태** 요약  
- **미래 예측**(이후 시험·학습 위험 등)

API 키가 하나도 없으면 **규칙 기반 휴리스틱**만 실행됩니다.

## 백엔드

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env
# GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY 중 필요한 것 설정
uvicorn learning_analysis.main:app --reload --host 127.0.0.1 --port 8000
```

- `GET /api/health` — 어떤 제공자 키가 설정됐는지 여부  
- `POST /api/analyze` — 본문 스키마는 `learning_analysis.schemas.AnalyzeRequest`  
- 문서: `http://127.0.0.1:8000/docs`

### 환경 변수 요약

| 키 | 모델 |
|----|------|
| `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY` | Gemini (`GEMINI_MODEL`, 기본 `gemini-2.0-flash`) |
| `OPENAI_API_KEY` | ChatGPT API (`OPENAI_MODEL`, 기본 `gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | Claude (`ANTHROPIC_MODEL`, 기본 `claude-3-5-sonnet-20241022`) |
| `XAI_API_KEY` 또는 `GROK_API_KEY` | Grok (`GROK_MODEL`, 기본 `grok-2-latest`, `XAI_BASE_URL` 기본 `https://api.x.ai/v1`) |

## 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

`http://127.0.0.1:5173` — Vite가 `/api`를 백엔드로 프록시합니다.

배포 시 API가 다른 도메인이면 `VITE_API_BASE`를 설정한 뒤 `npm run build` 하세요.

## 주의

본 시스템은 **교육 보조**용입니다. 부정행위 판정·징계는 인간의 심사와 절차에 따르며, 모델 출력은 오류·편향이 있을 수 있습니다.
