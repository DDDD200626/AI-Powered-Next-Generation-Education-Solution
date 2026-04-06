# ClassPulse AI (공모전 제출)

**AI 활용 차세대 교육 솔루션** — LMS 구축이 아니라, 교강사·수강생·운영자의 **피드백·근거 기반 답변·운영 요약** 업무를 AI로 보조합니다.

## 무엇이 있는지

| 역할 | 기능 |
|------|------|
| 수강생 | `corpus` 강의 자료 범위에서 **TF-IDF 검색** 후, OpenAI로 **출처 번호가 있는 답변** 생성 |
| 교강사 | 루브릭 + 학생 제출 → **피드백 초안** (교사 검수 전제) |
| 운영자 | 주차별 제출률·지각 등 샘플(또는 업로드 CSV) → **자연어 운영 요약** |

## 로컬 실행 (백엔드 + 프론트 분리)

**1) 백엔드 (FastAPI JSON API)** — 터미널 A

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env   # OPENAI_API_KEY 는 서버(백엔드)만
python -m classpulse.app
```

→ API: `http://127.0.0.1:8000` · 문서: `http://127.0.0.1:8000/docs`

**2) 프론트엔드 (Vite)** — 터미널 B

```bash
cd frontend
npm install
npm run dev
```

→ UI: `http://127.0.0.1:5173` (개발 시 `/api` 는 Vite가 8000번으로 프록시)

API 키 없이도 근거 검색·휴리스틱은 동작하고, 답변·요약·평가 생성에는 백엔드 `OPENAI_API_KEY` 가 필요합니다.

## 배포

- **API만:** `Procfile` 과 같이 `uvicorn classpulse.api_app:app` + `OPENAI_API_KEY` + 필요 시 `CORS_ORIGINS`(프론트 도메인 콤마 구분).
- **한 컨테이너에 UI+API:** `cd frontend && npm ci && npm run build` 후 환경 변수 `FRONTEND_DIST=frontend/dist` (저장소 루트 기준) 를 백엔드에 설정하면 정적 파일을 같은 호스트에서 서빙합니다.
- 프론트만 별도 호스트(Vercel 등)일 때: 빌드 시 `VITE_API_BASE=https://your-api.example.com` 로 API 절대 URL 지정.

## 기존 주식 데모 (CLI)

루트 `app.py`는 **한국 주식 방향 예측** CLI입니다. ClassPulse와 별도입니다.

```bash
python app.py --ticker 005930.KS --period 5y
python app.py --mode krx --krx-limit 10 --period 1y
```

## 제출 시 확인

- [ ] 저장소 public, API 키 미포함  
- [ ] 라이브 URL 동작  
- [ ] AI 리포트 PDF, 동의서 PDF 메일 제출
