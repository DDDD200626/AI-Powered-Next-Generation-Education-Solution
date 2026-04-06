# ClassPulse AI — AI 활용 차세대 교육 솔루션

교강사·수강생·교육 운영자의 **실무 페인**을 줄이기 위한 **AI 보조 레이어**입니다. (LMS·온라인 강의 플랫폼 구축과 무관)

**공개 저장소:** [github.com/DDDD200626/AI-Education-Solution-Competition](https://github.com/DDDD200626/AI-Education-Solution-Competition)

## 아키텍처

| 구분 | 기술 | 역할 |
|------|------|------|
| 백엔드 | FastAPI (`classpulse/api_app.py`) | JSON REST API, RAG 인덱스 로드, OpenAI 호출(서버 키만) |
| 프론트엔드 | Vite + TypeScript (`frontend/`) | 탭 UI, `fetch`로 `/api/*` 호출 |
| 코퍼스 | `classpulse/corpus/*.md` | TF-IDF 검색·답변·평가의 근거 범위 |

```
stock-predict/
├── classpulse/          # Python 패키지 (도메인 로직 + API)
│   ├── api_app.py       # FastAPI 앱 진입점
│   ├── app.py           # 로컬: uvicorn 기동
│   ├── rag.py           # 코퍼스 청킹·TF-IDF·답변 생성
│   ├── comprehension.py # 이해도 AI 평가
│   ├── integrity.py     # 무결성 휴리스틱·보조 설명
│   ├── insights.py      # 운영·통합 대시보드 데이터/요약
│   ├── teacher_feedback.py
│   └── corpus/
├── frontend/            # Vite SPA
├── app.py               # (별도) 한국 주식 예측 CLI
├── requirements.txt
├── Procfile
└── .env.example
```

## 기능 (역할별)

| 역할 | 기능 |
|------|------|
| 수강생 | 코퍼스 범위 **TF-IDF 검색** → OpenAI **출처 번호 답변**; **이해도 점검** 리포트 |
| 교강사 | **통합 대시보드**(이해도·무결성 구간); **루브릭 피드백 초안** |
| 운영자 | 주차별 지표 CSV → **자연어 운영 요약** |
| 무결성 보조 | 휴리스틱·자료 유사도·LLM **참고 설명**(적발·징계 단정 아님) |

## 로컬 실행

**터미널 A — 백엔드**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m classpulse.app
```

- API: http://127.0.0.1:8000  
- Swagger: http://127.0.0.1:8000/docs  

**터미널 B — 프론트** ([Node.js LTS](https://nodejs.org/) 설치 필요)

```bash
cd frontend
npm install   # package-lock.json 으로 동일 버전 재현
npm run dev
```

프로덕션 번들 확인: `npm run build` → `frontend/dist/` (저장소에는 `.gitignore`로 제외)

Windows에서 `npm`은 되는데 `node`를 못 찾는다면, 터미널을 다시 열거나 PATH에 `C:\Program Files\nodejs` 가 포함됐는지 확인하세요.

- UI: http://127.0.0.1:5173 (개발 시 Vite가 `/api` → `8000` 프록시)

#### PWA — 앱처럼 설치 (홈 화면 / 브라우저 “앱 설치”)

`npm run build` 후 **HTTPS**로 배포된 사이트에서 **홈 화면에 추가** 또는 **앱 설치**를 사용하면 전체 화면(standalone)으로 쓸 수 있습니다. API는 여전히 백엔드가 제공해야 합니다(같은 도메인 Docker 배포면 별도 `VITE_API_BASE` 없이 가능).

**앱 스토어용 네이티브 앱**은 [Capacitor](https://capacitorjs.com/) 등으로 이 프론트를 감싸는 추가 작업이 필요합니다.

`OPENAI_API_KEY`는 **백엔드 환경변수만** 사용합니다. 브라우저·저장소에 넣지 마세요.

## 배포

| 방식 | 설정 |
|------|------|
| **Docker (UI+API 한 포트)** | 저장소 루트에서 `docker build -t classpulse .` → `docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... classpulse` → 브라우저 `http://localhost:8000` |
| API만 | `Procfile`의 `uvicorn classpulse.api_app:app`; `OPENAI_API_KEY`; 프론트가 다른 도메인이면 `CORS_ORIGINS` |
| API+정적 UI (직접) | `frontend`에서 `npm ci && npm run build` 후 `FRONTEND_DIST=frontend/dist` |
| 프론트 단독 호스팅 | 빌드 시 `VITE_API_BASE=https://백엔드-URL` |
| Render | `render.yaml` Blueprint로 연결 후 대시보드에서 `OPENAI_API_KEY` 설정 |

## CI

GitHub에 푸시하면 `.github/workflows/ci.yml`이 Python 의존성·`api_app` import·`frontend` `npm ci`/`build`를 검증합니다.

## 기존 주식 CLI (선택)

```bash
python app.py --ticker 005930.KS --period 5y
```

## 제출 체크리스트 (공모전)

- [ ] 저장소 public, API 키·`.env` 미커밋  
- [ ] 배포 라이브 URL 동작  
- [ ] AI 리포트 PDF, 개인정보 동의·참가 각서 PDF 메일 제출  
- [ ] 제출 기한(04/13) 이후 불필요한 커밋 지양
