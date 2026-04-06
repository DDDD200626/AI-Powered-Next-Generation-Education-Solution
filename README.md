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
npm install
npm run dev
```

- UI: http://127.0.0.1:5173 (개발 시 Vite가 `/api` → `8000` 프록시)

`OPENAI_API_KEY`는 **백엔드 환경변수만** 사용합니다. 브라우저·저장소에 넣지 마세요.

## 배포

| 방식 | 설정 |
|------|------|
| API만 | `Procfile`의 `uvicorn classpulse.api_app:app`; `OPENAI_API_KEY`; 프론트가 다른 도메인이면 `CORS_ORIGINS`에 콤마로 나열 |
| API+정적 UI 한 호스트 | `frontend`에서 `npm ci && npm run build` 후 `FRONTEND_DIST=frontend/dist` (저장소 루트 기준) |
| 프론트 단독 호스팅 | 빌드 시 `VITE_API_BASE=https://백엔드-URL` |

## 기존 주식 CLI (선택)

```bash
python app.py --ticker 005930.KS --period 5y
```

## 제출 체크리스트 (공모전)

- [ ] 저장소 public, API 키·`.env` 미커밋  
- [ ] 배포 라이브 URL 동작  
- [ ] AI 리포트 PDF, 개인정보 동의·참가 각서 PDF 메일 제출  
- [ ] 제출 기한(04/13) 이후 불필요한 커밋 지양
