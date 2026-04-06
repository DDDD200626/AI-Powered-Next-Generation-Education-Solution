# 팀 프로젝트 기여도 자동 평가 시스템

Git 수치·태스크·자기·동료 서술을 입력하면 **AI(OpenAI)** 또는 **휴리스틱**으로 팀원별 기여도 추정을 돌려줍니다. 교육 과제 보조용이며 최종 성적을 대체하지 않습니다.

## 구성

- **백엔드**: `backend/` — FastAPI (`team_eval`)
- **프론트엔드**: `frontend/` — Vite + TypeScript

## 실행

### 1. 백엔드

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env
# .env에 OPENAI_API_KEY 설정 (선택)
uvicorn team_eval.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

브라우저에서 `http://127.0.0.1:5173` — 개발 시 Vite가 `/api`를 백엔드로 프록시합니다.

### API만 쓸 때

`POST /api/evaluate`에 JSON 본문(`TeamEvaluateRequest` 스키마)을 보냅니다. OpenAPI 문서: `http://127.0.0.1:8000/docs`

### 빌드 배포

프론트가 API와 다른 도메인이면 빌드 시 API 베이스를 지정합니다.

```bash
cd frontend
set VITE_API_BASE=https://your-api.example.com
npm run build
```

## 라이선스

프로젝트 정책에 따릅니다.
