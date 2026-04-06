# ClassPulse: Vite 빌드 + FastAPI (정적 UI + API 동일 포트)
# docker build -t classpulse .
# docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... classpulse

FROM node:22-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FRONTEND_DIST=/app/frontend/dist

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY classpulse/ ./classpulse/
COPY --from=frontend /fe/dist ./frontend/dist

EXPOSE 8000
CMD ["uvicorn", "classpulse.api_app:app", "--host", "0.0.0.0", "--port", "8000"]
