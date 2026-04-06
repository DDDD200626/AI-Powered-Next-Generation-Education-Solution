# 팀 기여도 API — 프로덕션 실행용
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY edu_tools/ edu_tools/
COPY learning_analysis/ learning_analysis/

EXPOSE 8000
CMD ["uvicorn", "learning_analysis.main:app", "--host", "0.0.0.0", "--port", "8000"]
