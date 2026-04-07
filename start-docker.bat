@echo off
cd /d "%~dp0"
echo Docker로 웹(8080) + API(8000) 실행...
docker compose up -d --build
echo 브라우저 주소: http://localhost:8080
