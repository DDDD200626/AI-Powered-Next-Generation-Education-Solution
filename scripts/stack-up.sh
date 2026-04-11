#!/usr/bin/env bash
# Linux/WSL/macOS: Docker 풀스택 — 컨테이너 nginx(80→호스트 8080) + API(8000)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker 가 없습니다. Linux 에서 Docker Engine 을 설치한 뒤 다시 실행하세요." >&2
  exit 1
fi

docker compose up --build -d

wait_http() {
  local url=$1
  local secs=${2:-90}
  local elapsed=0
  while [ "$elapsed" -lt "$secs" ]; do
    if command -v curl >/dev/null 2>&1; then
      curl -sfS --max-time 5 "$url" >/dev/null 2>&1 && return 0
    elif command -v wget >/dev/null 2>&1; then
      wget -q --timeout=5 -O /dev/null "$url" 2>/dev/null && return 0
    else
      echo "curl/wget 없음 — URL 대기 생략. START.txt 참고." >&2
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  return 1
}

echo ""
if wait_http "http://127.0.0.1:8080/" 60; then
  echo "웹(8080) 응답 OK."
else
  echo "경고: 8080 이 아직 응답하지 않습니다. docker compose logs web" >&2
fi

if wait_http "http://127.0.0.1:8000/api/live" 120; then
  echo "API(8000) 응답 OK."
else
  echo "경고: API 가 늦게 뜨는 중일 수 있습니다. docker compose logs api — 잠시 후 새로고침." >&2
fi

echo ""
echo "Stack:"
echo "  Site:  http://127.0.0.1:8080"
echo "  API:   http://127.0.0.1:8000/docs"
echo "  안내:  START.txt"
echo "  다른 PC: 이 머신 IP:8080 (방화벽 8080·8000 허용)"
