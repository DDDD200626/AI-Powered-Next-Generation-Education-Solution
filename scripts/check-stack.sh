#!/usr/bin/env bash
# HTTP·Docker 상태 점검 (Linux/macOS/WSL)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

probe() {
  local name=$1 url=$2
  if command -v curl >/dev/null 2>&1; then
    if curl -sfS --max-time 5 -o /dev/null "$url"; then
      echo "  OK   $name  $url"
    else
      echo "  FAIL $name  $url"
    fi
  else
    echo "  SKIP $name (curl 없음)"
  fi
}

echo "== HTTP =="
probe "nginx(8080)" "http://127.0.0.1:8080/"
probe "API live(8000)" "http://127.0.0.1:8000/api/live"
probe "Vite dev(5173)" "http://127.0.0.1:5173/"

echo ""
echo "== Docker =="
if command -v docker >/dev/null 2>&1; then
  docker compose ps || true
else
  echo "  docker 없음"
fi

echo ""
echo "개발: npm run dev → http://127.0.0.1:5173"
echo "Docker: ./scripts/stack-up.sh → http://127.0.0.1:8080"
echo "안내: START.txt"
