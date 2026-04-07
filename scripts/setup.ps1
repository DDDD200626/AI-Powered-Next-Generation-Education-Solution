# 방법 A — 백엔드(FastAPI) + 프론트(Vite) 개발 세팅만 수행합니다.
# 사용: 저장소 루트에서
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
# ClassPulse(Streamlit)까지 깔려면:
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -WithClassPulse

param(
    [switch]$WithClassPulse
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "== [방법 A] [1/3] Backend (Python) ==" -ForegroundColor Cyan
Push-Location backend
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "  backend\.env 생성됨 — API 키를 채우세요." -ForegroundColor Yellow
}
python -m pip install -r requirements.txt
Pop-Location

Write-Host "== [방법 A] [2/3] Node (루트 + concurrently) ==" -ForegroundColor Cyan
npm install

Write-Host "== [방법 A] [3/3] Frontend (Vite) ==" -ForegroundColor Cyan
npm run install:all

if ($WithClassPulse) {
    Write-Host "== [선택] ClassPulse (Streamlit) ==" -ForegroundColor Cyan
    python -m pip install -r classpulse/requirements.txt
}

Write-Host ""
Write-Host "방법 A 세팅 완료." -ForegroundColor Green
Write-Host "  실행:  npm run dev"
Write-Host "  웹:    http://127.0.0.1:5173"
Write-Host "  API:   http://127.0.0.1:8000/docs"
if (-not $WithClassPulse) {
    Write-Host "  (ClassPulse: setup.ps1 -WithClassPulse 후 streamlit run classpulse/app.py)"
}
