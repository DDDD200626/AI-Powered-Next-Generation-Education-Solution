# 방법 A — 백엔드(FastAPI) + 프론트(Vite) 개발 세팅
param(
    [switch]$WithClassPulse
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "== [1/3] Backend (Python) ==" -ForegroundColor Cyan
Push-Location backend
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "  backend\.env 생성됨 — API 키를 채우세요." -ForegroundColor Yellow
}
python -m pip install -r requirements.txt
Pop-Location

Write-Host "== [2/3] Node ==" -ForegroundColor Cyan
npm install

Write-Host "== [3/3] Frontend ==" -ForegroundColor Cyan
npm run install:all

if ($WithClassPulse) {
    Write-Host "== [선택] ClassPulse ==" -ForegroundColor Cyan
    python -m pip install -r classpulse/requirements.txt
}

Write-Host ""
Write-Host "세팅 완료" -ForegroundColor Green
Write-Host "  항상 주소 모드 설치: npm run always:on"
Write-Host "  주소:                http://127.0.0.1:8000"
Write-Host "  상태 확인:           npm run always:status"
Write-Host "  중지/제거:           npm run always:off"
Write-Host "  개발(HMR):           npm run dev"
