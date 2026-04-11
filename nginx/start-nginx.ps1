# 저장소 루트에서 nginx 기동 (포트 8080). 먼저: frontend npm run build, 백엔드 :8000
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$nginxDir = Join-Path $repoRoot "nginx"
$nginxExe = Join-Path $repoRoot "tools\nginx-win\nginx-1.26.3\nginx.exe"
$confName = "nginx.stock-predict.conf"
$dist = Join-Path $repoRoot "frontend\dist"

if (-not (Test-Path $nginxExe)) {
    Write-Error "nginx.exe 없음: $nginxExe — tools/nginx-win 설치 여부 확인"
}
if (-not (Test-Path (Join-Path $nginxDir "mime.types"))) {
    Write-Error "nginx/mime.types 없음"
}
if (-not (Test-Path (Join-Path $dist "index.html"))) {
    Write-Error "frontend/dist/index.html 없음. 먼저: cd frontend; npm ci; npm run build"
}

try {
    $busy = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
    if ($busy) {
        Write-Error "포트 8080 이 이미 사용 중입니다. docker compose web 이거나 다른 nginx 일 수 있습니다. 중지 후 다시 실행하세요."
    }
} catch {
    # 일부 환경에서는 TCP 조회가 제한될 수 있음 — nginx 기동 시 바인딩 오류로 판별
}

New-Item -ItemType Directory -Force -Path (Join-Path $nginxDir "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $nginxDir "temp") | Out-Null

Push-Location $nginxDir
try {
    & $nginxExe -t -p "$nginxDir\" -c $confName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $nginxExe -p "$nginxDir\" -c $confName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}

Write-Host "OK: http://127.0.0.1:8080  (중지: nginx\stop-nginx.ps1)"
