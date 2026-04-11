# 로컬 상태 점검
$ErrorActionPreference = "SilentlyContinue"

function Probe([string]$Name, [string]$Url) {
    try {
        $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        Write-Host "  OK   $Name  ($($r.StatusCode))  $Url" -ForegroundColor Green
    } catch {
        Write-Host "  FAIL $Name  $Url" -ForegroundColor Red
    }
}

Write-Host "== HTTP ==" -ForegroundColor Cyan
Probe "Site (always-on)" "http://127.0.0.1:8000/"
Probe "API live" "http://127.0.0.1:8000/api/live"
Probe "Vite dev" "http://127.0.0.1:5173/"

Write-Host ""
Write-Host "== Always-on Task ==" -ForegroundColor Cyan
& powershell -ExecutionPolicy Bypass -File "scripts/always-on-status.ps1"
