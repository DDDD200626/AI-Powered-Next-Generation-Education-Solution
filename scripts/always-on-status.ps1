$ErrorActionPreference = "SilentlyContinue"
$TaskName = "StockPredictAlwaysOn"
$StartupCmd = Join-Path ([Environment]::GetFolderPath("Startup")) "stock-predict-always-on.cmd"

$t = Get-ScheduledTask -TaskName $TaskName
if ($t) {
  $info = Get-ScheduledTaskInfo -TaskName $TaskName
  Write-Host "Task: $TaskName" -ForegroundColor Cyan
  Write-Host "State: $($t.State)"
  Write-Host "LastRun: $($info.LastRunTime)"
  Write-Host "LastResult: $($info.LastTaskResult)"
} else {
  Write-Host "Task: not installed" -ForegroundColor Yellow
}

if (Test-Path $StartupCmd) {
  Write-Host "Startup entry: installed ($StartupCmd)" -ForegroundColor Cyan
} else {
  Write-Host "Startup entry: not installed" -ForegroundColor Yellow
}

try {
  $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/live" -UseBasicParsing -TimeoutSec 3
  Write-Host "HTTP: $($r.StatusCode) http://127.0.0.1:8000/api/live" -ForegroundColor Green
} catch {
  Write-Host "HTTP: FAIL http://127.0.0.1:8000/api/live" -ForegroundColor Red
}
