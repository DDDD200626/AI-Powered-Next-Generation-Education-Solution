# Install always-on auto start (ScheduledTask first, Startup-folder fallback)
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$TaskName = "StockPredictAlwaysOn"
$Loop = Join-Path $Root "scripts\always-on-loop.ps1"
$StartupDir = [Environment]::GetFolderPath("Startup")
$StartupCmd = Join-Path $StartupDir "stock-predict-always-on.cmd"

if (-not (Test-Path $Loop)) { throw "missing $Loop" }

$installed = $false
try {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"$Loop\""
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Days 3650) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Keep stock-predict running on 127.0.0.1:8000" -Force -ErrorAction Stop | Out-Null
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    Write-Host "Installed ScheduledTask: $TaskName" -ForegroundColor Green
    $installed = $true
} catch {
    Write-Host "ScheduledTask 권한이 없어 Startup 폴더 방식으로 설치합니다." -ForegroundColor Yellow
}

if (-not $installed) {
    $line = "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Loop`"`r`n"
    [System.IO.File]::WriteAllText($StartupCmd, $line, (New-Object System.Text.UTF8Encoding($false)))
    Start-Process powershell -WindowStyle Hidden -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Loop) | Out-Null
    Write-Host "Installed Startup entry: $StartupCmd" -ForegroundColor Green
}

Write-Host "Open: http://127.0.0.1:8000" -ForegroundColor Green
