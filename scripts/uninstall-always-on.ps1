$ErrorActionPreference = "SilentlyContinue"
$TaskName = "StockPredictAlwaysOn"
$StartupCmd = Join-Path ([Environment]::GetFolderPath("Startup")) "stock-predict-always-on.cmd"

Stop-ScheduledTask -TaskName $TaskName | Out-Null
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false | Out-Null
if (Test-Path $StartupCmd) { Remove-Item $StartupCmd -Force }

# Kill loop powershell instances started by this project path
Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "powershell.exe" -and $_.CommandLine -like "*always-on-loop.ps1*"
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Write-Host "Removed always-on autostart." -ForegroundColor Yellow
