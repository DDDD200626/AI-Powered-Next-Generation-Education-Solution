# Always-on service loop for local site (:8000)
# - Starts app in child process
# - Probes /api/live periodically
# - If probe fails repeatedly, kills the whole process tree and restarts (zombie node/uvicorn 방지)
# - 브라우저 쪽 DL 연결 보조(`frontend` health MLP)와 함께 쓰면: 서버 프로세스는 이 스크립트가, 탭↔API는 프론트가 복구
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$probeUrl = "http://127.0.0.1:8000/api/live"
$probeIntervalSec = 2
$maxConsecutiveFails = 3
$restartCooldownSec = 5
$listenPort = 8000

function Stop-ListenersOnPort {
    param([int]$Port)
    try {
        $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        foreach ($c in $conns) {
            $owning = [int]$c.OwningProcess
            if ($owning -le 0) { continue }
            Write-Host "[always-on] freeing port ${Port}: stopping PID $owning" -ForegroundColor DarkYellow
            taskkill /PID $owning /T /F 2>$null | Out-Null
        }
    } catch {
        # Get-NetTCPConnection 없는 환경은 무시
    }
    Start-Sleep -Seconds 1
}

function Test-Live {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Start-AppProcess {
    param([string]$ProjectRoot)
    $cmd = "Set-Location -LiteralPath '$($ProjectRoot.Replace("'", "''"))'; npm run serve:app"
    return Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        $cmd
    )
}

while ($true) {
    Write-Host "[always-on] Building frontend..." -ForegroundColor Cyan
    npm run build:frontend
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[always-on] build failed. retry in 20s" -ForegroundColor Yellow
        Start-Sleep -Seconds 20
        continue
    }

    Write-Host "[always-on] Starting app with watchdog on http://127.0.0.1:8000" -ForegroundColor Green
    Stop-ListenersOnPort -Port $listenPort
    $proc = Start-AppProcess -ProjectRoot $Root
    $fails = 0
    $lastRestartAt = [DateTime]::MinValue

    while ($true) {
        Start-Sleep -Seconds $probeIntervalSec

        if ($proc.HasExited) {
            Write-Host "[always-on] app process exited (code=$($proc.ExitCode)). restart." -ForegroundColor Yellow
            Stop-ListenersOnPort -Port $listenPort
            break
        }

        if (Test-Live -Url $probeUrl) {
            $fails = 0
            continue
        }

        $fails++
        Write-Host "[always-on] health fail ($fails/$maxConsecutiveFails): $probeUrl" -ForegroundColor Yellow
        if ($fails -lt $maxConsecutiveFails) {
            continue
        }

        $now = Get-Date
        $since = ($now - $lastRestartAt).TotalSeconds
        if ($since -lt $restartCooldownSec) {
            Start-Sleep -Seconds ([int][Math]::Ceiling($restartCooldownSec - $since))
        }

        Write-Host "[always-on] health degraded. restarting app process (tree kill)..." -ForegroundColor Red
        try { taskkill /PID $proc.Id /T /F 2>$null | Out-Null } catch {}
        try { $proc.WaitForExit(3000) } catch {}
        Stop-ListenersOnPort -Port $listenPort
        $lastRestartAt = Get-Date
        break
    }

    Write-Host "[always-on] loop restart in 2s" -ForegroundColor DarkYellow
    Start-Sleep -Seconds 2
}
