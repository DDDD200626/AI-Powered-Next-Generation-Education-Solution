# 같은 Wi-Fi 등 LAN에서 접속할 때 쓸 URL 안내 (관리자 권한 불필요)
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host ""
Write-Host "=== 다른 기기에서 접속 (같은 네트워크) ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "1) 이 PC에서 먼저 실행:  npm run dev:public" -ForegroundColor Yellow
Write-Host "   (또는 Docker: docker compose up --build → 포트 8080)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "2) 아래 주소를 휴대폰/노트북 브라우저에 입력 (같은 Wi-Fi):" -ForegroundColor Yellow
Write-Host ""

$addrs = @()
try {
  $addrs = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
    Where-Object {
      $_.IPAddress -notlike '127.*' -and
      $_.IPAddress -notlike '169.254.*' -and
      $_.PrefixOrigin -ne 'WellKnown'
    } |
    Select-Object -ExpandProperty IPAddress -Unique
} catch {
  $addrs = @()
}

if (-not $addrs -or $addrs.Count -eq 0) {
  Write-Host "   (IPv4 주소를 자동으로 못 찾았습니다. ipconfig 로 본인 IP를 확인하세요.)" -ForegroundColor DarkYellow
} else {
  foreach ($ip in $addrs) {
    Write-Host "   개발(Vite):  http://${ip}:5173/" -ForegroundColor Green
    Write-Host "   Docker 웹:   http://${ip}:8080/" -ForegroundColor Green
    Write-Host ""
  }
}

Write-Host "3) 접속이 안 되면 Windows 방화벽에서 TCP 5173(개발) 또는 8080(Docker) 인바운드 허용을 확인하세요." -ForegroundColor DarkGray
Write-Host ""
