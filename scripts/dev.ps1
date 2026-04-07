# 개발: API(8000) + Vite(5173). 사이트는 브라우저 주소 http://localhost:5173 만 입력해 엽니다.
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
npm run dev
