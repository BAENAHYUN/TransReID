# 설치 점검 — scripts\check_install.py 를 프로젝트 .venv 로 돌린다.
#   powershell -ExecutionPolicy Bypass -File scripts\check_install.ps1          # 환경 · 가중치 · Qdrant · GUI 정의
#   powershell -ExecutionPolicy Bypass -File scripts\check_install.ps1 -Tests   # + 단위 테스트
param([switch]$Tests)
$Root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host ".venv 가 없습니다. 먼저 scripts\setup.ps1 을 실행하세요."; exit 1 }
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$a = @((Join-Path $Root "scripts\check_install.py"))
if ($Tests) { $a += "--tests" }
& $py @a
exit $LASTEXITCODE
