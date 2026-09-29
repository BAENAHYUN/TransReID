# TransReID 설치 스크립트 (Windows PowerShell) — 새 PC 에서 한 번 실행한다.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# 하는 일: .venv 생성(Python 3.11) → CUDA 12.8 PyTorch → requirements.lock.txt(검증 환경 그대로) 설치
#          → 빈 폴더(weights/, data/, storage/) 준비 → 다음 할 일 안내.
# 옵션: -Python "C:\path\python.exe"  (py 런처가 없을 때)   -NoLock  (잠금 파일 대신 requirements.txt + 필수 패키지)
param(
    [string]$Python = "",
    [switch]$NoLock
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "== TransReID 설치: $Root"

$TorchIndex = "https://download.pytorch.org/whl/cu128"
$PygLinks = "https://data.pyg.org/whl/torch-2.11.0+cu128.html"

# 1) 가상환경
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    if ($Python) { & $Python -m venv .venv }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { & py -3.11 -m venv .venv }
    else { & python -m venv .venv }
    if (-not (Test-Path $venvPy)) { throw ".venv 를 만들지 못했습니다. Python 3.11 을 설치하거나 -Python 으로 경로를 주세요." }
}
& $venvPy -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; print('python', sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) { throw ".venv 가 Python 3.11 이 아닙니다. .venv 를 지우고 3.11 로 다시 만드세요." }
& $venvPy -m pip install --upgrade pip

# 2) PyTorch (CUDA 12.8) 먼저 — 다른 패키지가 CPU 판 torch 를 끌어오지 않게
& $venvPy -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch 설치 실패" }

# 3) 나머지 패키지
if (-not $NoLock) {
    & $venvPy -m pip install -r requirements.lock.txt --extra-index-url $TorchIndex -f $PygLinks
} else {
    & $venvPy -m pip install -r requirements.txt --extra-index-url $TorchIndex
    & $venvPy -m pip install PySide6==6.11.2 rfdetr==1.9.3 ultralytics==8.4.163 boxmot==25.0.0 lapx==0.10.0 filterpy==1.4.5 `
        leidenalg==0.12.0 igraph==1.0.0 python-igraph==1.0.0 optuna==5.0.0 sentencepiece==0.2.2 sacremoses==0.2.0
    & $venvPy -m pip install torch-geometric==2.8.0.post1
    & $venvPy -m pip install torch-scatter torch-sparse pyg-lib -f $PygLinks
}
if ($LASTEXITCODE -ne 0) { throw "패키지 설치 실패 — 위 오류를 확인하세요 (-NoLock 으로 다시 시도할 수 있습니다)" }

# 4) 데이터 폴더 (git 에 없는 것)
foreach ($d in @("weights\IRRA\cuhk_pedes", "weights\SOLIDER", "data", "storage", "outputs")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null
}

Write-Host ""
Write-Host "== 설치 완료. 다음 순서:"
Write-Host "  1. Qdrant:   docker compose -f docker-compose.qdrant.yml up -d"
Write-Host "  2. 가중치:   README 2.4 표대로 weights\ 와 third_party\SUSHI\ 에 넣기 (SigLIP2·DINOv2·Qwen 은 첫 실행 때 자동 다운로드)"
Write-Host "  3. 점검:     powershell -ExecutionPolicy Bypass -File scripts\check_install.ps1"
Write-Host "  4. 실행:     run_gui.bat"
