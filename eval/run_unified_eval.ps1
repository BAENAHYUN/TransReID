<#
  통합검색(2단계) PRW GT 평가
  ===========================
  GUI person 이미지 검색 경로(SigLIP2+IRRA 가중 RRF 후보 200 → SOLIDER 재정렬)를
  PRW GT crop 위에서 오프라인 재현해 mAP / Rank-k 를 잰다 (eval/prw_eval_unified.py).
  메모리 압박(15.7GB RAM)으로 3개 모델을 한 프로세스에서 올리면 torch 가 죽을 수 있어
  모델별 임베딩(캐시 저장)을 따로 실행한 뒤 마지막에 채점만 한다.

  사용:
    .\eval\run_unified_eval.ps1
    .\eval\run_unified_eval.ps1 -Device cuda:0 -Pools "50,100,500,1000"
  결과: eval/results/unified_eval.json, unified_eval.csv (+ 캐시 eval/results/cache/prw_gt_<model>.npz)
#>
param(
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$Config = "pipeline.yaml",
    [string]$DataRoot = "./data/PRW",
    [string]$Device = "cuda:0",
    [string]$Pools = "50,100,500,1000",
    [int]$Pool = 200,
    [int]$Prefetch = 200,
    [string[]]$Models = @("siglip2", "irra", "solider"),
    [switch]$SkipEmbed
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

if (-not $SkipEmbed) {
    foreach ($m in $Models) {
        Write-Host "=== [$m] GT crop 임베딩 (캐시) ===" -ForegroundColor Cyan
        & $PythonExe -u eval/prw_eval_unified.py --config $Config --data-root $DataRoot --device $Device --only-embed $m
        if ($LASTEXITCODE -ne 0) { throw "[$m] 임베딩 실패 (exit $LASTEXITCODE)" }
    }
}

Write-Host "=== 통합검색 변형 채점 ===" -ForegroundColor Cyan
& $PythonExe -u eval/prw_eval_unified.py --config $Config --data-root $DataRoot --device $Device `
    --pool $Pool --prefetch $Prefetch --pools $Pools
if ($LASTEXITCODE -ne 0) { throw "채점 실패 (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "=== 결과 (eval/results/unified_eval.csv) ===" -ForegroundColor Green
Import-Csv "eval/results/unified_eval.csv" | Select-Object variant, mAP, "Rank-1", "Rank-5", "Rank-10", "pool_recall(%)" | Format-Table -AutoSize
