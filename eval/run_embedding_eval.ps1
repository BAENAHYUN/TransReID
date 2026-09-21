<#
  실험 1: 임베딩별 단독 검색 성능평가
  ====================================
  PRW GT(query_box + annotation)를 직접 써서 Qdrant/실제 검색 파이프라인을
  거치지 않고 SigLIP2 / IRRA / SOLIDER 각 임베딩 모델만 단독으로 Rank-1 /
  Rank-5 / mAP 비교한다. eval/prw_eval.py (기존 스크립트, 수정 없음) 를 모델별로
  3번 실행해서 결과를 하나의 표로 합친다.

  사용:
    .\eval\run_embedding_eval.ps1
    .\eval\run_embedding_eval.ps1 -DataRoot "./data/PRW" -OutDir "eval/results"
#>
param(
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$DataRoot = "./data/PRW",
    [string]$OutDir = "eval/results",
    [string[]]$Models = @("siglip2", "irra", "solider")
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

$summary = @()

foreach ($model in $Models) {
    $outJson = Join-Path $OutDir "embedding_$model.json"
    Write-Host ""
    Write-Host "=== [$model] PRW 평가 시작 ===" -ForegroundColor Cyan

    & $PythonExe eval/prw_eval.py --model $model --data-root $DataRoot --save-result $outJson

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[$model] 평가 실패 (exit $LASTEXITCODE) - 건너뜀"
        continue
    }
    if (-not (Test-Path $outJson)) {
        Write-Warning "[$model] 결과 파일이 생성되지 않았습니다: $outJson"
        continue
    }

    $result = Get-Content $outJson -Raw | ConvertFrom-Json
    $summary += [PSCustomObject]@{
        Model         = $result.model
        GallerySize   = $result.gallery_size
        ValidQueries  = $result.valid_queries
        "mAP(%)"      = $result.mAP
        "Rank-1(%)"   = $result."Rank-1"
        "Rank-5(%)"   = $result."Rank-5"
        "Rank-10(%)"  = $result."Rank-10"
    }
}

Write-Host ""
Write-Host "=== 실험 1 결과: 임베딩별 단독 검색 성능 (PRW GT) ===" -ForegroundColor Green
$summary | Format-Table -AutoSize

$summaryCsv = Join-Path $OutDir "embedding_summary.csv"
$summary | Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8
Write-Host "저장: $summaryCsv"

