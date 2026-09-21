<#
  실험 2: 클러스터링 적용에 따른 검색 결과 중복 감소 효과 평가
  ============================================================
  "검색 정확도(Rank-1 등 GT 기반 지표)" 가 아니라 "검색 결과 중복/다양성"만 본다.
  영상 track에는 실제 동일인 GT가 없으므로, cluster_leiden_id를 identity처럼
  써서 정답을 맞았다/틀렸다고 말하지 않는다 (그러면 Leiden 결과로 Leiden을
  평가하는 순환 평가가 된다).

  동일 Top-M candidate pool에서:
    Baseline : 1차 retrieval 결과 그대로
    Leiden   : cluster_leiden_id 기준 grouping 후 대표만
  을 비교한다. rerank/verify는 켜지 않는다 (clustering 단독 효과 격리).

  search.py --log-jsonl 은 --group-by-cluster 여부와 무관하게 baseline/leiden
  두 조건을 raw pool 하나에서 항상 같이 계산해서 남기므로 (grouping_log_rows),
  이 스크립트는 --group-by-cluster 를 켜지 않고 로그만 축적한다.

  사용:
    .\eval\build_video_query_set.ps1          # 쿼리셋 먼저 생성
    .\eval\run_clustering_dup_eval.ps1
    .\eval\summarize_clustering_dup.ps1        # 집계
#>
param(
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$QuerySetFile = "eval/queryset_video_person.json",
    [string]$LogJsonl = "eval/results/clustering_log.jsonl",
    [int]$CandidateLimit = 100,   # M
    [int]$Limit = 10,             # K
    [string]$Scope = "person",
    [string]$Config = "pipeline.yaml"
)

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Test-Path $QuerySetFile)) {
    Write-Error "쿼리셋이 없습니다: $QuerySetFile (먼저 build_video_query_set.ps1 실행)"
    exit 1
}

$outDir = Split-Path $LogJsonl
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}
if (Test-Path $LogJsonl) {
    Write-Warning "기존 로그 삭제: $LogJsonl"
    Remove-Item $LogJsonl
}

$queries = Get-Content $QuerySetFile -Raw | ConvertFrom-Json
Write-Host "쿼리 수: $($queries.Count)  M(candidate-limit)=$CandidateLimit  K(limit)=$Limit"
Write-Host "주의: 모델 로딩이 쿼리마다 새로 일어나서(별도 프로세스) 느릴 수 있습니다."

$i = 0
$failed = 0
foreach ($q in $queries) {
    $i++
    Write-Host ""
    Write-Host "[$i/$($queries.Count)] $($q.query_id) -> $($q.image_path)"

    & $PythonExe search.py `
        --image $q.image_path `
        --scope $Scope `
        --candidate-limit $CandidateLimit `
        --limit $Limit `
        --config $Config `
        --log-jsonl $LogJsonl `
        --query-id $q.query_id `
        --no-rerank

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "  실패 (exit $LASTEXITCODE) - 건너뜀"
        $failed++
    }
}

Write-Host ""
Write-Host "완료: $($queries.Count - $failed)/$($queries.Count) 성공"
Write-Host "로그: $LogJsonl"
Write-Host "다음: .\eval\summarize_clustering_dup.ps1 -Limit $Limit"

