<#
  실험 2 결과 집계
  ================
  clustering_log.jsonl 을 읽어서 query별 / 평균 지표를 낸다.

  cluster_leiden_id 는 Leiden 알고리즘이 부여한 값일 뿐 실제 동일인 여부를
  보증하는 GT가 아니다. 여기서 "unique"는 "cluster_leiden_id 가 다르다"는 뜻이고,
  "검색 정확도"가 아니라 "결과 중복/다양성"만 측정한다.

  Self-match 제외
  ----------------
  쿼리 이미지 자체가 forensic_person 안에 이미 들어있는 point라서, 그 point가
  자기 자신의 검색 결과 후보에 그대로 나온다 (거의 완벽한 유사도로 상위권을
  차지). 이걸 그대로 두면 clustering 중복 제거 효과가 과장될 수 있으므로,
  QuerySetFile 의 query_point_id 를 이용해 pool 에서 그 point를 제거한 뒤
  baseline/leiden 지표를 계산한다. leiden 쪽은 search.py 가 이미 계산해준
  grouped_rank/selected_as_representative 를 그대로 쓰지 않고, self-match를
  뺀 나머지 pool 로 grouping(cluster_leiden_id 기준 max retrieval_score
  대표 선정 -> 대표 점수로 정렬)을 이 스크립트에서 처음부터 다시 계산한다 --
  self-match가 원래 대표였던 그룹은 다음으로 점수가 높은 멤버가 새 대표가
  되어야 하기 때문에, 기존 필드를 그대로 재사용하면 틀린다.

  용어: "Unique Group" (cluster + unclustered singleton을 합친 distinct
  group 수). "Unique Cluster"라고 부르면 cluster가 없는 point까지 cluster로
  세는 것처럼 오해될 수 있어서 group으로 표현한다. Duplicate Rate 역시
  cluster_leiden_id(Leiden이 부여한 값)만 기준으로 하는 "Leiden grouping
  기준" 지표다 -- 실제 identity 기준 중복률이 아니다.

  지표:
    unique_group_at_K_baseline     : baseline Top-K(self-match 제외) 안의
                                      distinct group 수
    duplicate_rate_at_K_baseline   : 1 - unique/K (Leiden grouping 기준)
    rank_to_reach_K_unique_baseline: self-match 제외 후 남은 pool을 순서대로
                                      훑을 때, K개의 distinct group을
                                      모으는 데 필요한 depth(자기 자신을
                                      제외한 목록에서 몇 번째인지, raw_rank
                                      원값이 아님 -- self-match 제거로
                                      한 자리씩 당겨지므로 raw_rank를 그대로
                                      쓰면 1 이상 부풀려진다.) 도달 못하면 $null
    reached_K_unique                : 위 도달에 성공했는지 (M pool 안에서
                                      distinct group이 K개 미만이면 실패)
    leiden_unique_at_K              : self-match 제외 후 다시 계산한 leiden
                                      Top-K 안의 distinct group 수 (그루핑
                                      정의상 pool에 distinct group이 K개
                                      이상이면 항상 K -- 성능 지표가 아니라
                                      sanity check)

  사용:
    .\eval\summarize_clustering_dup.ps1
    .\eval\summarize_clustering_dup.ps1 -Limit 10
#>
param(
    [string]$LogJsonl = "eval/results/clustering_log.jsonl",
    [string]$QuerySetFile = "eval/queryset_video_person.json",
    [int]$Limit = 10,
    [string]$OutCsv = "eval/results/clustering_dup_summary.csv"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Test-Path $LogJsonl)) {
    Write-Error "로그 파일이 없습니다: $LogJsonl"
    exit 1
}
if (-not (Test-Path $QuerySetFile)) {
    Write-Error "쿼리셋 파일이 없습니다: $QuerySetFile"
    exit 1
}

# cluster_leiden_id 가 0 같은 값이어도(현재 실제 값은 항상 문자열 해시거나
# null이지만) 방어적으로 truthy 체크 대신 $null 여부로만 판단한다.
function Get-GroupKey {
    param($ClusterId, $PointId)
    if ($null -ne $ClusterId) { return "cluster::$ClusterId" }
    return "singleton::$PointId"
}

$queries = Get-Content $QuerySetFile -Raw | ConvertFrom-Json
$queryPointId = @{}
foreach ($q in $queries) {
    if ($q.query_point_id) { $queryPointId[$q.query_id] = "$($q.query_point_id)" }
}

$rows = Get-Content $LogJsonl | ForEach-Object { $_ | ConvertFrom-Json }
$byQuery = $rows | Group-Object query_id

$perQuery = @()
foreach ($g in $byQuery) {
    $qid = $g.Name
    $selfId = $queryPointId[$qid]

    # baseline 조건 행에 (point_id, raw_rank, cluster_leiden_id,
    # retrieval_score) 가 다 있고, leiden 조건도 같은 pool 이므로 baseline
    # 행만 가지고 self-match 제외 + 그룹핑 재계산을 전부 처리한다.
    #
    # 아래 @() 래핑은 전부 의도적임: PowerShell 파이프라인은 결과가 정확히
    # 1건이면 배열이 아니라 단일 객체로 풀어버려서 .Count가 $null이 된다
    # (합성 테스트로 실제 재현/확인함). 매번 @() 로 강제 배열화해서 방지한다.
    $pool = @($g.Group | Where-Object { $_.condition -eq "baseline" })
    if ($selfId) {
        $before = $pool.Count
        $pool = @($pool | Where-Object { "$($_.point_id)" -ne $selfId })
        if ($pool.Count -eq $before) {
            Write-Warning "[$qid] self-match point_id($selfId)를 pool에서 못 찾음 -- candidate-limit이 너무 작았거나 쿼리 이미지가 DB에 없을 수 있음"
        }
    }
    else {
        Write-Warning "[$qid] 쿼리셋에 query_point_id가 없어 self-match 제외를 못함"
    }
    if ($pool.Count -eq 0) { continue }

    $pool = @($pool | Sort-Object raw_rank)

    # --- Baseline: self-match 제외 후 상위 K ---
    $baselineTopK = @($pool | Select-Object -First $Limit)
    $groupKeysTopK = @($baselineTopK | ForEach-Object { Get-GroupKey $_.cluster_leiden_id $_.point_id })
    $uniqueAtK = @($groupKeysTopK | Select-Object -Unique).Count
    $dupRate = 1.0 - ($uniqueAtK / [double]$Limit)

    # depth 는 self-match 제외 후 남은 pool 안에서의 순번(1부터)이다.
    # row.raw_rank 원값을 쓰면 self-match가 앞쪽에서 빠져나간 만큼 실제보다
    # 부풀려진 값이 나온다 (예: self-match가 1위였으면 전부 1씩 부풀려짐).
    $seen = @{}
    $rankToReachK = $null
    $depth = 0
    foreach ($row in $pool) {
        $depth++
        $key = Get-GroupKey $row.cluster_leiden_id $row.point_id
        if (-not $seen.ContainsKey($key)) { $seen[$key] = $true }
        if ($seen.Count -ge $Limit) { $rankToReachK = $depth; break }
    }
    $reachedK = $null -ne $rankToReachK

    # --- Leiden: self-match 제외한 pool 전체로 그룹핑을 처음부터 다시 계산 ---
    $groups = @{}
    foreach ($row in $pool) {
        $key = Get-GroupKey $row.cluster_leiden_id $row.point_id
        if (-not $groups.ContainsKey($key)) { $groups[$key] = @() }
        $groups[$key] += $row
    }
    $reps = @(foreach ($key in $groups.Keys) {
        ($groups[$key] | Sort-Object retrieval_score -Descending | Select-Object -First 1)
    })
    $leidenTopK = @($reps | Sort-Object retrieval_score -Descending | Select-Object -First $Limit)
    $leidenUniqueAtK = $leidenTopK.Count
    $distinctGroupsInPool = $groups.Keys.Count

    $perQuery += [PSCustomObject]@{
        query_id                        = $qid
        pool_size_after_selfmatch       = $pool.Count
        unique_group_at_K_baseline      = $uniqueAtK
        duplicate_rate_at_K_baseline    = [Math]::Round($dupRate, 3)
        rank_to_reach_K_unique_baseline = $rankToReachK
        reached_K_unique                = $reachedK
        leiden_unique_at_K              = $leidenUniqueAtK
        distinct_groups_in_pool         = $distinctGroupsInPool
    }
}

Write-Host "=== 쿼리별 결과 (K=$Limit, self-match 제외됨) ==="
$perQuery | Format-Table -AutoSize

$perQuery | Export-Csv -Path $OutCsv -NoTypeInformation -Encoding UTF8
Write-Host "저장: $OutCsv"

if ($perQuery.Count -gt 0) {
    $avgUniq = ($perQuery.unique_group_at_K_baseline | Measure-Object -Average).Average
    $avgDup  = ($perQuery.duplicate_rate_at_K_baseline  | Measure-Object -Average).Average
    # Where-Object 결과가 1건이면 배열이 아니라 단일 객체로 풀려서 .Count가
    # $null이 되는 PowerShell 특성 때문에 @() 로 강제 배열화한다.
    $reached = @($perQuery | Where-Object { $_.reached_K_unique })
    $reachedCount = $reached.Count
    $avgRankToReach = ($reached | Select-Object -ExpandProperty rank_to_reach_K_unique_baseline | Measure-Object -Average).Average

    Write-Host ""
    Write-Host "=== 평균 (쿼리 $($perQuery.Count)개, K=$Limit) ===" -ForegroundColor Green
    Write-Host ("Baseline 평균 Unique Group@K                    : {0:N2} / {1}" -f $avgUniq, $Limit)
    Write-Host ("Baseline 평균 Duplicate Rate@K (Leiden grouping 기준) : {0:P1}" -f $avgDup)
    Write-Host ("K개 distinct group 도달 성공        : {0}/{1}" -f $reachedCount, $perQuery.Count)
    if ($reachedCount -gt 0) {
        Write-Host ("Baseline이 Leiden Top-{0} 만큼의 다양성을 보려면 평균 {1:N1}위까지 필요 (성공한 {2}개 쿼리 기준)" -f $Limit, $avgRankToReach, $reachedCount)
    }
    if ($reachedCount -lt $perQuery.Count) {
        Write-Warning "$($perQuery.Count - $reachedCount)개 쿼리는 M(candidate-limit) 안에서도 distinct 그룹 K개를 못 모았습니다 -- candidate-limit을 늘리는 걸 고려하세요."
    }
}


