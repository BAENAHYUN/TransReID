<#
  실험 2용 쿼리셋 생성
  ====================
  forensic_person 컬렉션에서 source=video_tracks_stitched (영상 track) 포인트를
  track_key 기준 1개씩(다양성 확보) 샘플링해서 crop_path 목록을 만든다.

  PRW 포인트는 여기 쿼리셋에 넣지 않는다 — PRW는 cluster_leiden_id 가 전부
  null(43,343/43,343 확인됨)이라 grouping 효과가 절대 안 보이기 때문.
  클러스터링이 실제로 적용된 건 영상 track 132,105건뿐이라 쿼리도 거기서 뽑는다.

  사용:
    .\eval\build_video_query_set.ps1
    .\eval\build_video_query_set.ps1 -SampleSize 30
#>
param(
    [string]$QdrantUrl = "http://localhost:6333",
    [string]$Collection = "forensic_person",
    [int]$SampleSize = 30,
    [string]$OutFile = "eval/queryset_video_person.json"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$outDir = Split-Path $OutFile
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

function Scroll-AllPoints {
    param($Filter)
    $all = @()
    $offset = $null
    do {
        $body = @{
            limit        = 1000
            with_payload = @("crop_path", "track_key", "cluster_leiden_id")
            with_vector  = $false
            filter       = $Filter
        }
        if ($offset) { $body.offset = $offset }
        $json = $body | ConvertTo-Json -Depth 6
        $resp = Invoke-RestMethod -Uri "$QdrantUrl/collections/$Collection/points/scroll" `
            -Method Post -Body $json -ContentType "application/json" -TimeoutSec 30
        $all += $resp.result.points
        $offset = $resp.result.next_page_offset
    } while ($null -ne $offset -and $resp.result.points.Count -gt 0)
    return $all
}

Write-Host "forensic_person / source=video_tracks_stitched 스크롤 중..."
$filter = @{ must = @(@{ key = "source"; match = @{ value = "video_tracks_stitched" } }) }
$points = Scroll-AllPoints -Filter $filter
Write-Host "video 포인트 총합: $($points.Count)"

# track_key 당 1개만 남겨서 같은 track에서 여러 개 뽑히는 것을 방지.
# crop_path가 없거나 실제 파일이 없는 포인트는 제외.
$byTrack = @{}
foreach ($p in $points) {
    $tk = $p.payload.track_key
    $cp = $p.payload.crop_path
    if (-not $tk -or -not $cp) { continue }
    if ($byTrack.ContainsKey($tk)) { continue }
    if (-not (Test-Path $cp)) { continue }
    $byTrack[$tk] = $p
}
Write-Host "사용 가능한 distinct track: $($byTrack.Count)"

if ($byTrack.Count -eq 0) {
    Write-Error "쿼리로 쓸 수 있는 track이 없습니다 (crop 파일 경로 확인 필요)."
    exit 1
}

$n = [Math]::Min($SampleSize, $byTrack.Count)
$candidates = $byTrack.Values | Get-Random -Count $n

$queries = @()
$i = 0
foreach ($c in $candidates) {
    $i++
    $queries += [PSCustomObject]@{
        query_id          = "vq{0:D3}" -f $i
        image_path        = $c.payload.crop_path
        query_point_id    = "$($c.id)"
        track_key         = $c.payload.track_key
        cluster_leiden_id = $c.payload.cluster_leiden_id
    }
}

$queries | ConvertTo-Json -Depth 5 | Set-Content -Path $OutFile -Encoding UTF8
Write-Host "쿼리셋 저장: $OutFile ($($queries.Count)개)"

# 주의: cluster_leiden_id가 0(정수) 인 경우 PowerShell에서 $_.cluster_leiden_id가
# falsy로 취급되므로 truthy 체크(if ($_.cluster_leiden_id))를 쓰면 안 됨 --
# $null 여부로만 판단한다.
$withCluster = ($queries | Where-Object { $null -ne $_.cluster_leiden_id }).Count
Write-Host "  cluster_leiden_id 있는 쿼리: $withCluster / $($queries.Count)"
Write-Host "  query_point_id 저장됨 (evaluate 단계에서 self-match 제외용)"


