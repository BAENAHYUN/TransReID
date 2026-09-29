# 배포 zip 만들기 — 커밋된 코드만 담는다 (git archive). 가중치·데이터·DB·실험 산출물은 넣지 않는다.
#   powershell -ExecutionPolicy Bypass -File scripts\make_release.ps1
# 결과: dist\TransReID_<날짜>_<커밋>.zip 과 .sha256. 받은 사람은 zip 을 풀고 scripts\setup.ps1 부터 실행한다.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$dirty = & git status --porcelain --untracked-files=no
if ($dirty) { Write-Host "[경고] 커밋되지 않은 변경이 있습니다 — zip 에는 마지막 커밋만 들어갑니다:"; $dirty | Select-Object -First 10 | ForEach-Object { Write-Host "   $_" } }
$commit = (& git rev-parse --short HEAD).Trim()
$name = "TransReID_" + (Get-Date -Format "yyyyMMdd") + "_" + $commit
New-Item -ItemType Directory -Force -Path (Join-Path $Root "dist") | Out-Null
$zip = Join-Path $Root "dist\$name.zip"
# 실행에 필요 없는 큰 폴더: 평가 캐시(eval_cache 375MB) · 옛 코드 사본(src 224MB) · 작업 산출물(outputs) · 개발용(.claude, code_bundle, person_debug)
$exclude = @("eval_cache", "src", "outputs", "code_bundle", "person_debug", ".claude")
$spec = @(".") + ($exclude | ForEach-Object { ":(exclude)$_" })
& git archive --format=zip --prefix="$name/" -o $zip HEAD -- @spec
if ($LASTEXITCODE -ne 0) { throw "git archive 실패" }
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
Set-Content -Path "$zip.sha256" -Value "$hash  $name.zip" -Encoding ascii
$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "== $zip ($mb MB)"
Write-Host "   sha256 $hash"
