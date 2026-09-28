# bench — 자동 벤치마크 도구

로드맵(`outputs/audit/autobench_roadmap_20260926.md`)의 P1~P7 을 담는 패키지. "어떤 모델이 더 좋은지"를
같은 정답·같은 명령·같은 원장으로 재고, 남이 돌려도 같은 숫자가 나오는지 확인하는 것이 목적이다.

| 파일 | 역할 | 단계 |
|---|---|---|
| `ledger.py` | 실행 원장 — 모든 평가 결과를 `bench/ledger.jsonl` 한 곳에 append-only 로 기록·조회·이관 | P1 |
| `ledger.jsonl` | 원장 본체 (git 추적. 한 줄 = 한 실행의 한 method/variant) | — |
| `run.py` | 단일 러너 — 한 명령으로 한 단계 평가 → 실행 폴더 + 원장. `verify` 로 재현성 검증 | P2 |
| `check.py` | 계약(공통 입출력 규격) 검사 + 운영 DB 임베딩 지문 대조 | P2 (P7 준비) |
| `splits.py` | 인물(pid) 단위 tune / holdout 분할 파일 (탐색은 tune, 최종 비교는 holdout) | P2 (P3 준비) |
| `splits/prw_pids_seed42.json` | PRW 쿼리 인물 450 명 → tune 135 (쿼리 633) / holdout 315 (쿼리 1,424) | — |
| `optimize.py` | Optuna 탐색(클러스터·검색) + 검출 임계값 스윕 — tune 으로 찾고 holdout 으로 검증, 추천 yaml | P3 |
| `combos.py` | 조합 탐색 — 임베더×재정렬×후보 수 / 벡터×클러스터러 격자 → 리더보드(운영 조합 순위) → 상위 미세조정 → `pipeline_best.yaml` | P4 |
| `criteria.py` | 채택 기준(기준표 운영값)과 채택→yaml 생성기 — GUI 벤치마크 탭(`gui/bench_page.py`)이 쓴다 | P5 |
| `register.py` | 새 모델 등록기 — 어댑터 스켈레톤(template) · 계약 검사 · yaml 등록 · 단계 벤치 · 원장 순위. 가이드 `REGISTER_GUIDE.md` | P7 |
| `run.py track \| object \| qwen` | 사람 정답 3종 평가를 러너로 (스크립트는 `eval/track_gt_eval.py`, `eval/object_pair_eval.py`, `eval/qwen_verify_eval.py`; 정답·가이드 `eval/gt/`) | P6 |
| `runs/`, `studies/`, `combos/` | 러너 실행 · 스터디 · 조합 격자 폴더 (산출물·로그·원장 조각; git 제외) | — |

## 1. 설계 원칙 (외부 검토에서 받은 보완점 4개)

1. **공통 입출력 규격** — 검출기 `BaseDetector.detect → Detection(bbox, confidence, class_id, class_name)`, 임베더
   `BaseEmbedder.embed_crops → (N, DIM) float32`, 클러스터러 `BaseClusterer.cluster → ClusterResult(labels, stats)`.
   yaml 은 모듈을 고르고 어댑터가 연결한다. `check.py` 가 등록 전에 규격·생성자 인자·출력 형태를 검사한다.
2. **단독 평가와 전체 평가의 분리** — 단독: 고정 GT 입력(GT crop / GT 박스 / DB 벡터)으로 한 단계만 잰다
   (`detect / embed / search / cluster`). 전체: 실제 파이프라인이 만든 운영 DB 를 GUI 와 같은 검색 경로로 잰다
   (`e2e`, `eval/prw_e2e_search_eval.py`). e2e 의 `map`(분모 = GT 박스) 과 `map_db`(분모 = DB 에 들어간 point) 의 차이가
   검출·crop 필터의 손실이다.
3. **호환성·캐시 무효화** — 임베더 지문 = module/class/params + 가중치 파일 내용 sha256
   (`ingest.build_db.declared_retriever_fingerprint`, `ledger.retriever_fingerprint_sha`). 통합 검색 평가 캐시
   (`eval/results/cache/prw_gt_<model>.npz`) 와 클러스터링 벡터 캐시(`<접두어>_<target>_<vector>_<지문8>.npz`) 는
   이 지문으로 검증·분리된다. 운영 DB 는 `check.py --db` 가 point 의 `embedding_build_id` 매니페스트 지문과 지금 yaml 을
   대조한다 (다르면 그 벡터는 지금 임베더로 만든 것이 아니므로 재적재).
4. **공정·재현 가능한 기록** — 원장 행에 데이터 분할(`gt`, `pid_split`), 코드(`env.git_commit/git_dirty`),
   가중치(`weights` sha256), 설정(`config` yaml sha, `params`), seed, 하드웨어(`hardware`: GPU/VRAM/CUDA/CPU/RAM),
   실행 시간(`timing`), 명령(`command`, `bench.commands`) 이 남는다. 탐색(P3)은 `tune` 분할, 최종 비교는 `holdout`.

## 2. 러너 `bench/run.py`

```powershell
# 단독 평가 4단계 + 전체 파이프라인 1단계. 산출물: bench/runs/<단계_시각_이름>/, 원장 행 자동 추가
.\.venv\Scripts\python.exe bench\run.py detect  --config pipeline_tracking_yolo26.yaml --name yolo26m_test           # 검출 (6,112 프레임 ≈ 4~8분)
.\.venv\Scripts\python.exe bench\run.py detect  --module detect.detectors.yolo26_detector --class YOLO26Detector --param weights=yolo26s.pt --limit 300
.\.venv\Scripts\python.exe bench\run.py embed   --model solider                                                          # 임베더 단독 Re-ID
.\.venv\Scripts\python.exe bench\run.py search  --weights siglip2=1,irra=1.5,solider=1.5 --rrf-k 2 --prefetch 200 --pool 200   # 조합 (모든 변형)
.\.venv\Scripts\python.exe bench\run.py cluster --method leiden --param knn=30 threshold=0.97 --max-points 3000 --name leiden_t3000   # Leiden 은 threshold, DBSCAN 은 score_threshold
.\.venv\Scripts\python.exe bench\run.py cluster --method-config my_clusterer.yaml --pid-split bench\splits\prw_pids_seed42.json:tune
.\.venv\Scripts\python.exe bench\run.py e2e     --stage1 siglip2 irra --rerank solider --limit 200                       # 전체 파이프라인 (2,057 쿼리 ≈ 5~15분)
.\.venv\Scripts\python.exe bench\run.py e2e     --stage1 solider --rerank none --max-queries 200                         # 빠른 비교

# 재현성 검증: 원장의 실행을 같은 설정으로 다시 돌려 허용 오차 안인지 판정 (PASS/FAIL 행이 원장에 남음)
.\.venv\Scripts\python.exe bench\run.py verify detect_20260926T175301_0e3512c6
.\.venv\Scripts\python.exe bench\run.py verify cluster_2026… --tol pair_f1=0.01           # 판정 PASS / FAIL / UNVERIFIED(비교 지표 없음)
.\.venv\Scripts\python.exe bench\run.py show-cmd cluster_2026…          # 재현 명령만 출력
.\.venv\Scripts\python.exe bench\run.py e2e --dry-run …                  # 하위 명령만 출력
```

- 모든 단계가 같은 옵션 집합을 받는다(단계와 무관한 옵션은 무시) — GUI 평가 5단계가 한 화면으로 모든 단계를 돌린다.
- 하위 스크립트는 원장 조각(`ledger_part.jsonl`)에 쓰고, 러너가 `bench`(인자·명령·폴더·소요), `hardware`, `weights`, `env` 를
  덧붙여 본 원장에 기록한다. 스크립트를 직접 돌린 행도 `verify` 가 엔트리(component/params/gt)에서 인자를 복원해 재현한다
  (운영 DB import 행, 플러그인 정보 없는 옛 클러스터링 행은 제외).
- verify 허용 오차 기본값(`ledger.VERIFY_TOLERANCES`): 검출 AP·P/R/F1 0.005, 임베딩·검색·e2e mAP 0.1 %p / Rank 0.2 %p,
  클러스터 쌍·B³ 0.005. `verify` 행에는 `verify.against / passed / rows(지표별 Δ) / same_git_commit / same_host` 가 남는다.

## 3. 원장 `bench/ledger.py`

```powershell
.\.venv\Scripts\python.exe bench\ledger.py import                                   # eval/results 산출물 이관 (멱등)
.\.venv\Scripts\python.exe bench\ledger.py table --stage detect --latest             # 기준표 §1 재생성 (embed / search / cluster / e2e)
.\.venv\Scripts\python.exe bench\ledger.py table --stage cluster --latest --metrics pair_precision,pair_recall,b3_f1,mixed_clusters
.\.venv\Scripts\python.exe bench\ledger.py list --stage e2e --last 10
.\.venv\Scripts\python.exe bench\ledger.py show e2e_2026…
```

엔트리 스키마 (schema_version 1): `run_id` · `fingerprint`(stage, name, component, params, gt, metrics 의 sha1 — 이관 중복 방지)
· `created_at` · `stage`(detect | embed | search | cluster | e2e) · `producer` · `name` · `component` · `params` · `gt` · `metrics`
· `timing` · `versions` · `env` · `inputs` · `config` · `report` · `command` · `seed` · `note`, 러너 실행이면 `bench` · `hardware`
· `weights`, verify 면 `verify`. 표준 지표 이름은 `ledger.METRIC_KEYS`.

## 4. 검사 `bench/check.py`

```powershell
.\.venv\Scripts\python.exe bench\check.py                          # 정적: retriever 3종, pipeline_tracking*.yaml 의 detector/tracker/stitcher, 내장 클러스터러
.\.venv\Scripts\python.exe bench\check.py --instantiate             # 실제 생성 → 2장 임베딩 (2, dim) / 빈 프레임 검출 / 합성 2덩어리 클러스터링
.\.venv\Scripts\python.exe bench\check.py --db                      # 운영 DB point 의 embedding_build_id 매니페스트 vs 지금 yaml 임베더 지문
.\.venv\Scripts\python.exe bench\check.py --method-config my.yaml --instantiate --no-retrievers --no-tracking
```

새 모델 등록 절차(P7 의 수동 판): 어댑터 1개(BaseDetector / BaseEmbedder / BaseClusterer 상속) → yaml 블록 → `check.py --instantiate`
→ `run.py <단계> …` → `ledger.py table` 로 비교.

## 5. 분할 `bench/splits.py`

```powershell
.\.venv\Scripts\python.exe bench\splits.py make --seed 42            # bench/splits/prw_pids_seed42.json (tune 0.3 / holdout 0.7)
.\.venv\Scripts\python.exe bench\splits.py show bench\splits\prw_pids_seed42.json
```

`--pid-split bench/splits/prw_pids_seed42.json:tune` 을 `eval/prw_cluster_gt_eval.py`, `eval/prw_e2e_search_eval.py`, `bench/run.py cluster|e2e`
가 받는다 (그 인물의 point / 쿼리만 평가). 임베딩·통합 검색 단독 평가는 P3 에서 같은 인자를 받는다.

## 6. 탐색 `bench/optimize.py` (P3)

```powershell
# 클러스터러 파라미터 (벡터를 한 번만 받고 trial 마다 안에서 다시 군집화 → GT 평가). 기본 60 trial, tune 분할, 제약 pair_precision ≥ 0.90 아래 b3_f1 최대
.\.venv\Scripts\python.exe bench\optimize.py cluster --method leiden --trials 30 --validate --name leiden_tune
.\.venv\Scripts\python.exe bench\optimize.py cluster --method dbscan_v6 --space eps=float:0.05:0.3 --space knn=int:10:50 --constraint "pair_precision>=0.92"
# 검색 조합 (GT crop 임베딩 캐시로 1차 조합·가중치·RRF k·후보 수·재정렬을 다시 계산). 기본 40 trial, 제약 pool_recall ≥ 90 아래 mAP 최대
.\.venv\Scripts\python.exe bench\optimize.py search --trials 40 --validate --name unified_tune
# 검출 임계값 스윕 (detections.jsonl 재채점, 재검출 없음; Optuna 대신 결정적 스윕)
.\.venv\Scripts\python.exe bench\optimize.py detect --detections eval\results\detect_prw\yolo26m_test\detections.jsonl --objective f1 --constraint "recall>=0.85"
```

- PowerShell 에서 `>=` 가 든 제약은 반드시 따옴표로 감싼다 (`cmd /c` 로 넘기면 리다이렉션으로 먹힌다).
- 산출물 `bench/studies/<단계_이름_시각>/`: `trials.jsonl`(trial 마다 params·metrics·feasible·violations), `best.json`, `recommended.yaml`
  (그대로 쓸 수 있는 yaml 블록: clusterer / detector / 검색 조합), `pareto.json`(목적 vs 제약 지표의 비지배 집합), `report.md`
  (운영값 vs 추천, holdout 검증, 상위 10 trial), `optuna.db`(재개 가능). 원장에 `study:<이름>` 행 (+ `--validate` 면 `:holdout_best` / `:holdout_baseline`).
- 첫 trial 은 운영값(baseline)이라 "지금 설정이 몇 등인지" 가 같은 표에 나온다. 제약 위반 trial 은 벌점으로 밀어내고 추천은 feasible 중 최대.
- 잡음 기준(`NOISE`): 클러스터 0.002, 검색 0.1 %p, 검출 0.005 — 이보다 작은 차이는 보고서가 "잡음 이내" 로 표시한다.
- 실측 (2026-09-28): 검색 40 trial 122초 → 운영 조합 mAP 70.6 → 추천(3-RRF, SOLIDER 재정렬, prefetch 850/pool 1000) 89.4 (tune) · holdout 71.5 → 88.9
  (기준표 §4 의 3-RRF+SOLIDER 88.07 을 탐색이 스스로 찾음).

## 6b. 조합 탐색 `bench/combos.py` (P4)

```powershell
# 검색: 임베더 부분집합(7) × 재정렬(none/solider/irra/siglip2) × 후보 수(200/1000) = 50 조합을 GT crop 캐시로 전부 채점,
#        운영 조합(siglip2+irra→solider@200)의 순위, holdout 재평가, 상위 3개 Optuna 미세조정, 추천 가중치로 pipeline_best.yaml
.\.venv\Scripts\python.exe bench\combos.py search --params-from bench\studies\search_unified_tune_<시각>\best.json --validate --refine-trials 20 --top 3
# 클러스터: 벡터(siglip2/irra/solider) × 알고리즘(leiden/dbscan_v6) 격자, 9단계 탐색값 적용, 상위 2개 holdout 검증
.\.venv\Scripts\python.exe bench\combos.py cluster --params-from bench\studies\cluster_leiden_tune_<시각>\best.json --validate --top 2
# 원장의 combo 행으로 리더보드
.\.venv\Scripts\python.exe bench\combos.py report --stage search
```

- `--adopt` 를 켜면 추천이 프로젝트 루트의 `pipeline_best.yaml`(검색: 원본 pipeline.yaml 사본에서 retrievers.*.weight 만 변경, 주석 보존, 로더 검증)
  / `clusterer_best.yaml`(클러스터: clusterer 블록) 로 복사되어 GUI 드롭다운(임베더 구성 / 4b 플러그인 --method-config)에 자동으로 나타난다.
  원본 `pipeline.yaml` 은 절대 바꾸지 않는다.
- 산출물 `bench/combos/<단계_이름_시각>/`: `grid.jsonl`, `leaderboard.md`(순위·feasible·holdout·미세조정 값·운영 표시), `best.json`,
  `recommended.yaml`, `pipeline_best.yaml`, 미세조정 스터디 폴더 `refine_<순위>_<라벨>/`. 원장에 `combo:<라벨>` 행(extra.combo 에 순위·holdout)과 `combo_best` 행.
- 미세조정은 `optimize.run_study` 재사용(조합 축은 고정, 가중치·rrf_k·후보 수만 탐색).

## 6c. GUI 벤치마크 탭 (P5)

`gui/bench_page.py` — 원장 리더보드. 채택 기준은 `bench/criteria.py:ADOPTION_RULES` (기준표 §1~§5 운영값: 검출 AP@0.5 ≥ 0.866·최대 재현율 ≥ 0.90·75–119 px ≥ 0.660·120–199 px ≥ 0.844·fps ≥ 10 / 임베딩 mAP ≥ 89.06 / 검색 mAP ≥ 88.07·pool_recall ≥ 90 / 클러스터 쌍 정밀도 ≥ 0.90·B³F1 ≥ 0.851·혼합 ≤ 138 / e2e mAP ≥ 58.3·검출 상한 ≥ 0.90). 판정 pass/partial/fail/n/a 가 행 색(초록/노랑/빨강/없음). `criteria.adopt_yaml(entry)` 가 행을 실행 가능한 yaml 로 바꾼다(원본 pipeline.yaml / pipeline_tracking.yaml 불변). 기준값을 바꾸려면 ADOPTION_RULES 만 고친다.

## 6d. 사람 정답 평가 3종 (P6)

원장 stage `track` / `object` / `qwen`. 시트(HTML) → 사람 라벨 → `labels.json` → eval. 라벨이 없으면 pseudo 모드(파이프라인 출력 = 정답 가정)로 배관만 확인하고 원장에는 쓰지 않는다. 자세한 절차는 `eval/gt/README.md`.

| stage | 정답 단위 | 예측 | 지표 (METRIC_KEYS) | 채택 기준 (criteria) |
|---|---|---|---|---|
| track | 구간(추적기 id 연속 구간)별 gt_id / ignore | raw(추적기 id) · before(구간) · after(긴 트랙) | idf1 hota deta assa mota idsw fragments splits over_merges + `_before`/`_raw`, idsw_ratio | 과병합 0, IDSW after/before ≤ 0.5 (IDF1/HOTA 절대값은 첫 라벨 측정 뒤) |
| object | 객체 트랙 쌍 같음/다름 → identity 그룹 | 트랙 중심 벡터(`--vector`) 코사인 | map rank1 map_labeled pair_auc pair_f1 pair_threshold pair_acc_at_threshold cluster_pair_precision/recall | 쌍 AUC ≥ 0.90(잠정), 클러스터 쌍 정밀도 ≥ 0.90 |
| qwen | (쿼리, 후보) 맞다/아니다/모름 | qwen_stage 별도 프로세스 (캐시, `--rescore`) | p5/p10/p20 before·after, p10_gain_pp, false_drop_rate, unknown_ratio, sec_per_candidate | P@10 ≥ +10 %p, 오탈락률 ≤ 10 %, ≤ 30 s/후보 |

러너: `bench/run.py track [--tracking-config yaml] [--videos …]` — yaml 을 주면 GT 영상만 `video/batch_preprocess_videos_parallel.py` 로 다시 추적·스티칭한 뒤 같은 정답으로 평가한다(검출기·추적기·스티처 교체 비교). `bench/run.py object --vector siglip2`, `bench/run.py qwen --verify-mode filter --max-queries 5`. verify 는 다른 단계와 같다 (Qwen 은 생성이 비결정적이라 허용 오차가 느슨함).

## 7. 외부 검토(Astra) 반영 (2026-09-28)

verify 는 3상태(PASS / FAIL / UNVERIFIED — 비교 지표가 없거나 원본 지표가 재실행에 없으면 통과 아님)이고 비교한 행에만 붙는다.
e2e 는 `--rerank none` 이 실제로 재정렬을 끄며, 같은 GT 의 중복 검출은 GT 기준 지표에서 한 번만 TP(그 뒤는 비관련), `map_db` 는 DB 양성이
있는 쿼리만의 평균, 매칭 캐시는 iou/sources/collection 이 다르면 다시 만든다. 러너는 숫자 0 을 버리지 않고, `--param a=1 b=2` 처럼 여러 값을
받으며, 실행 폴더에 고유 접미사를 붙이고 원장 append 는 파일 잠금 아래 직렬화된다. 원장 fingerprint 는 fps/sec 같은 시간 계열을 뺀다.
`check.py --db` 는 매니페스트에 없는 retriever(missing)도 FAIL 로 보고, 정적 계약 검사는 base 의 메서드를 재정의하지 않은 클래스를 잡는다.
단독 임베딩 평가(`eval/prw_eval.py`)는 CLI 로 checkpoint 를 덮어쓰지 않으면 yaml 의 module/class 그대로(registry) 만든다 — 원장 `params.loader`.
분할 파일은 로드할 때 겹침·digest 를 검사한다. 남은 한계: 클러스터 벡터 캐시는 DB 빌드 id 를 추적하지 않는다(파일명 지문 = yaml 임베더),
`gallery=test` 는 "전체 DB 에서 검색한 뒤 test 프레임만 채점" 하는 혼합 프로토콜(운영 동작에 가깝지만 사전 제한 gallery 와는 다름), 클러스터 분할
평가는 전체를 군집화한 뒤 일부만 채점하는 전이적 평가다.

## 8. 주의

- 원장은 append-only. 지우거나 고치지 말고 다시 실행해 새 줄을 남긴다 (재현성 검증의 증거).
- 같은 설정을 다시 돌리면 새 줄이 또 생긴다 (`table --latest` 가 이름별 마지막 줄만). `import` 만 fingerprint 로 중복을 거른다.
- 테스트는 `--no-ledger` 또는 임시 `--ledger` 를 쓴다 (`tests/test_bench_*.py`, `tests/test_prw_e2e_search_eval.py`).
- e2e 와 cluster 는 Qdrant 가 떠 있어야 한다 (`curl localhost:6333/collections`). Docker Desktop 이 죽어 있으면 README 2.5.
