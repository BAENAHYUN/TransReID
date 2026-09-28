# TransReID 정량 평가 기준표 (2026-09-26)

검출 → 추적·스티칭 → 임베딩 → 클러스터링 → 검색 → 후처리 각 단계에서 **무엇을, 어떤 정답으로, 어떤 도구로** 재고,
모델·알고리즘을 **바꿔도 되는지**를 어떤 숫자로 판정하는지 정리한 표. 현재값은 전부 이 프로젝트에서 실제로 측정한 값.
HTML 판(공유용): `outputs/audit/eval_criteria_20260926.html`.

## 0. 단계별 측정 현황

| 단계 | 교체 가능 (GUI) | 정답(GT) | 측정 도구 | 상태 |
|---|---|---|---|---|
| 1 검출 | RF-DETR / YOLO26 (드롭다운) | PRW annotations 사람 박스 (test 6,112 프레임) | `eval/detect_eval_prw.py` | 측정됨 (6,112 프레임 전체, 09-26) — YOLO26m 미채택 |
| 2 추적·스티칭 | BoT-SORT / Deep-OC-SORT · SUSHI (yaml) | 준정답 시트 3편 (P6, 라벨링 대기) | `eval/track_gt_eval.py` (IDF1/HOTA/IDSW) | 도구 준비 — 라벨 후 측정 |
| 3 임베딩 (사람) | SigLIP2 / IRRA / SOLIDER | PRW query_box 2,057 / gallery GT crop 19,127 | `eval/prw_eval.py` | 측정됨 |
| 3 임베딩 (객체) | SigLIP2 / DINOv2 | 객체 트랙 쌍 75 (P6, 라벨링 대기) | `eval/object_pair_eval.py` (mAP · 쌍 AUC) | 도구 준비 — 라벨 후 측정 |
| 4 통합 검색 (이미지) | 1차 조합 + 2차 재정렬 (검색 탭 드롭다운) | PRW GT | `eval/prw_eval_unified.py` (단독), `eval/prw_e2e_search_eval.py` (전체 파이프라인) | 측정됨 (단독 + 전체 §4b) |
| 4 통합 검색 (영상) | 벡터 선택 | 없음 (30 쿼리 다양성 지표만) | `eval/run_clustering_dup_eval.ps1` | 다양성만 |
| 5 클러스터링 (사람) | Leiden / DBSCAN v6 / 커스텀 (4b) | PRW GT 인물 ID (30,919점 / 933명) | `eval/prw_cluster_gt_eval.py`, `eval/cluster_ablation_prw.py` | 측정됨 (재현성 포함) |
| 5 클러스터링 (객체) | Leiden (DINOv2) | 같은 쌍 정답 (P6) | `eval/object_pair_eval.py` (클러스터 쌍 정밀도/재현율) | 도구 준비 — 라벨 후 측정 |
| 6 Qwen 후처리 | Qwen3-VL 2B / 4B | 30 쿼리 × 20 후보 시트 (P6, 라벨링 대기) | `eval/qwen_verify_eval.py` (P@K 전·후, 오탈락률) | 도구 준비 — 라벨 후 측정 |

## R. 판정 원칙 (공통)

1. 같은 데이터·같은 정답·같은 스크립트로 잰 값끼리만 비교한다.
2. 정밀도와 재현율은 항상 같이 적는다 (오병합 ↔ 과분할 맞바꿈). 노이즈 비율 단독 보고 금지.
3. 개선 폭 > 재현성 잡음. Leiden seed 42/1/777 쌍 F1 변동 ≤ 0.002 → 클러스터링은 Δ ≥ 0.01 부터 유의미. 검출은 test 전체(6,112 프레임)로 판정.
4. 운영 동작점 명시 (검출 0.5 / 영상 0.2, 클러스터 0.97, 후보 200). 설정 변경 ≠ 모델 교체.
5. 속도는 제약 (검출 ≥ 10 fps@1080p, 검색 1건 ≤ 3 s 로드 제외). 정확도 하락을 속도로 만회하지 않는다.
6. 결과 파일에 조합을 남긴다: 검색 `pipeline`/`models`, 검출 checkpoint `detector`, 클러스터 report `plugin`.

## 1. 검출 (PRW test 6,112 프레임 전체, GT 25,062 박스, IoU ≥ 0.5) — 2026-09-26 재측정

| 지표 | YOLO26m (640) | RF-DETR Medium | 운영 DB (RF-DETR, thr 0.5, h≥75) | 채택 기준 (신규 검출기) | YOLO26m 판정 |
|---|---|---|---|---|---|
| AP@0.5 | 0.876 | 0.876 | 0.810 (곡선 잘림) | 기존 − 0.01 이상 | ✓ 동률 (−0.0001) |
| AP@[.5:.95] | 0.537 | 0.537 | 0.511 | 기존 이상 | ✓ 동률 (−0.0003, 잡음 이내) |
| 최대 재현율 | 0.947 | **0.954** | 0.841 | ≥ 0.90 | ✓ (기존 −0.007) |
| P / R / F1 @0.5 (공통 동작점) | **0.851** / 0.818 / **0.834** | 0.785 / **0.865** / 0.823 | 0.874 / 0.841 / 0.857 | 같은 임계값에서 F1 기존 이상 | ✓ (+0.011) |
| FP 중복 / 배경 @0.5 | 33 / 3,557 | 48 / 5,884 | 42 / 3,003 | 중복 FP 비율 ≤ 5 % | ✓ (0.1 %) |
| FN (놓친 GT 박스) @0.5 | 4,551 | **3,385** | 3,975 | — | ✗ +1,166 |
| 재현율 75–119 px @0.5 | 0.559 | **0.660** | 0.581 | 기존 이상 (작은 사람 손실 금지) | ✗ (−0.101) |
| 재현율 120–199 / 200+ px @0.5 | 0.815 / 0.974 | **0.864** / **0.983** | 0.863 / 0.983 | 기존 − 0.02 이상 | ✗ (−0.049 / −0.009) |
| 속도 (RTX 5090 fp16, 6,112 프레임) | **25.6 ms / 39.1 fps** | 64.1 ms / 15.6 fps | — | ≥ 10 fps | ✓ 2.5× |

**판정: YOLO26m 미채택 — 운영 검출기 RF-DETR Medium 유지.** 300 프레임에서 보였던 AP 우위(0.871 vs 0.859)는 전체 프레임에서 사라져 동률이다.
같은 임계값 0.5 에서 YOLO26m 은 정밀도가 높지만(0.851 vs 0.785) 작은 사람(75–119 px, DB 에 들어가는 crop 최소 높이 구간) 재현율이 0.10, 중형(120–199 px)이 0.05 낮아 놓친 GT 박스가 1,166개 더 많다.
속도는 2.5배 빨라 영상 경로처럼 처리량이 우선인 곳의 대안이다. 임계값을 검출기별로 맞추면 동급이 된다(YOLO 0.4: P 0.800 / R 0.859 ≈ RF-DETR 0.5: P 0.785 / R 0.865; RF-DETR 0.6: P 0.849 / R 0.824 ≈ YOLO 0.5) → 검출기별 운영 임계값은 P3 탐색 대상.
객체 클래스는 GT 가 없어 사람만 평가. 임계값별 곡선 `eval/results/detect_prw/detect_eval_prw.html`, 0.5 공통 동작점 `eval/results/detect_prw/op050/`, 300 프레임 값 `eval/results/detect_prw/t300_report/`, 모든 값은 원장 `bench/ledger.jsonl` (`python bench/ledger.py table --stage detect --latest`).

**검출기별 운영 임계값 (P3 스윕, 2026-09-28):** recall ≥ 0.85 제약 아래 F1 최대 — YOLO26m 0.42 (P 0.812 / R 0.852 / F1 0.831, 배경 FP 4,868) ≈ RF-DETR-M 0.54 (P 0.811 / R 0.851 / F1 0.831, 배경 FP 4,930). 같은 재현율 제약에서는 두 검출기가 동급이고, 운영값 0.2 는 둘 다 배경 FP 가 과다(F1 0.60·0.76). `bench/studies/detect_*/report.md`, 추천 yaml 은 `recommended.yaml`.

## 2. 추적 · 스티칭 (도구 준비, 라벨링 대기)

**P6 (09-28) 도구 준비**: `eval/track_gt_eval.py sheet` 가 영상별 준정답 시트(`eval/gt/tracks/<영상>/sheet.html`, 썸네일 내장)를 만들고 `eval` 이 labels.json 으로 raw(추적기 id)/before(구간)/after(긴 트랙)의 IDF1·HOTA·MOTA·IDSW·단절·갈라짐·과병합을 낸다 (원장 stage=track). 시트 3편(048·289·100: 구간 116, 긴 트랙 77 중 person 54) 라벨링 대기. **실측 주의**: 추적기 id 는 시간이 지나면 재사용되고(한 id 가 5 명), SUSHI 는 512 프레임 창을 독립 처리해 같은 사람이 창마다 다른 긴 트랙 id 를 받는다 (048 영상의 정지 박스 하나가 L1·L8·L18·L28) — 라벨이 붙으면 after 의 `splits`/IDSW 로 정량화된다. pseudo(제안값 = 정답 가정) 실행(09-28 초판 지표): raw IDF1 0.585 / before 0.931 / after 0.998, IDSW 96 / 105 / 0. **외부 검토(Astra, 09-28 16:09) 반영**: HOTA·CLEAR·ignore 를 TrackEval 정의로 재구현(프레임당 매칭 1회 후 α 필터, 직전 프레임 우선 매칭, MOTChallenge ignore), before 를 스티처와 무관한 추적기 tracklet 로 정의, 라벨 검토 체크·검토율·manifest 도입(제안값 승격 방지, 시트 재생성 시 옛 라벨 거부), 채택 판정에 `incomplete`(지표 누락·기준값 미정 = 확인 불가) 추가, Qwen 은 불변 후보 id 로 라벨 연결·filter 를 평가기에서 재현·캐시 계약(`<qid>.meta.json`)·오탈락률 두 정의, 객체는 캐시 정합성 검사·없는 트랙 쌍 제외·모순 보고·point 목록 보존. SUSHI 창 경계 연결 후처리(`pipeline_tracking_sushi_link.yaml`, raw track 연속 + IoU + 일대일)를 추가해 러너로 비교 가능. 창 경계 연결(restitch, 같은 검출·추적 출력, 3편 pseudo GT = 추적기 연속성 기준): 원 스티처 after IDF1 0.619 / IDSW 41 / 갈라짐 32 / 과병합 8 → 연결 후 IDF1 0.937 / IDSW 19 / 갈라짐 10 / 과병합 7 (048 긴 트랙 34→16, 289 26→18, 100 16→12; 거절된 후보는 간격 6·15·31 프레임). 사람 라벨이 붙어야 '같은 사람' 여부가 확정된다.

| 지표 | 정의 | 정답 확보 | 채택 기준 |
|---|---|---|---|
| IDF1 | ID 일관성 F1 | ① 보유 영상 3~5편 트랙 30~50개 수동 준정답 ② MOT17/20 공개셋 | 기존 − 1 %p 이상 |
| HOTA | 검출·연관 정확도 기하평균 | 동일 | 기존 이상 |
| ID switch | ID 변경 횟수 (스티칭 전/후) | 동일 | 스티칭 후 ≤ 전의 50 % |
| 트랙 단절 / 과병합 | 갈라짐 수 / 섞임 수 | 동일 | 과병합 0 우선 |

## 3. 임베딩 (PRW Re-ID: query 2,057 / gallery 19,127)

| 모델 | 차원 | 텍스트 | mAP (%) | Rank-1 | Rank-5 | Rank-10 | 역할 |
|---|---|---|---|---|---|---|---|
| SOLIDER (Swin-B, Market) | 1024 | 불가 | **89.06** | **97.08** | **98.69** | **98.98** | identity 기준 (재정렬·클러스터·영상) |
| IRRA (ViT-B/16) | 512 | 가능 | 58.59 | 91.54 | 96.50 | 97.42 | 자연어 → 사람 1차 |
| SigLIP2 (base) | 768 | 가능 | 12.38 | 58.77 | 73.12 | 78.56 | 속성 검색 보조 |
| DINOv2 (g14 reg) | 1536 | 불가 | — | — | — | — | 객체, GT 없음 |

채택: 사람 이미지용 mAP ≥ 89.1 → SOLIDER 대체 후보, ≥ 80 → 조합 후보. 텍스트용 supports_text + mAP ≥ 58.6. 차원이 다르면 새 `collection_prefix` 로 적재 후 비교. 객체 임베더는 identity 세트(재출현 50쌍+) 라벨링 선행.

**P6 (09-28) 도구 준비**: `eval/object_pair_eval.py sheet` 가 운영 DB 객체 트랙 2,233개에서 쌍 75개(클러스터 안 40 · 최근접-다른 클러스터 25 · 무작위 10)를 제안한 시트 `eval/gt/object_pairs/sheet.html` 을 만들고, `eval --vector dinov2|siglip2` 가 같음 쌍으로 identity 그룹을 만들어 검색 mAP/R1·쌍 AUC·최적 임계값·운영 클러스터의 쌍 정밀도/재현율을 낸다 (원장 stage=object). 라벨링 대기. pseudo(클러스터 쌍 = 같음 가정) 실행: mAP 81.0 (라벨 갤러리 90.4), 쌍 AUC 0.74, 최적 임계 0.739 — 운영 0.97 과 차이가 커서 라벨 후 재확인 필요.

## 4. 통합 검색 (PRW GT, GUI 경로 재현)

| 조합 | mAP | Rank-1 | Rank-10 | pool recall % | 정답 전부 후보 안 % | 비고 |
|---|---|---|---|---|---|---|
| SigLIP2+IRRA → SOLIDER (운영 기본) | 71.20 | 96.79 | 98.88 | 73.3 | 12.3 | 후보에서 정답 27 % 손실 |
| SOLIDER 단독 | **89.06** | **97.08** | 98.98 | 100 | 100 | 상한 |
| SigLIP2+IRRA+SOLIDER → SOLIDER | 88.07 | 97.08 | 98.98 | 93.5 | 54.8 | 후보 200 안에서 최선 |
| IRRA → SOLIDER | 74.82 | 96.84 | 98.88 | 77.4 | 16.2 | SigLIP2 제외 시 상승 |
| SigLIP2 → SOLIDER | 25.52 | 92.81 | 94.65 | 26.0 | 0.2 | 부적합 |
| SigLIP2+IRRA, 재정렬 없음 | 46.07 | 91.01 | 97.13 | 73.3 | 12.3 | 재정렬 효과 +25 |
| 운영 조합, 후보 1000 | 82.52 | 96.99 | 99.03 | 87.6 | 33.5 | 후보 수 효과 +11 |

채택: mAP 상승 AND pool recall ≥ 90 %. 영상 검색은 GT 없음 — Top-10 서로 다른 그룹 수 (30 쿼리) 기본 2.5 → 클러스터 묶기 7.7 (다양성 지표, 정답성 아님).

**조합 탐색 (P4, 2026-09-28) — 검색:** 임베더 부분집합 × 재정렬 × 후보 수 50조합을 tune 분할로 전부 채점(feasible 30, pool_recall ≥ 90). 운영 조합 `siglip2+irra→solider@200` 은 34위 (mAP 67.46, holdout 68.50); 1위 `solider→none@1000` mAP 89.36 (holdout 88.91); 상위 3개 미세조정 후 최종 추천 `solider→none@1000` (grid) mAP 89.36, holdout 88.91. 추천 가중치로 만든 `pipeline_best.yaml`(가중치만 변경, 로더 검증)은 `search_grid_20260928T120139/`. `leaderboard.md` 참고.

### 4b. 전체 파이프라인 검색 (운영 DB 의 검출 crop, e2e, 2026-09-28)

> 프로토콜 주의(외부 검토 지적): gallery=test 는 운영 DB 안의 PRW test 프레임 point 만 남기지만 쿼리는 query_box 이고 DB 는 검출 crop 이라 §4 단독 평가(GT crop)와 정확히 같은 갤러리가 아니다. 두 표는 같은 순위 정의를 쓰되 절대값을 직접 비교하지 않는다. 등록기 표본 갤러리(`__sampleN`)는 GT 양성의 일부만 담아 `map_db`·R1 로만 비교한다.

단독 평가(위 표, GT crop 입력)와 달리 실제 파이프라인이 만든 운영 DB(RF-DETR 검출 → crop 필터 → 임베딩 → Qdrant 43,343 point)를
GUI 와 같은 검색 경로로 PRW query_box 로 검색하고 point→pid IoU≥0.5 매칭으로 채점 (`eval/prw_e2e_search_eval.py`, 순위 길이 K=200, gallery = test 프레임).
mAP 는 GT 박스 수가 분모(검출 손실 포함), mAP(db) 는 DB 에 들어간 point 수가 분모(검색만). 검출 상한 = 검출이 DB 에 넣은 GT 비율.

| 조합 | mAP | mAP(db) | Rank-1 | Rank-10 | recall@K | 검출 상한 | distractor | s/쿼리 | 쿼리 | run_id |
|---|---|---|---|---|---|---|---|---|---|---|
| SigLIP2+IRRA → SOLIDER (운영 기본) | 58.32 | 63.26 | 92.00 | 98.00 | 61.62 | 0.908 | 0.39 | 0.59 | 500 | `e2e_20260928T114923_df2008fd` |
| SOLIDER 단독 (재정렬 없음) | 54.28 | 59.04 | 89.20 | 96.40 | 59.89 | 0.908 | 0.56 | 0.19 | 500 | `e2e_20260928T115123_6bdde575` |
| 운영 기본, 50쿼리 스모크 | 58.51 | 61.33 | 94.00 | 96.00 | 62.68 | 0.926 | 0.37 | 1.86 | 50 | `e2e_20260928T111341_f5b3b1b6` |

읽는 법: 단독 88.1(3-RRF+SOLIDER)·89.1(SOLIDER) 과 전체 파이프라인의 차이는 (1) 검출·crop 필터가 놓친 사람(검출 상한 < 1), (2) 후보 200 안에 train 프레임·다른 source 의 distractor 가 섞여(distractor 비율) 정답이 밀려나는 것, (3) 후보 단계 pool recall. 채택 판정은 이 표(전체)와 위 표(단독)를 함께 본다 — 단독만 오르고 전체가 안 오르면 후보·검출 쪽 문제. 500 쿼리는 2,057 개에서 고르게 뽑은 부분집합(1.9 s/쿼리라 전체는 약 1시간; `bench/run.py e2e` 로 언제든 전체 실행). 원장 stage=e2e.


## 5. 클러스터링 (PRW GT 30,919점 / 933명, SOLIDER, 정확 kNN k=30 상호)

**Leiden 탐색 (P3, 2026-09-28):** tune 분할 30 trial (43k 점, trial당 ~2분) — pair_precision ≥ 0.90 제약 아래 B³F1 운영값 0.884 → 추천 0.936 (knn 33 · threshold 0.950 · resolution 1.47 · mutual off · 최대 500). holdout 에서는 0.876 → 0.898 (+0.022, 잡음 0.002 초과, 부호 일치)이지만 **쌍 정밀도가 0.959 → 0.840 으로 떨어져 제약이 깨지고 혼합 클러스터 28 → 69** → 그대로 채택하지 않음(정밀 우선 원칙). 다음: 제약을 0.93 이상으로 올리거나 정밀도를 목적에 넣어 재탐색. `bench/studies/cluster_leiden_tune_*/report.md`.

**조합 탐색 (P4, 2026-09-28) — 클러스터:** 벡터 × 알고리즘 4조합(Leiden 탐색값 적용): leiden@solider 0.936, dbscan_v6@solider 0.784, leiden@irra 0.087, leiden@siglip2 0.185(NG). 운영 `leiden@solider` 가 1위(holdout 0.898); IRRA·SigLIP2 벡터는 Leiden 임계값(SOLIDER cosine 스케일)이 맞지 않아 벡터별 미세조정(`--refine-trials`) 없이는 불리 — 격자 비교는 미세조정과 함께 읽어야 한다. `cluster_grid_20260928T120142/leaderboard.md`.

| 방법 | 쌍 P (오병합↓) | 쌍 R (과분할↓) | 쌍 F1 | B³ F1 | purity | 혼합 클러스터 | 갈라진 인물 | 노이즈 % |
|---|---|---|---|---|---|---|---|---|
| Leiden 0.97 (운영) | **0.921** | 0.748 | **0.825** | 0.851 | 0.949 | **138** | 413 | 7.1 |
| Leiden 0.96 | 0.823 | 0.799 | 0.811 | **0.861** | 0.888 | 177 | 345 | 3.6 |
| Leiden 0.95 | 0.671 | **0.815** | 0.736 | 0.821 | 0.793 | 197 | **294** | 1.9 |
| DBSCAN v6 (eps 0.12, 결합) | 0.926 | 0.361 | 0.519 | 0.754 | 0.951 | 356 | 615 | 0.4 |
| Leiden 0.95 HNSW (이전 운영) | 0.648 | 0.494 | 0.561 | 0.649 | 0.806 | 184 | 247 | 27.6 |

- 재현성: seed 42/1/777 쌍 F1 0.8254/0.8235/0.8254 (Δ ≤ 0.002).
- 알고리즘 효과: 같은 벡터에서 Leiden > DBSCAN (B³ F1 0.862 vs 0.765; DBSCAN 쌍 R ≤ 0.37 한계).
- 벡터 효과: SOLIDER 단독 > 결합 (결합 cosine 의 78 % 가 SOLIDER).
- 이웃 효과: 정확 kNN 필수 (HNSW+필터는 이웃 70 % → 노이즈 30 %).

채택 (4b 플러그인): 같은 벡터·점 집합에서 B³ F1 ≥ 0.85 AND 혼합 클러스터 ≤ 138. 쌍 P ≥ 0.90 유지한 채 R 상승이면 채택. 노이즈 비율은 참고치. 객체 클러스터링은 GT 없음.

## 6. Qwen 후처리 (정확도: 도구 준비, 라벨링 대기; 속도 26 s/후보 텍스트, 35 s/후보 crop)

**P6 (09-28) 도구 준비**: `eval/qwen_verify_eval.py sheet` 가 `eval/gt/qwen_queries.json` 의 자연어 30 쿼리를 GUI 경로로 검색해 상위 20 후보(600)를 GUI 가 Qwen 에 넘기는 형식 그대로 저장하고 판정 시트 `eval/gt/qwen/sheet.html` 을 만든다. `eval` 은 후보마다 `verifiers/qwen_stage.py` 를 별도 프로세스로 돌려(캐시) P@5/10/20 전·후, 오탈락률, UNKNOWN 비율, 후보당 초를 낸다 (원장 stage=qwen). 라벨링 대기. 600 후보 ≈ 4~5 시간(2B) → `--max-queries` 분할 + `--rescore`.

| 지표 | 정의 | 정답 확보 | 채택 기준 |
|---|---|---|---|
| Precision@K 변화 | 검증 전/후 상위 K 정답 비율 | 자연어 30 쿼리 × 상위 20 후보 사람 판정 (600) | P@10 ≥ +10 %p |
| 오탈락률 | FAIL 판정 중 실제 정답 비율 | 동일 | ≤ 10 % |
| UNKNOWN 비율 | 판정 불능 비율 | — | 보고 |
| 처리 시간/후보 | — | — | 2B ≤ 30 s, 4B ≤ 60 s |

## 7. 속도 · 규모 (RTX 5090)

| 단계 | 측정값 | 조건 |
|---|---|---|
| 검출 | YOLO26m 48.8 fps · RF-DETR-M 12.5 fps | 1080p fp16, 검출만 |
| 임베딩 4종 로드 | ~50 s | 첫 검색 지연 원인 |
| 사진 1장 → crop → 검색 | 14 crop 137 s (로드 포함), crop 당 ~1.7 s | `search/image_search.py` |
| 클러스터링 43k | 수집 30 s + kNN 12 s + Leiden 7 s | SOLIDER, gRPC, GPU |
| 클러스터링 3k (driver) | Leiden 11 s · DBSCAN v6 3.5 s (+결합 수집 14 s) | 캐시 재사용 |
| Qwen 검증 | 26 s / 35 s per 후보 | Qwen3-VL-2B |

## H. 측정 명령

| 단계 | 명령 | 산출물 |
|---|---|---|
| 검출 | `python eval/detect_eval_prw.py --mode run --name yolo26m_test --detector-config pipeline_tracking_yolo26.yaml --split test --no-score` (RF-DETR 는 `pipeline_tracking.yaml`, 운영 DB 는 `--mode import-qdrant --name prod_db_test`) → `--mode score --method yolo26m_test=… rfdetr_medium_test=… prod_db_test=… [--operating-threshold 0.5]` (GUI 평가 3단계) | `eval/results/detect_prw/detect_eval_prw.html`, `summary.csv`, `op050/` |
| 러너·verify | `python bench/run.py detect|embed|search|cluster|e2e …` → `bench/runs/<…>/` + 원장; `python bench/run.py verify <run_id>` 재현 판정; `python bench/check.py --instantiate --db` 계약·DB 지문 (GUI 평가 5~8단계) | `bench/runs/`, `bench/ledger.jsonl` |
| 러너·verify | `python bench/run.py detect|embed|search|cluster|e2e …` → `bench/runs/<…>/` + 원장; `python bench/run.py verify <run_id>` 재현 판정; `python bench/check.py --instantiate --db` 계약·DB 지문 (GUI 평가 5~8단계) | `bench/runs/`, `bench/ledger.jsonl` |
| 원장 | 위 4개 스크립트가 자동 기록 (`--no-ledger` 로 끔). 표: `python bench/ledger.py table --stage detect --latest`, 목록: `list --stage cluster`, 기존 산출물 이관: `import` (GUI 평가 4단계) | `bench/ledger.jsonl` |
| 임베딩 | `python eval/prw_eval.py --model solider --data-root ./data/PRW --save-result …` / `eval/run_embedding_eval.ps1` | `eval/results/embedding_summary.csv` |
| 통합 검색 | `eval/run_unified_eval.ps1` | `eval/results/unified_eval.csv`, `unified_eval_report.md` |
| 클러스터링 | `python clustering/cluster_qdrant.py --target person --sources prw_image --method dbscan_v6 …` → `python eval/prw_cluster_gt_eval.py --method 이름=assignments.jsonl …` | `eval/results/cluster_gt_prw*/summary.csv` |
| 재현성 | seed 3회 → GT 평가 `--method seed42=… --method seed1=…` | `eval/results/repro_check/report/summary.csv` |
| 영상 검색 다양성 | `eval/run_clustering_dup_eval.ps1` → `eval/summarize_clustering_dup.ps1` | `eval/results/clustering_dup_summary.csv` |

출처: `bench/ledger.jsonl` (원장) · `eval/results/` (detect_prw, embedding_summary, unified_eval, cluster_gt_prw_exact, cluster_ablation_prw, repro_check, clustering_dup_summary). 배경: `outputs/audit/eval_framework_design_20260921.md`.
