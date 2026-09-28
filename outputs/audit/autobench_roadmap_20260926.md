# TransReID 자동 벤치마크 로드맵 (2026-09-26)

목표: 어떤 모델이 더 좋은지 정량으로 · 자유롭게 탈부착 · 하이퍼파라미터는 수기 없이 · 남이 돌려도 같은 숫자 · 새 AI 를 넣으면 최적 조합 — 을 만들어 주는 GUI·도구.
HTML 판: `outputs/audit/autobench_roadmap_20260926.html`.

## Q. 다섯 질문의 현재 답

| 질문 | 지금 상태 | 부족한 것 | 단계 |
|---|---|---|---|
| 정량 테스트로 우열 판단 | 가능 — 검출·임베딩·검색·클러스터링 GT 평가 + 기준표 | 결과가 스크립트별 파일에 흩어짐; 추적·객체·Qwen 정답 없음 | P1, P6 |
| 자유 탈부착 | 가능 — 검출기·임베더·클러스터링·검색 조합 yaml/드롭다운 교체 | Qwen 검증기·Qdrant 하드코딩; 새 모델은 어댑터 수기 | P7 |
| 하이퍼파라미터 자동 평가 | 설계만 — 목적함수는 기준표에 정의 | Optuna 탐색기·단일 러너·train/val 분할 없음 | P2, P3 |
| 타인 재현 | 가능 — `bench/run.py verify run_id` 가 같은 설정으로 재실행해 허용 오차 판정(검출·클러스터 PASS, Δ 0); 원장에 git·yaml·가중치 sha·하드웨어 | 다른 PC 실측은 아직 (github_upload 재동기화 뒤) | P1, P2 ✓ |
| 새 AI → 최적 조합 | 없음 — 사람이 표 보고 고름 | 조합 탐색기·리더보드 GUI·채택→yaml | P4, P5, P7 |

## G. 목표 시스템 흐름

구성 요소 yaml(+어댑터) → `bench/run.py`(단일 러너, 고정 GT) → `bench/ledger.jsonl`(원장: 지표·설정·해시·버전·seed) → `bench/optimize.py`(Optuna + 조합 탐색, Pareto) → GUI 벤치마크 탭(리더보드·그래프·채택) → `pipeline_<이름>.yaml` → 기존 드롭다운에 자동 등장. `bench/run.py --verify` 로 재현성 검증.

이미 있음: yaml 교체, GT 평가 스크립트 4종, PRW GT 캐시, 드롭다운 자동 반영. 없음: 원장·러너·탐색기·리더보드·verify·등록기.

## M. 의존 순서

P0 → P1 → P2 → {P3 → P4 → P5b, P5a → P5b, P7}. P6(정답 라벨링, 사람)은 처음부터 병렬.

## P. 단계

| 단계 | 내용 | 산출물 | 완료 조건 | 소요 |
|---|---|---|---|---|
| **P0 완료 (09-26)** | 검출 평가를 test 6,112 프레임 전체로 재측정 (YOLO26m·RF-DETR-M·운영 DB, 운영 임계값 0.5 동작점) | `eval/results/detect_prw/` 갱신, `op050/` | ✓ 기준표 §1 갱신 — AP@0.5 동률 0.876, YOLO26m **미채택**(75–119 px 재현율 −0.10, 120–199 px −0.05), 속도 2.5× | 0.5일 |
| **P1 완료 (09-26)** | 실행 원장: 모든 평가를 append-only JSONL 한 곳에 (run_id, fingerprint, stage, component, params, gt, metrics, timing, versions, env(git sha), inputs sha, yaml sha, 명령) | `bench/ledger.py` + `bench/ledger.jsonl` (39행 = 기존 33 이관 + P0 6), 평가 스크립트 4개 자동 기록(`--no-ledger` 로 끔), GUI 평가 4단계, `bench/README.md`, 테스트 19개 | ✓ `python bench/ledger.py table --stage detect --latest` 가 기준표 §1 을, `--stage embed/search/cluster` 가 §3·§4·§5 를 재생성 | 1일 |
| **P2 완료 (09-28)** | 단일 러너 + 재현성 검증: `bench/run.py detect|embed|search|cluster|e2e …`, `verify run_id`(허용 오차 F1·AP 0.005, mAP 0.1%p, Rank 0.2%p), `show-cmd` | `bench/run.py`, `bench/check.py`(계약·DB 지문), `bench/splits.py`(tune/holdout), `eval/prw_e2e_search_eval.py`(전체 파이프라인), GUI 평가 5~8단계, `bench/README.md`, 테스트 46개(전체 287) | ✓ 5단계 한 줄 실행 실측(검출 300f · Leiden 3,000점 · e2e 50/500쿼리) · verify PASS 2건(Δ 0) · 계약 검사 62 OK · DB 지문 same | 1.5일 |
| **P3 완료 (09-28)** | `bench/optimize.py`: Optuna(TPE) 로 클러스터(pair_precision≥0.90 아래 b3_f1)·검색(pool_recall≥90 아래 mAP), 검출은 임계값 결정적 스윕(recall≥0.85 아래 f1). tune(135명)으로 찾고 holdout(315명)으로 검증, 첫 trial = 운영값, Pareto·recommended.yaml | `bench/optimize.py`, `bench/studies/<단계_이름_시각>/`, GUI 평가 9단계, 테스트 8개 | ✓ 검색 40 trial 122초: 운영 70.6 → 추천 89.4 (holdout 71.5 → 88.9, 3-RRF+SOLIDER prefetch 850/pool 1000) · ✓ 검출 임계값 YOLO26m 0.42 ≈ RF-DETR 0.54 (F1 0.831 동급) · ✓ Leiden 30 trial: B³F1 0.884 → 0.936 (holdout 0.876 → 0.898) 이지만 holdout 쌍 정밀도 0.84 < 0.90 → 미채택, 제약 강화 재탐색 필요 | 2일 |
| **P4 완료 (09-28)** | 조합 탐색: 임베더 부분집합×재정렬×후보 수(50) / 벡터×클러스터러(4) 격자를 tune 으로 채점 → 리더보드 → 상위 Optuna 미세조정 → holdout | `bench/combos.py`, `bench/combos/<…>/leaderboard.md`, `pipeline_best.yaml`(가중치만 변경, 로더 검증, `--adopt` 로 루트 복사), GUI 평가 10단계, 테스트 8개 | ✓ 운영 검색 조합 34위 → 추천 `solider→none@1000` mAP 89.36 (holdout 88.91) · 클러스터는 운영 leiden@solider 1위 | 1.5일 |
| **P5a 완료 (09-28)** | GUI '벤치마크' 탭: 원장 리더보드(단계·이름·최근만·통과만 필터, 숫자 정렬), 채택 기준(`bench/criteria.py` = 기준표 운영값) 통과 색, 상세 JSON, verify 실행, 재현 명령 | `gui/bench_page.py`, `bench/criteria.py`, 테스트 12개 | ✓ GUI 에서 원장 행 비교·verify — 새 실행은 평가 탭 5·9·10 | 1일 |
| **P5b 완료 (09-28)** | 그래프 탭(단계별 목적 vs 제약 산점도 + 채택 기준선, 검출 행 PR 곡선), "채택 → yaml": 검출기 → `pipeline_tracking_<이름>.yaml`(tracker/stitcher 는 원본에서), 클러스터 플러그인 → `clusterer_<이름>.yaml`, 검색 조합 → `pipeline_<이름>.yaml`(가중치만) + `.search.json` | `bench/criteria.py:adopt_yaml` | ✓ 만든 yaml 이 드롭다운(pipeline*.yaml glob)에 등장; 검출기 드롭다운은 class 로 중복 제거하므로 같은 class 변형은 경로 지정 | 1일 |
| **P6 도구 완료 (09-28) · 라벨링 대기(사람)** | 정답 시트 3종(HTML, 썸네일 내장, 자동 저장 → labels.json) + 평가 스크립트 3종: 추적 `eval/track_gt_eval.py`(구간 라벨 → raw/before/after IDF1·HOTA·MOTA·IDSW·단절·갈라짐·과병합), 객체 `eval/object_pair_eval.py`(쌍 같음/다름 → identity 그룹 → mAP·쌍 AUC·클러스터 일치, 벡터 교체 비교), Qwen `eval/qwen_verify_eval.py`(qwen_stage 별도 실행·캐시 → P@K 전·후·오탈락률·UNKNOWN·초/후보). 원장 stage track/object/qwen, 러너 `bench/run.py track|object|qwen`(track 은 `--tracking-config` 로 GT 영상 재추적), 벤치마크 탭 3단계, GUI 평가 12~14, 테스트 44(전체 350) | `eval/gt/README.md`(라벨 가이드), 시트: tracks/{048,289,100} 구간 116 · object_pairs 75쌍 · qwen 30×20 | ✓ 도구·시트 준비, pseudo 실행으로 배관 검증 (추적 after IDF1 0.998 / before 0.931 / raw 0.585 · 객체 pseudo mAP 81.0). 실측 발견: SUSHI 512 프레임 창 독립 → 같은 사람이 창마다 다른 긴 트랙 id. Qwen eval 데모는 qwen_stage 프로세스 크래시(0xC0000005) 로 미완 → 라벨 후 재시도 | 2~3일 라벨링 (사람) |
| **P7 완료 (09-28)** | 새 AI 등록기: 어댑터 스켈레톤(`template`) + 명령 1줄 → 계약 검사(정적 + 실제 생성) → yaml 등록(원본 불변) → 단계 벤치(bench/run.py) → 원장 순위·채택 기준 | `bench/register.py`, `bench/REGISTER_GUIDE.md`(새 모델 10분 가이드), GUI 평가 11단계, 테스트 5개(전체 323) | ✓ YOLO26-small 을 코드 수정 없이 등록(기존 YOLO26Detector + weights=yolo26s.pt): 계약 검사 4건 OK → `pipeline_tracking_yolo26s.yaml` → 300 프레임 벤치 AP@0.5 0.8742 / 최대 재현율 0.9359 / F1 0.8265 → 원장 검출 순위 5/10, 채택 기준 일부 통과. 임베더는 단독 Re-ID 평가까지 자동, 운영 DB 적재는 수동(한계) | 1일 |

## T. 일정 (작업일 ≈ 10일)
P0(0–0.5) → P1(0.5–1.5) → P2(1.5–3) → P3(3–5) ‖ P5a(3–4) → P4(5–6.5) → P5b(6.5–7.5) → P7(7.5–8.5) → P6 스크립트(8.5–10). P6 라벨링은 0–3일에 병렬.

## D. 결정 필요
- 목적함수 우선순위: 권고 = 정밀 우선 제약 + B³F1 최대화.
- 탐색 예산: 클러스터 60 · 검색 40 · 검출 20 trial (≈4 GPU시간).
- 정답 라벨링 담당·기한: P0 과 같은 주 시작, 추적 준정답 최우선.
- 벡터 DB 추상화: 하지 않음 (30개 파일 수정 대비 이득 없음).
- 배포: P2 완료 시 github_upload 재동기화 + bench/README 로 verify 절차.

## B. 외부 검토 보완점 4개 → 반영 (2026-09-28)

| 보완점 | 반영 | 확인 |
|---|---|---|
| 1 단계별 공통 입출력 규격 | 계약 = `BaseDetector.detect → Detection` / `BaseEmbedder.embed_crops → (N, DIM)` / `BaseClusterer.cluster → ClusterResult`. `bench/check.py` 가 등록 전에 import·상속·생성자 인자·출력 형태(`--instantiate`)를 검사 | 검사 62건 OK (retriever 4, detector/tracker/stitcher 3 yaml, 클러스터러 2) |
| 2 단독 평가와 전체 평가 분리 | 단독 4단계(고정 GT 입력)에 **전체 파이프라인 `e2e`** 추가: 운영 DB(검출 crop)를 GUI 와 같은 검색 경로로 PRW query_box 로 검색, point→pid IoU 매칭으로 채점. `map`(GT 분모, 검출 손실 포함) vs `map_db`(DB 분모) + `det_ceiling` | 50쿼리 스모크: mAP 58.5 / mAP(db) 61.3 / Rank-1 94 / 검출 상한 0.926 / distractor 0.37 / 1.86 s·쿼리. 단독 88.1 과의 차이 = 후보 200·train 프레임 distractor·검출 손실 → P3 탐색 대상(pool/prefetch). 500쿼리 결과는 기준표 §4b |
| 3 호환성 검사·캐시 관리 | 임베더 지문 = module/class/params + 가중치 파일 내용 sha256 (`ledger.retriever_fingerprint_sha`). 통합 검색 캐시(`prw_gt_<model>.npz`)는 지문으로 검증, 클러스터링 벡터 캐시는 파일명에 지문. `check.py --db` 가 DB point 의 `embedding_build_id` 매니페스트 지문과 yaml 을 대조 | person/object 컬렉션 `emb_20260917_866abdad` = same(siglip2·irra·solider·dinov2); build id 없는 legacy point 경고(표본 2,000 중 1,321 / 475) |
| 4 공정·재현 가능한 기록 | 원장 행에 `hardware`(GPU/VRAM/CUDA/CPU/RAM), `weights`(sha256), `env`(git sha·dirty·host), `bench`(인자·명령·폴더·소요), `gt`·`pid_split`. 인물 분할 `bench/splits/prw_pids_seed42.json` (tune 135명·633쿼리 / holdout 315명·1,424쿼리); `--pid-split` 을 클러스터·e2e 평가와 러너가 받음. Optuna 는 tune, 최종 비교는 holdout | `verify` PASS 2건: 검출 YOLO26m 300f (Δ 0 전 지표), Leiden 3,000점 (Δ 0). verify 행에 same_git_commit / same_host |

## A. Astra(GPT) 서브 에이전트 검토 → 반영 (2026-09-28)

P1·P2 산출물을 `codex-astra` 로 단계별 검토시켰다 (`codex_out/astra/20260928_112138_7172d548e2/result.md`). 높음 지적 8건 중 실제 결함으로 확인된 것을 전부 고쳤다:

| 지적 | 판정 | 반영 |
|---|---|---|
| e2e `--rerank none` 이 재정렬을 못 끔 (검색기가 None 을 운영 기본으로 해석) | 결함 확인 | "none" 문자열로 전달; SOLIDER 단독 500쿼리 재측정 |
| 중복 검출이 GT 기준 AP·recall·검출 상한을 부풀림 | 결함 확인 | 프레임별 GT 수만큼만 TP, 나머지는 비관련; `map_db` 는 DB 양성 있는 쿼리만 |
| 매칭 캐시를 검증 없이 재사용 | 결함 확인 | 헤더 iou/sources/collection 대조, 다르면 재생성 |
| verify 가 공통 지표 0개여도 PASS, 검색 variant 전부에 verify 표시 | 결함 확인 | PASS / FAIL / UNVERIFIED 3상태, 원본 지표 누락 = FAIL, 비교한 행에만 기록 |
| 클러스터 직접 실행 행 복원이 생성자와 불일치 (score_threshold≠threshold) | 결함 확인 | component.params 만 사용, pid_split 보존, 생성자 호출 테스트 |
| 숫자 0 이 `0 == False` 로 사라짐 · GUI list 인자가 `--param a b` 로 전달 | 결함 확인 | `_empty()`, `action=extend nargs=+` (--param/--tol/--compare) |
| 병렬 trial 폴더 충돌·원장 잠금 없음 | 위험 확인 | 폴더 uuid 접미사, 원장 append 파일 잠금 |
| 단독 임베딩 평가가 yaml module/class 를 안 씀 | 결함 확인 | CLI override 없으면 registry(yaml) 로 생성, 원장 `params.loader` |
| check.py missing→OK, base 메서드 미재정의 통과 | 결함 확인 | missing = FAIL, base 의 메서드 그대로면 FAIL |
| 임베딩 캐시 legacy/실패 fallback · split 무결성 · fingerprint 에 fps 포함 | 확인 | 지문 실패 시 재사용 안 함, split 겹침·digest 검사, fingerprint 에서 시간 계열 제외 |

미반영(한계로 문서화): 클러스터 벡터 캐시의 DB 빌드 id 추적, gallery=test 혼합 프로토콜, 클러스터 분할의 전이적 평가. 수정 후 테스트 299 통과, verify 재실행 PASS(검출·클러스터 Δ 0).

## N. 오늘 할 일
1. ~~P0~~ 완료: YOLO26m·RF-DETR-M·운영 DB 6,112 프레임, 0.5 동작점 비교 → 기준표 §1 갱신 (YOLO26m 미채택, 검출기별 임계값은 P3 로).
2. ~~P1~~ 완료: 원장 스키마(schema_version 1), `bench/ledger.py`, 스크립트 4개 자동 기록, 기존 결과 이관, GUI 평가 4단계.
4. ~~P2~~ 완료: `bench/run.py`(5단계 + verify), `bench/check.py`, `bench/splits.py`, e2e 평가, GUI 평가 5~8단계 (B 절).
5. ~~P3 착수~~: `bench/optimize.py` — 검색·검출 스터디 완료(A·P3 행), Leiden 스터디 진행 중. 다음: 스터디 결과를 채택(recommended.yaml → 드롭다운, P5b)하고 P4 조합 탐색·P5a 리더보드.
3. ~~P6 도구~~ 완료: 시트 3종 생성됨 — 사용자 라벨링(eval/gt/README.md) 후 `eval` 실행이 남음.
