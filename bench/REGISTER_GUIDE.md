# 새 모델 10분 가이드 (P7 — `bench/register.py`)

새 검출기 · 클러스터러 · 임베더를 **어댑터 파일 1개 + 명령 1줄**로 등록하고, 계약 검사 → 단계 벤치 → 원장 순위(GUI 벤치마크 탭)까지 자동으로 받는다.
코드 수정이 필요 없는 경우(같은 어댑터에 다른 가중치/모델 id)는 어댑터 없이 명령 1줄이다.

## 0. 코드 수정 없이 되는 것

```powershell
# YOLO26 어댑터로 다른 가중치 (yolo26s.pt 는 자동 다운로드) → 300 프레임 벤치 → 순위
.\.venv\Scripts\python.exe bench\register.py detector --name yolo26s --module detect.detectors.yolo26_detector --class YOLO26Detector --param weights=yolo26s.pt --limit 300
# 같은 SigLIP2 어댑터로 다른 model_id (dim 은 모델에 맞게)
.\.venv\Scripts\python.exe bench\register.py embedder --name siglip2_so400m --module embedders.siglip2_embedder --class SigLIP2Embedder --dim 1152 --scope all --supports-text --param model_id=google/siglip2-so400m-patch16-naflex
# 내장 클러스터러의 다른 설정을 이름 붙여 등록 (3,000 점 벤치)
.\.venv\Scripts\python.exe bench\register.py clusterer --name leiden_t95 --module clustering.methods.leiden --class LeidenClusterer --param threshold=0.95 mutual_knn=false --max-points 3000
```

## 1. 어댑터 만들기 (새 라이브러리/모델일 때)

```powershell
.\.venv\Scripts\python.exe bench\register.py template detector --name mydet        # → detect/detectors/mydet_detector.py
.\.venv\Scripts\python.exe bench\register.py template clusterer --name myclu       # → clustering/methods/myclu.py
.\.venv\Scripts\python.exe bench\register.py template embedder --name myemb --dim 768   # → embedders/myemb_embedder.py
```

스켈레톤의 TODO 만 채우면 된다. 계약(공통 입출력 규격):

| 종류 | 상속 | 구현 | 출력 |
|---|---|---|---|
| 검출기 | `detect.base.BaseDetector` | `__init__(…params…)`, `detect(frame, *, frame_idx, timestamp_sec=None)`; 속성 `conf_threshold` | `List[Detection]` — `bbox=(x1,y1,x2,y2)` 픽셀, `confidence`, `class_id`(COCO 1-based, `coco_id_by_name()`), `class_name` |
| 클러스터러 | `clustering.base.BaseClusterer` | `name`, `required_vectors`, `params()`, `cluster(ids, primary, vectors, log)` | `ClusterResult(labels, stats)` — labels 는 ids 순서의 int / None(노이즈) |
| 임베더 | `embedders.base.BaseEmbedder` | `DIM`, `_encode(images) → (N, DIM)`; 텍스트 검색이면 `embed_text(texts) → (N, DIM)` | 전처리·배치·L2 정규화는 base 가 처리 |

## 2. 등록 명령이 하는 일

```powershell
.\.venv\Scripts\python.exe bench\register.py detector  --name mydet --module detect.detectors.mydet_detector --class MydetDetector --param weights=weights/mydet.pt --limit 300
.\.venv\Scripts\python.exe bench\register.py clusterer --name myclu --module clustering.methods.myclu --class MycluClusterer --param knn=30 --max-points 3000
.\.venv\Scripts\python.exe bench\register.py embedder  --name myemb --module embedders.myemb_embedder --class MyembEmbedder --dim 768 --scope person --param model_id=...
```

1. **계약 검사** (`bench/check.py`): import → base 상속·메서드 재정의 → 생성자가 params 를 받는지 → 실제로 만들어 작은 입력으로 출력 규격
   (검출기: 빈 프레임 → Detection 목록 / 클러스터러: 합성 2덩어리 → labels 길이 / 임베더: 2장 → (2, dim) float32). FAIL 이면 멈춘다.
2. **yaml 등록** (원본 불변): 검출기 `pipeline_tracking_<이름>.yaml`(tracker/stitcher 는 `pipeline_tracking.yaml` 그대로), 클러스터러 `clusterer_<이름>.yaml`,
   임베더 `pipeline_<이름>.yaml`(`pipeline.yaml` 사본의 `retrievers:` 에 블록 추가, 주석 보존, 로더 검증).
3. **단계 벤치** (`bench/run.py detect | cluster | embed`) → 실행 폴더 `bench/runs/…` + 원장 행.
4. **순위**: 같은 단계의 이름별 최근 행과 비교한 위치, 채택 기준(`bench/criteria.py`) 판정. GUI **벤치마크** 탭에서 같은 표·그래프·채택→yaml.

옵션: `--check-only`(검사만) · `--no-bench`(yaml 까지) · `--no-instantiate`(정적 검사만) · `--overwrite` · `--root`(yaml 을 쓸 폴더) · `--ledger`.
GUI: 평가 탭 **11. 새 모델 등록**.

## 3. 그 다음

- 검출기: 드롭다운은 검출기 블록이 통째로 같은 사본만 합친다 — 같은 class 라도 가중치·임계값이 다르면(yolo26 / yolo26s) 별도 항목으로 보인다. 영상 1단계는 `pipeline_tracking*.yaml` 을 파일마다 한 항목으로 보여 스티처 변형도 고를 수 있다.
  전체 프레임 벤치는 `bench/run.py detect --config pipeline_tracking_<이름>.yaml --name <이름>`.
- 임베더: 단독 Re-ID 평가(GT crop)까지 자동. 검색·클러스터링에 쓰려면 운영 DB 에 새 named vector 를 적재해야 한다
  (`ingest/build_db.py --config pipeline_<이름>.yaml`, README 6.1) — 시간이 걸리므로 수동. 통합 검색 단독 평가(`eval/prw_eval_unified.py`)의 MODELS 는 아직 3종 고정.
- 클러스터러: 이미지 4b 단계의 `--method-config clusterer_<이름>.yaml`, 탐색은 `bench/optimize.py cluster --method-config …`.
