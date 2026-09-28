# detect/ — 검출 단계

이미지 경로와 영상 경로가 여기 모여 있다.

- 이미지: `RF-DETR_batch.py` (배치 검출 + crop, checkpoint/resume) → `detect_rf.py` (검출기 플러그인 호출,
  detection_id·crop 파일명) → `rfdetr_adapter.py` (검출 결과 → `router.Detection`, `ingest/build_db.py` 가 사용).
- 영상: 아래 modular tracking starter (`runner.py` + `loader.py` + `detectors/ trackers/ stitchers/`).

## 검출기 플러그인 (2026-09-26 기준)

| module.class | 백엔드 | 비고 |
|---|---|---|
| `detect.detectors.rfdetr_detector.RFDETRDetector` | rfdetr (RF-DETR Medium, fp16) | 기본. `conf_threshold`, `filter_forensic`, `input_color` |
| `detect.detectors.yolo26_detector.YOLO26Detector` | ultralytics YOLO26 (NMS-free) | `weights`(yolo26n/s/m/l/x.pt, `weights/yolo/` 자동 다운로드), `imgsz`, `device`, `half`, `classes` |

둘 다 `BaseDetector.detect(frame_bgr, frame_idx=…) -> List[Detection]` 계약을 따르고, `class_name` 은 COCO 이름,
`class_id` 는 RF-DETR 과 같은 COCO 1-based id (`detect.base.coco_id_by_name`) 로 맞춘다. `filter_forensic` 은
`detect.base.FORENSIC_CLASSES` 를 공유한다.

교체 = yaml 의 `detector:` 블록만 변경. 준비된 판: `pipeline_tracking.yaml`(RF-DETR), `pipeline_tracking_yolo26.yaml`(YOLO26).

```powershell
# 영상 파이프라인 1단계에서 검출기 선택 (GUI 의 '검출/추적 설정 yaml' 과 같음)
.\.venv\Scripts\python.exe video\batch_preprocess_videos_parallel.py --tracking-config pipeline_tracking_yolo26.yaml ...

# 같은 PRW GT 로 검출기 비교 (AP / P·R / 크기별 재현율 / FPS)
.\.venv\Scripts\python.exe eval\detect_eval_prw.py --mode run --name yolo26m --detector-config pipeline_tracking_yolo26.yaml `
    --compare rfdetr_medium=eval\results\detect_prw\rfdetr_medium\detections.jsonl
```

새 검출기를 붙이려면 `detectors/` 에 `BaseDetector` 구현 하나를 추가하고 yaml 의 module/class 를 가리키면 된다.

이미지 경로도 같은 플러그인을 쓴다: `detect_rf.load_detector(spec)` → `detect_and_crop(detector, …)`.
`RF-DETR_batch.py --detector-config pipeline_tracking_yolo26.yaml` 로 교체하며, checkpoint config 에 검출기
식별자(module/class/params) 가 들어가 검출기를 바꾼 채 같은 출력 폴더로 resume 하면 거부된다. 이미지 경로에서는
`filter_forensic` 이 항상 False 로 강제된다 (클래스 필터는 `target_classes`, 기존 DB 는 전체 COCO class).

## Video tracking modular starter

최종 목표:
Video -> RF-DETR -> Tracker -> Stitcher -> tracks.jsonl -> DB builder

현재:
- RF-DETR adapter
- BoT-SORT adapter
- Deep OC-SORT adapter
- YAML 동적 로딩
- JSONL runner
- SUSHI adapter 자리

중요:
SUSHI는 단순 tracker.update() API가 아니라 별도 hierarchical graph association
구조이므로, 현재 파일은 임의 heuristic으로 대체하지 않고 명시적으로
NotImplementedError를 낸다.

1차 테스트:
pip install boxmot
python -m detect.runner "data\videos\YOUR_VIDEO.mp4" --no-stitch

BoT-SORT/Deep OC-SORT 교체는 pipeline.yaml의 tracker module/class만 변경.

위 1~4단계는 완료됐다(SUSHI 공식 repo 연결 포함). "5) DB 적재 연결"은 `build_video_db.py`
라는 이름의 스크립트가 아니라 아래 두 스크립트로 실제 구현됐다:
- `video/batch_preprocess_videos_parallel.py` — detect.runner 로 영상마다 검출·추적·스티칭 실행
- `ingest/batch_ingest_all_videos.py` — 그 결과를 crop 으로 뽑아 Qdrant 에 적재
  (내부적으로 `ingest/build_final_db_candidates_canonical.py` + `ingest/ingest_final_candidates_qdrant.py` 를 subprocess 로 호출)

GUI 파이프라인 탭의 "영상" 단계가 이 두 스크립트를 그대로 실행한다.
