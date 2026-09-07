"""
Normal_Videos person track/crop extractor (standalone, V4-compatible).

엔진: RF-DETR Medium (detection) + sv.ByteTrack (tracking)
출력: data/video_person_tracks_v4/{video_stem}/track_{tid:04d}/frame_{n:08d}_person_01.jpg

build_video_index.py 가 기대하는 rsplit("_", 2) 파싱을 그대로 따른다:
  stem = "frame_00001770_person_01"
  rsplit → ["frame_00001770", "person", "01"]  →  person_id = "01"

미처리 영상만 처리 (data/video_person_tracks_v4/ 에 디렉터리가 없는 것).
--force 로 강제 재처리 가능.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

# ──────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
VIDEO_ROOT  = ROOT / "data" / "videos"
TRACK_ROOT  = ROOT / "data" / "video_person_tracks_v4"
STATE_ROOT  = ROOT / "data" / "video_person_track_state_v4"
STATE_FILE  = STATE_ROOT / "completed.json"
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

# Detection / crop 파라미터
CONF_THRESHOLD       = 0.50
MIN_PERSON_W         = 25
MIN_PERSON_H         = 120
INTERVAL_SEC         = 1.0   # 샘플링 간격 (초)
# ──────────────────────────────────────────


def load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")).get("completed", []))
    except Exception:
        return set()


def save_state(done: set[str]) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"completed": sorted(done)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_videos() -> list[Path]:
    return sorted(
        p for p in VIDEO_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def already_tracked(video: Path) -> bool:
    """track_root 에 이 영상 디렉터리가 이미 존재하면 완료로 간주."""
    return (TRACK_ROOT / video.stem).is_dir()


def load_model():
    print("[*] Loading RF-DETR Medium ...")
    model = RFDETRMedium()
    try:
        model.inference(compile=False, dtype="float16")
    except Exception:
        try:
            model.inference(compile=False)
        except Exception:
            pass
    print("[+] RF-DETR ready")
    return model


def detect_persons(model, frame: np.ndarray, threshold: float) -> list[dict]:
    """RF-DETR 로 person 만 검출, 최소 크기 필터 포함."""
    dets = model.predict(frame, threshold=threshold)
    h, w = frame.shape[:2]
    out = []

    for i in range(len(dets)):
        class_id   = int(dets.class_id[i])
        class_name = str(COCO_CLASSES[class_id])
        if class_name.lower() != "person":
            continue

        conf = float(dets.confidence[i])
        x1, y1, x2, y2 = map(int, dets.xyxy[i])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        out.append({
            "class_id": class_id,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
        })
    return out


def process_video(
    model,
    video_path: Path,
    threshold: float,
    interval_sec: float,
    min_w: int,
    min_h: int,
) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    frame_step = max(1, int(round(fps * interval_sec)))

    tracker   = sv.ByteTrack()
    video_out = TRACK_ROOT / video_path.stem

    sampled = 0
    saved   = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        sampled += 1
        detected = detect_persons(model, frame, threshold)

        if detected:
            xyxy  = np.asarray([d["bbox"]       for d in detected], dtype=np.float32)
            confs = np.asarray([d["confidence"] for d in detected], dtype=np.float32)
            cids  = np.zeros(len(detected), dtype=int)

            tracked = tracker.update_with_detections(
                sv.Detections(xyxy=xyxy, confidence=confs, class_id=cids)
            )

            for i in range(len(tracked)):
                tid  = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else i
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue
                ch, cw = crop.shape[:2]
                if cw < min_w or ch < min_h:
                    continue

                track_dir = video_out / f"track_{tid:04d}"
                track_dir.mkdir(parents=True, exist_ok=True)

                fname = f"frame_{frame_idx:08d}_person_01.jpg"
                if cv2.imwrite(str(track_dir / fname), crop):
                    saved += 1

        if sampled % 50 == 0:
            print(f"    sampled={sampled:,}  saved={saved:,}", end="\r")

        frame_idx += 1

    cap.release()
    print()
    return sampled, saved


def main():
    ap = argparse.ArgumentParser(description="Normal_Videos person track/crop (standalone V4)")
    ap.add_argument("--threshold",    type=float, default=CONF_THRESHOLD)
    ap.add_argument("--interval-sec", type=float, default=INTERVAL_SEC,
                    help=f"프레임 샘플링 간격 (초). 기본={INTERVAL_SEC}")
    ap.add_argument("--min-width",    type=int,   default=MIN_PERSON_W)
    ap.add_argument("--min-height",   type=int,   default=MIN_PERSON_H)
    ap.add_argument("--max-videos",   type=int,   default=None,
                    help="처리 영상 수 제한 (디버그용)")
    ap.add_argument("--force",        action="store_true",
                    help="이미 처리된 영상도 재처리")
    args = ap.parse_args()

    videos = list_videos()

    if not args.force:
        videos = [v for v in videos if not already_tracked(v)]

    if args.max_videos:
        videos = videos[:args.max_videos]

    print("=" * 76)
    print("VIDEO PERSON TRACK V4  (standalone)")
    print("=" * 76)
    print(f"video root   : {VIDEO_ROOT}")
    print(f"track root   : {TRACK_ROOT}")
    print(f"대상 영상    : {len(videos):,} 개")
    print(f"threshold    : {args.threshold}")
    print(f"interval sec : {args.interval_sec}")
    print(f"min crop     : {args.min_width}x{args.min_height}")
    print()

    if not videos:
        print("Nothing to do. (모든 영상이 이미 처리됨)")
        return

    TRACK_ROOT.mkdir(parents=True, exist_ok=True)
    model = load_model()

    done        = load_state()
    total_saved = 0
    started     = time.time()

    for idx, video in enumerate(videos, 1):
        vid_key = video.stem
        print(f"\n[{idx}/{len(videos)}] {video.name}")
        t0 = time.time()
        try:
            sampled, saved = process_video(
                model, video,
                threshold=args.threshold,
                interval_sec=args.interval_sec,
                min_w=args.min_width,
                min_h=args.min_height,
            )
        except Exception as e:
            print(f"  [!] 오류: {e}")
            continue

        total_saved += saved
        done.add(vid_key)
        save_state(done)

        elapsed = time.time() - t0
        print(
            f"  sampled={sampled:,}  crops={saved:,}  "
            f"time={elapsed:.1f}s  total_crops={total_saved:,}"
        )
        print(
            f"  전체 경과={(time.time()-started)/60:.1f}m"
        )

    print("\n" + "=" * 76)
    print("PERSON TRACK V4 COMPLETE")
    print(f"총 저장 crop : {total_saved:,} 개")
    print(f"track root   : {TRACK_ROOT}")
    print(f"state file   : {STATE_FILE}")


if __name__ == "__main__":
    main()
