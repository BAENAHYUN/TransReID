from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect_rf import load_detect_model, _class_name


VIDEO_ROOT = ROOT / "data" / "videos"
OUT_ROOT   = ROOT / "data" / "video_tracks" / "person"
TEMP_ROOT  = ROOT / "data" / "video_person_track_temp"
STATE_ROOT = ROOT / "data" / "video_person_track_state_v2"
STATE_FILE = STATE_ROOT / "completed.json"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

# ──────────────────────────────────────────────────────────────────────────
# Person 필터 기준 (detect_rf.py 와 동일한 값 유지)
#   MIN_PERSON_CROP_WIDTH  : MEVID 논문 기준 25px
#   MIN_PERSON_CROP_HEIGHT : 기존 경험값 120px
#   MIN_ASPECT_RATIO       : 사람은 세로가 가로보다 길다 (h/w).
#                            1.3 미만이면 차·소파·맨홀 등 가로 객체일 가능성 높음.
# ──────────────────────────────────────────────────────────────────────────
MIN_PERSON_CROP_W  = 25
MIN_PERSON_CROP_H  = 120
MIN_ASPECT_RATIO   = 1.3   # h / w


def _is_valid_person_bbox(x1, y1, x2, y2) -> bool:
    """bbox 단계에서 사람이 아닐 가능성이 높은 것을 사전 제거."""
    bw = x2 - x1
    bh = y2 - y1
    if bw < MIN_PERSON_CROP_W or bh < MIN_PERSON_CROP_H:
        return False   # 너무 작음
    if bh / bw < MIN_ASPECT_RATIO:
        return False   # 가로가 세로보다 넓음 → 사람이 아닐 확률 높음
    return True


def rel_video(p: Path) -> str:
    return p.relative_to(VIDEO_ROOT).as_posix()


def list_videos() -> list[Path]:
    if not VIDEO_ROOT.exists():
        return []
    return sorted(
        p for p in VIDEO_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


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


def probe_video(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    opened = cap.isOpened()
    fps    = float(cap.get(cv2.CAP_PROP_FPS))    if opened else 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0

    frame_ok = False
    if opened:
        ok, frame = cap.read()
        frame_ok = bool(ok and frame is not None and frame.size > 0)
    cap.release()

    return {
        "opened": opened, "frame_ok": frame_ok,
        "fps": fps, "frames": frames,
        "width": width, "height": height,
        "duration": frames / fps if fps > 0 else 0.0,
    }


def empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )


def detect_frame_exact_image_pipeline(
    model,
    frame_bgr: np.ndarray,
    temp_frame_path: Path,
    threshold: float,
):
    """
    이미지 detect_rf.py 와 같은 방식으로 RF-DETR 에 파일 경로를 넘긴다.
    BGR/RGB 혼동 없음.

    Person 필터 (bbox 단계):
      ① 최소 크기  : w >= MIN_PERSON_CROP_W, h >= MIN_PERSON_CROP_H
      ② 종횡비     : h/w >= MIN_ASPECT_RATIO
      → 이 두 조건을 통과한 bbox 만 ByteTrack 에 넘긴다.
    """
    temp_frame_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(temp_frame_path), frame_bgr):
        raise RuntimeError(f"Failed to save temp frame: {temp_frame_path}")

    dets = model.predict(str(temp_frame_path), threshold=threshold)

    h, w = frame_bgr.shape[:2]
    raw_records   = []
    person_boxes  = []
    person_confs  = []
    person_class_ids = []

    for i in range(len(dets)):
        cid   = int(dets.class_id[i])
        cname = _class_name(cid)
        conf  = float(dets.confidence[i])

        x1, y1, x2, y2 = map(int, dets.xyxy[i])
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        raw_records.append({
            "class_id": cid, "class_name": cname,
            "confidence": conf, "bbox": [x1, y1, x2, y2],
        })

        if cname.lower() == "person":
            # ── 핵심 필터: 크기 + 종횡비 ──────────────────────────────
            if not _is_valid_person_bbox(x1, y1, x2, y2):
                continue   # false positive 사전 차단
            # ──────────────────────────────────────────────────────────
            person_boxes.append([x1, y1, x2, y2])
            person_confs.append(conf)
            person_class_ids.append(cid)

    if not person_boxes:
        return empty_detections(), raw_records

    person_dets = sv.Detections(
        xyxy=np.asarray(person_boxes, dtype=np.float32),
        confidence=np.asarray(person_confs, dtype=np.float32),
        class_id=np.asarray(person_class_ids, dtype=int),
    )
    return person_dets, raw_records


def make_tracker(interval_sec: float):
    sample_fps = max(1, int(round(1.0 / interval_sec)))
    try:
        return sv.ByteTrack(frame_rate=sample_fps)
    except TypeError:
        return sv.ByteTrack()


def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def process_video(
    model,
    video_path: Path,
    threshold: float,
    interval_sec: float,
    preview_every: int,
    force: bool,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    frame_step = max(1, int(round(fps * interval_sec)))

    video_out = OUT_ROOT / video_path.stem
    if force and video_out.exists():
        shutil.rmtree(video_out)
    video_out.mkdir(parents=True, exist_ok=True)

    preview_dir = video_out / "preview"
    if preview_every > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    temp_frame   = TEMP_ROOT / video_path.stem / "current_frame.jpg"
    metadata_path = video_out / "tracks.jsonl"
    summary_path  = video_out / "raw_detection_summary.json"

    tracker = make_tracker(interval_sec)

    sampled     = 0
    frame_idx   = 0
    saved_crops = 0
    rejected_size   = 0   # 크기 부족으로 버린 crop
    rejected_ratio  = 0   # 종횡비 불량으로 버린 crop
    track_ids   = set()
    raw_total   = 0
    raw_person  = 0
    class_counts = Counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        sampled += 1

        person_dets, raw_records = detect_frame_exact_image_pipeline(
            model=model,
            frame_bgr=frame,
            temp_frame_path=temp_frame,
            threshold=threshold,
        )

        raw_total  += len(raw_records)
        for r in raw_records:
            class_counts[r["class_name"]] += 1
            if r["class_name"].lower() == "person":
                raw_person += 1

        tracked = tracker.update_with_detections(person_dets)

        preview = frame.copy() if preview_every > 0 else None

        if preview is not None:
            for r in raw_records:
                x1, y1, x2, y2 = r["bbox"]
                label = f"RAW {r['class_name']} {r['confidence']:.2f}"
                cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 180, 0), 1)
                cv2.putText(preview, label, (x1, max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1, cv2.LINE_AA)

        for i in range(len(tracked)):
            if tracked.tracker_id is None:
                continue

            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])

            fh, fw = frame.shape[:2]
            x1 = max(0, min(fw, x1))
            y1 = max(0, min(fh, y1))
            x2 = max(0, min(fw, x2))
            y2 = max(0, min(fh, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            ch, cw = crop.shape[:2]

            # ── crop 단계 2차 필터 (tracker bbox 팽창 후 재검증) ──────
            if cw < MIN_PERSON_CROP_W or ch < MIN_PERSON_CROP_H:
                rejected_size += 1
                continue
            if ch / cw < MIN_ASPECT_RATIO:
                rejected_ratio += 1
                continue
            # ──────────────────────────────────────────────────────────

            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0

            track_dir = video_out / f"track_{tid:04d}"
            track_dir.mkdir(parents=True, exist_ok=True)

            crop_path = track_dir / f"frame_{frame_idx:08d}_{conf:.4f}.jpg"
            if not cv2.imwrite(str(crop_path), crop):
                continue

            record = {
                "video":          rel_video(video_path),
                "video_path":     str(video_path.resolve()),
                "video_id":       video_path.stem,
                "media_type":     "video",
                "scope":          "person",
                "class_name":     "person",
                "track_id":       tid,
                "frame_idx":      frame_idx,
                "timestamp_sec":  frame_idx / fps,
                "confidence":     conf,
                "bbox":           [x1, y1, x2, y2],
                "bbox_space":     "frame",
                "crop_path":      str(crop_path.resolve()),
            }
            append_jsonl(metadata_path, record)

            saved_crops += 1
            track_ids.add(tid)

            if preview is not None:
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(preview, f"TRACK T{tid} {conf:.2f}",
                            (x1, min(fh - 5, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        if preview is not None and sampled % preview_every == 0:
            cv2.imwrite(str(preview_dir / f"frame_{frame_idx:08d}.jpg"), preview)

        if sampled % 50 == 0:
            print(
                f"    sampled={sampled:,} raw={raw_total:,} "
                f"raw_person={raw_person:,} crops={saved_crops:,} "
                f"rej_size={rejected_size} rej_ratio={rejected_ratio}",
                end="\r",
            )

        frame_idx += 1

    cap.release()
    print()

    metadata_rows = 0
    if metadata_path.exists():
        metadata_rows = sum(
            1 for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    summary = {
        "video":                  rel_video(video_path),
        "sampled_frames":         sampled,
        "raw_detection_total":    raw_total,
        "raw_person_detections":  raw_person,
        "raw_class_counts":       dict(class_counts.most_common()),
        "saved_person_crops":     saved_crops,
        "rejected_size":          rejected_size,
        "rejected_ratio":         rejected_ratio,
        "unique_person_tracks":   len(track_ids),
        "metadata_rows":          metadata_rows,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary, summary_path, metadata_path


def main():
    ap = argparse.ArgumentParser(description="Video person track V4 (detect_rf 기반)")
    ap.add_argument("--max-videos",   type=int,   default=None)
    ap.add_argument("--force",        action="store_true")
    ap.add_argument("--dry-run",      action="store_true")
    ap.add_argument("--preview-every",type=int,   default=0,
                    help="N 프레임마다 bbox 시각화 저장 (0=비활성)")
    ap.add_argument("--threshold",    type=float, default=0.5)
    ap.add_argument("--interval-sec", type=float, default=0.2,
                    help="프레임 샘플링 간격 (초). 기본=0.2")
    ap.add_argument("--min-aspect",   type=float, default=MIN_ASPECT_RATIO,
                    help=f"최소 h/w 종횡비. 기본={MIN_ASPECT_RATIO}")
    args = ap.parse_args()

    # CLI 로 종횡비 조정 가능
    global MIN_ASPECT_RATIO
    MIN_ASPECT_RATIO = args.min_aspect

    videos = list_videos()
    if args.max_videos is not None:
        videos = videos[:args.max_videos]

    print("=" * 76)
    print("VIDEO PERSON TRACK V4")
    print("=" * 76)
    print("input root    :", VIDEO_ROOT)
    print("output root   :", OUT_ROOT)
    print("videos        :", len(videos))
    print("threshold     :", args.threshold)
    print("interval sec  :", args.interval_sec)
    print("min size      :", f"{MIN_PERSON_CROP_W}x{MIN_PERSON_CROP_H}px")
    print("min aspect    :", f"h/w >= {MIN_ASPECT_RATIO}")
    print("preview every :", args.preview_every)

    if not videos:
        raise RuntimeError(f"No videos under {VIDEO_ROOT}")

    print("\n[CHECK 1] VIDEO DECODE")
    for vp in videos:
        info = probe_video(vp)
        status = "PASS" if info["opened"] and info["frame_ok"] else "FAIL"
        print(
            f"  [{status}] {rel_video(vp)} | "
            f"{info['width']}x{info['height']} fps={info['fps']:.3f} "
            f"frames={info['frames']} duration={info['duration']:.2f}s"
        )
        if status == "FAIL":
            raise RuntimeError(f"Decode failed: {vp}")

    if args.dry_run:
        print("\nDRY RUN PASS")
        return

    done    = set() if args.force else load_state()
    pending = [v for v in videos if rel_video(v) not in done]
    print("\npending       :", len(pending))

    if not pending:
        print("Nothing to do.")
        return

    model   = load_detect_model()
    started = time.time()

    for idx, vp in enumerate(pending, 1):
        print(f"\n[{idx}/{len(pending)}] {vp.name}")

        summary, summary_path, metadata_path = process_video(
            model=model,
            video_path=vp,
            threshold=args.threshold,
            interval_sec=args.interval_sec,
            preview_every=args.preview_every,
            force=args.force,
        )

        print("  sampled frames       :", summary["sampled_frames"])
        print("  raw detections       :", summary["raw_detection_total"])
        print("  raw person dets      :", summary["raw_person_detections"])
        print("  rejected (size)      :", summary["rejected_size"])
        print("  rejected (ratio)     :", summary["rejected_ratio"])
        print("  raw classes          :", summary["raw_class_counts"])
        print("  person crops saved   :", summary["saved_person_crops"])
        print("  unique person tracks :", summary["unique_person_tracks"])
        print("  metadata rows        :", summary["metadata_rows"])
        print("  summary              :", summary_path)

        if summary["metadata_rows"] != summary["saved_person_crops"]:
            raise RuntimeError("crop count != metadata rows — 중단")

        done.add(rel_video(vp))
        save_state(done)

        print(f"  elapsed              : {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
