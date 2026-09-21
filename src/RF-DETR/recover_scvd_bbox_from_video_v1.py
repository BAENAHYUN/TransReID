"""
recover_scvd_bbox_from_video_v1.py

현재 forensic_object_g14의 SCVD crop 중,
원본 Normal_Videos_XXX_x264.mp4가 로컬에 존재하는 항목에 대해
crop 이미지를 해당 frame에 template matching하여 bbox를 복구한다.

왜 이 방식인가
-------------
현재 data/video_tracks/object의 tracks.jsonl은 scvd_object_tracks_v1과
다른 detection/tracking run에서 생성된 것으로 보이며,
같은 video/frame/label이어도 confidence/track_id가 일치하지 않는다.

따라서 기존 tracks.jsonl을 억지로 매칭하지 않고,
"crop 자체가 원본 frame의 어느 영역인지"를 직접 찾는다.

안전 조건
---------
- exact video number + frame_idx 사용
- crop 원본 크기 그대로 cv2.matchTemplate
- top1 score threshold
- top1 주변을 가린 뒤 top2 score 계산
- top1 - top2 margin이 충분할 때만 MATCHED
- 기본은 read-only
- --apply 일 때만 bbox payload 추가

실행
----
검증:
    python .\\src\\RF-DETR\\recover_scvd_bbox_from_video_v1.py

300개 제한:
    python .\\src\\RF-DETR\\recover_scvd_bbox_from_video_v1.py --limit 300

반영:
    python .\\src\\RF-DETR\\recover_scvd_bbox_from_video_v1.py --apply
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QDRANT = ROOT / "data" / "qdrant_local"
DEFAULT_COLLECTION = "forensic_object_g14"
DEFAULT_VIDEO_ROOT = ROOT / "data" / "videos"
DEFAULT_OUTPUT = ROOT / "outputs" / "recover_scvd_bbox_from_video_v1"


def parse_crop_video(name: str) -> Optional[int]:
    m = re.fullmatch(r"n(\d+)_converted", name, re.I)
    return int(m.group(1)) if m else None


def parse_frame(path: str) -> int:
    m = re.search(r"frame_(\d+)", Path(path).stem, re.I)
    return int(m.group(1)) if m else -1


def build_video_index(root: Path):
    out = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        m = re.fullmatch(r"Normal_Videos_(\d+)_x264", p.stem, re.I)
        if m:
            out[int(m.group(1))] = p
    return out


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None, fps

    return frame, fps


def second_peak_score(result: np.ndarray, loc: Tuple[int, int], tw: int, th: int) -> float:
    """
    top1 위치 주변을 충분히 가리고 두 번째 독립 peak를 찾는다.
    반복 패턴에 의한 ambiguous match를 걸러내기 위함.
    """
    x, y = loc
    masked = result.copy()

    # template 크기의 절반 정도 주변은 같은 물체/인접 peak로 간주해 제외
    rx = max(2, tw // 2)
    ry = max(2, th // 2)

    x1 = max(0, x - rx)
    y1 = max(0, y - ry)
    x2 = min(masked.shape[1], x + rx + 1)
    y2 = min(masked.shape[0], y + ry + 1)

    masked[y1:y2, x1:x2] = -1.0

    if masked.size == 0:
        return -1.0

    return float(masked.max())


def recover_bbox(frame: np.ndarray, crop: np.ndarray):
    fh, fw = frame.shape[:2]
    ch, cw = crop.shape[:2]

    if cw > fw or ch > fh or cw < 2 or ch < 2:
        return None

    # grayscale + mild blur: JPEG/codec 차이에 조금 더 robust
    fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    fg = cv2.GaussianBlur(fg, (3, 3), 0)
    cg = cv2.GaussianBlur(cg, (3, 3), 0)

    result = cv2.matchTemplate(fg, cg, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(result)
    second = second_peak_score(result, maxloc, cw, ch)

    x1, y1 = maxloc
    x2, y2 = x1 + cw, y1 + ch

    return {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "score": float(maxv),
        "second_score": float(second),
        "margin": float(maxv - second),
        "crop_width": int(cw),
        "crop_height": int(ch),
        "frame_width": int(fw),
        "frame_height": int(fh),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--apply", action="store_true")

    # 보수적 기본값. 결과 보고 조정 가능.
    ap.add_argument("--score-min", type=float, default=0.92)
    ap.add_argument("--margin-min", type=float, default=0.03)

    args = ap.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    video_index = build_video_index(args.video_root.resolve())

    print("=" * 80)
    print("RECOVER SCVD BBOX FROM VIDEO V1")
    print("Collection :", args.collection)
    print("Videos     :", len(video_index))
    print("Apply      :", args.apply)
    print("score_min  :", args.score_min)
    print("margin_min :", args.margin_min)
    print("=" * 80)

    client = QdrantClient(path=str(args.qdrant_path.resolve()))
    if not client.collection_exists(args.collection):
        raise SystemExit(f"collection 없음: {args.collection}")

    offset = None
    processed = 0
    counts = Counter()
    report = []

    # 같은 video/frame이면 frame decode를 재사용
    frame_cache_key = None
    frame_cache = None
    fps_cache = None

    while True:
        n = 128
        if args.limit:
            remain = args.limit - processed
            if remain <= 0:
                break
            n = min(n, remain)

        pts, offset = client.scroll(
            collection_name=args.collection,
            limit=n,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for p in pts:
            payload = dict(p.payload or {})
            qpath = str(payload.get("path") or payload.get("crop_path") or "")
            pp = Path(qpath)

            try:
                video_dir = pp.parent.parent.name
            except Exception:
                video_dir = ""

            video_num = parse_crop_video(video_dir)
            frame_idx = payload.get("frame_idx")
            try:
                frame_idx = int(frame_idx)
            except Exception:
                frame_idx = parse_frame(qpath)

            row = {
                "point_id": str(p.id),
                "path": qpath,
                "video_dir": video_dir,
                "video_num": video_num,
                "frame_idx": frame_idx,
                "label": payload.get("label"),
            }

            if video_num is None:
                row.update(status="SKIP_NON_NORMAL")
                counts["SKIP_NON_NORMAL"] += 1
                report.append(row)
                processed += 1
                continue

            video_path = video_index.get(video_num)
            if video_path is None:
                row.update(status="SKIP_VIDEO_MISSING")
                counts["SKIP_VIDEO_MISSING"] += 1
                report.append(row)
                processed += 1
                continue

            if frame_idx < 0:
                row.update(status="UNMATCHED", reason="frame_idx_missing")
                counts["UNMATCHED"] += 1
                report.append(row)
                processed += 1
                continue

            crop = cv2.imread(qpath, cv2.IMREAD_COLOR)
            if crop is None:
                row.update(status="UNMATCHED", reason="crop_read_failed")
                counts["UNMATCHED"] += 1
                report.append(row)
                processed += 1
                continue

            cache_key = (str(video_path), frame_idx)
            if cache_key != frame_cache_key:
                frame_cache, fps_cache = read_frame(video_path, frame_idx)
                frame_cache_key = cache_key

            if frame_cache is None:
                row.update(status="UNMATCHED", reason="frame_read_failed")
                counts["UNMATCHED"] += 1
                report.append(row)
                processed += 1
                continue

            rec = recover_bbox(frame_cache, crop)
            if rec is None:
                row.update(status="UNMATCHED", reason="template_invalid")
                counts["UNMATCHED"] += 1
                report.append(row)
                processed += 1
                continue

            row.update(
                match_score=rec["score"],
                second_score=rec["second_score"],
                match_margin=rec["margin"],
                candidate_bbox=rec["bbox"],
                crop_size=[rec["crop_width"], rec["crop_height"]],
                frame_size=[rec["frame_width"], rec["frame_height"]],
                video_path=str(video_path),
            )

            if rec["score"] < args.score_min:
                row.update(status="REVIEW", reason="low_template_score")
                counts["REVIEW"] += 1

            elif rec["margin"] < args.margin_min:
                row.update(status="REVIEW", reason="ambiguous_template_match")
                counts["REVIEW"] += 1

            else:
                row.update(status="MATCHED", reason="template_match")
                counts["MATCHED"] += 1

                if args.apply:
                    timestamp_sec = (
                        float(frame_idx / fps_cache)
                        if fps_cache and fps_cache > 0
                        else None
                    )

                    client.set_payload(
                        collection_name=args.collection,
                        payload={
                            "bbox": rec["bbox"],
                            "bbox_space": "frame",
                            "frame_idx": frame_idx,
                            "timestamp_sec": timestamp_sec,
                            "bbox_attach_source": "video_frame_template_match",
                            "bbox_attach_version": "template_v1",
                            "bbox_match_score": rec["score"],
                            "bbox_match_margin": rec["margin"],
                        },
                        points=[str(p.id)],
                        wait=True,
                    )

            report.append(row)
            processed += 1

            if processed % 100 == 0:
                print(
                    f"{processed:,} | matched={counts['MATCHED']:,} "
                    f"review={counts['REVIEW']:,}"
                )

        if offset is None:
            break

    eligible = counts["MATCHED"] + counts["REVIEW"] + counts["UNMATCHED"]

    summary = {
        "collection": args.collection,
        "processed": processed,
        "counts": dict(counts),
        "eligible_points": eligible,
        "matched_rate_eligible": counts["MATCHED"] / max(1, eligible),
        "score_min": args.score_min,
        "margin_min": args.margin_min,
        "applied": args.apply,
    }

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "match_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("COMPLETE")
    print("Processed          :", f"{processed:,}")
    print("Matched            :", f"{counts['MATCHED']:,}")
    print("Review             :", f"{counts['REVIEW']:,}")
    print("Unmatched          :", f"{counts['UNMATCHED']:,}")
    print("Non-Normal skipped :", f"{counts['SKIP_NON_NORMAL']:,}")
    print("Video missing      :", f"{counts['SKIP_VIDEO_MISSING']:,}")
    print("Eligible match rate:", f"{summary['matched_rate_eligible']*100:.2f}%")
    print("Applied            :", args.apply)
    print("Summary            :", out / "summary.json")
    print("=" * 80)

    client.close()


if __name__ == "__main__":
    main()
