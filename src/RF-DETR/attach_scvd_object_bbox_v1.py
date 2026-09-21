"""
attach_scvd_object_bbox_v1.py

SCVD object tracking metadata(data/video_tracks/object/*/tracks.jsonl)를
forensic_object_g14 Qdrant payload에 연결해서 bbox/confidence/frame metadata를 붙인다.

매칭 전략
---------
Qdrant crop:
    data/scvd_object_tracks_v1/n003_converted/potted_plant_0001/frame_00000000_....jpg

Tracking metadata:
    data/video_tracks/object/Normal_Videos_003_x264/tracks.jsonl
    video_id=Normal_Videos_003_x264
    track_id=1
    frame_idx=0
    class_name="potted plant"
    bbox=[x1,y1,x2,y2]

안전 매칭 키:
    namespace(normal) + video_number(003) + frame_idx + normalized_label

주의
----
- n003_converted -> Normal_Videos_003_x264 만 매칭
- x264의 264를 video id로 잘못 읽지 않음
- label은 underscore/space 차이를 normalize
- track_id는 SCVD crop 폴더의 *_0001 숫자와 tracking track_id가
  실제로 일치할 때 우선 사용
- track_id가 불일치하면 동일 video/frame/label 후보 중 confidence가 높은
  단일 후보만 fallback
- ambiguous candidate는 쓰지 않고 skip

기본 실행:
    python .\src\RF-DETR\attach_scvd_object_bbox_v1.py

테스트:
    python .\src\RF-DETR\attach_scvd_object_bbox_v1.py --limit 300

실제 payload 반영:
    python .\src\RF-DETR\attach_scvd_object_bbox_v1.py --apply

출력:
    outputs/attach_scvd_object_bbox_v1/
      attach_report.json
      unmatched.json
      ambiguous.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QDRANT = ROOT / "data" / "qdrant_local"
DEFAULT_COLLECTION = "forensic_object_g14"
DEFAULT_META_ROOT = ROOT / "data" / "video_tracks" / "object"
DEFAULT_OUTPUT = ROOT / "outputs" / "attach_scvd_object_bbox_v1"


def normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def parse_scvd_video_key(name: str) -> Optional[Tuple[str, int]]:
    """
    n003_converted -> ("normal", 3)
    v020_converted 등은 아직 안전 매칭 규칙이 없으므로 None.
    """
    m = re.fullmatch(r"n(\d+)_converted", name, re.I)
    if m:
        return ("normal", int(m.group(1)))
    return None


def parse_tracking_video_key(name: str) -> Optional[Tuple[str, int]]:
    """
    Normal_Videos_003_x264 -> ("normal", 3)
    """
    m = re.fullmatch(r"Normal_Videos_(\d+)_x264", name, re.I)
    if m:
        return ("normal", int(m.group(1)))
    return None


def parse_track_folder(name: str) -> Tuple[str, Optional[int]]:
    """
    potted_plant_0001 -> ("potted plant", 1)
    dining_table_0012 -> ("dining table", 12)
    """
    m = re.fullmatch(r"(.+?)_(\d+)", name)
    if not m:
        return normalize_label(name), None

    return normalize_label(m.group(1)), int(m.group(2))


def parse_frame_idx(path: str) -> int:
    m = re.search(r"frame_(\d+)", Path(path).stem, re.I)
    return int(m.group(1)) if m else -1


@dataclass
class TrackMeta:
    video_key: Tuple[str, int]
    video_id: str
    track_id: int
    frame_idx: int
    label: str
    confidence: float
    bbox: List[float]
    timestamp_sec: Optional[float]
    crop_path: str


@dataclass
class ReportRow:
    point_id: str
    qdrant_path: str
    video_dir: str
    track_dir: str
    frame_idx: int
    label: str
    track_id: Optional[int]
    status: str
    matched_video_id: Optional[str]
    matched_track_id: Optional[int]
    matched_confidence: Optional[float]
    matched_bbox: Optional[List[float]]
    reason: str


def load_tracking_metadata(meta_root: Path):
    by_exact = defaultdict(list)
    by_fallback = defaultdict(list)

    video_dirs = [p for p in meta_root.iterdir() if p.is_dir()]
    loaded_lines = 0
    usable_lines = 0

    for video_dir in sorted(video_dirs):
        vkey = parse_tracking_video_key(video_dir.name)
        if vkey is None:
            continue

        jsonl = video_dir / "tracks.jsonl"
        if not jsonl.is_file():
            continue

        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                loaded_lines += 1

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                bbox = row.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                try:
                    track_id = int(row.get("track_id"))
                    frame_idx = int(row.get("frame_idx"))
                    confidence = float(row.get("confidence", 0.0))
                    bbox = [float(x) for x in bbox]
                except Exception:
                    continue

                label = normalize_label(
                    str(
                        row.get("final_class_name")
                        or row.get("class_name")
                        or row.get("raw_class_name")
                        or ""
                    )
                )
                if not label:
                    continue

                meta = TrackMeta(
                    video_key=vkey,
                    video_id=str(row.get("video_id", video_dir.name)),
                    track_id=track_id,
                    frame_idx=frame_idx,
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    timestamp_sec=(
                        float(row["timestamp_sec"])
                        if row.get("timestamp_sec") is not None
                        else None
                    ),
                    crop_path=str(row.get("crop_path", "")),
                )

                by_exact[(vkey, frame_idx, label, track_id)].append(meta)
                by_fallback[(vkey, frame_idx, label)].append(meta)
                usable_lines += 1

    return by_exact, by_fallback, loaded_lines, usable_lines


def pick_match(
    video_key,
    frame_idx,
    label,
    track_id,
    by_exact,
    by_fallback,
):
    if track_id is not None:
        exact = by_exact.get((video_key, frame_idx, label, track_id), [])
        if len(exact) == 1:
            return exact[0], "exact"
        if len(exact) > 1:
            exact = sorted(exact, key=lambda x: x.confidence, reverse=True)
            if len(exact) >= 2 and abs(exact[0].confidence - exact[1].confidence) < 1e-9:
                return None, "ambiguous_exact"
            return exact[0], "exact_high_conf"

    candidates = by_fallback.get((video_key, frame_idx, label), [])
    if len(candidates) == 1:
        return candidates[0], "fallback_single"

    if len(candidates) > 1:
        candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
        # track_id 정보가 있는데 exact 실패했고 여러 후보가 있으면 잘못 붙일 위험이 커서 skip
        if track_id is not None:
            return None, "ambiguous_fallback"
        if len(candidates) >= 2 and abs(candidates[0].confidence - candidates[1].confidence) < 1e-9:
            return None, "ambiguous_fallback"
        return candidates[0], "fallback_high_conf"

    return None, "not_found"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    meta_root = args.meta_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_root.is_dir():
        raise SystemExit(f"metadata root 없음: {meta_root}")

    print("=" * 80)
    print("ATTACH SCVD OBJECT BBOX V1")
    print(f"Collection : {args.collection}")
    print(f"Meta root  : {meta_root}")
    print(f"Apply      : {args.apply}")
    print("=" * 80)

    print("[1/3] tracking metadata index 생성")
    by_exact, by_fallback, loaded_lines, usable_lines = load_tracking_metadata(meta_root)
    print(f"      jsonl lines={loaded_lines:,}")
    print(f"      usable={usable_lines:,}")
    print(f"      exact keys={len(by_exact):,}")

    client = QdrantClient(path=str(args.qdrant_path.resolve()))
    if not client.collection_exists(args.collection):
        raise SystemExit(f"collection 없음: {args.collection}")

    print("[2/3] Qdrant payload 매칭")

    reports: List[ReportRow] = []
    unmatched = []
    ambiguous = []

    offset = None
    processed = 0
    matched = 0
    unsupported_video = 0

    while True:
        n = 256
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
            qpath = str(payload.get("path", ""))

            pp = Path(qpath)
            try:
                track_dir = pp.parent.name
                video_dir = pp.parent.parent.name
            except Exception:
                track_dir = ""
                video_dir = ""

            video_key = parse_scvd_video_key(video_dir)
            label, track_id = parse_track_folder(track_dir)

            payload_label = normalize_label(str(payload.get("label", "")))
            if payload_label:
                label = payload_label

            frame_idx = payload.get("frame_idx")
            try:
                frame_idx = int(frame_idx)
            except Exception:
                frame_idx = parse_frame_idx(qpath)

            if video_key is None:
                unsupported_video += 1
                rr = ReportRow(
                    point_id=str(p.id),
                    qdrant_path=qpath,
                    video_dir=video_dir,
                    track_dir=track_dir,
                    frame_idx=frame_idx,
                    label=label,
                    track_id=track_id,
                    status="UNMATCHED",
                    matched_video_id=None,
                    matched_track_id=None,
                    matched_confidence=None,
                    matched_bbox=None,
                    reason="unsupported_video_name",
                )
                reports.append(rr)
                unmatched.append(asdict(rr))
                processed += 1
                continue

            meta, reason = pick_match(
                video_key,
                frame_idx,
                label,
                track_id,
                by_exact,
                by_fallback,
            )

            if meta is None:
                rr = ReportRow(
                    point_id=str(p.id),
                    qdrant_path=qpath,
                    video_dir=video_dir,
                    track_dir=track_dir,
                    frame_idx=frame_idx,
                    label=label,
                    track_id=track_id,
                    status="AMBIGUOUS" if reason.startswith("ambiguous") else "UNMATCHED",
                    matched_video_id=None,
                    matched_track_id=None,
                    matched_confidence=None,
                    matched_bbox=None,
                    reason=reason,
                )
                reports.append(rr)
                (ambiguous if rr.status == "AMBIGUOUS" else unmatched).append(asdict(rr))
                processed += 1
                continue

            rr = ReportRow(
                point_id=str(p.id),
                qdrant_path=qpath,
                video_dir=video_dir,
                track_dir=track_dir,
                frame_idx=frame_idx,
                label=label,
                track_id=track_id,
                status="MATCHED",
                matched_video_id=meta.video_id,
                matched_track_id=meta.track_id,
                matched_confidence=meta.confidence,
                matched_bbox=meta.bbox,
                reason=reason,
            )
            reports.append(rr)
            matched += 1

            if args.apply:
                client.set_payload(
                    collection_name=args.collection,
                    payload={
                        "bbox": meta.bbox,
                        "bbox_space": "frame",
                        "confidence": meta.confidence,
                        "source_video_id": meta.video_id,
                        "source_track_id": meta.track_id,
                        "timestamp_sec": meta.timestamp_sec,
                        "bbox_attach_source": "video_tracks/object/tracks.jsonl",
                        "bbox_attach_match": reason,
                    },
                    points=[str(p.id)],
                    wait=True,
                )

            processed += 1

            if processed % 250 == 0:
                print(f"      {processed:,} processed / matched={matched:,}")

        if offset is None:
            break

    print("[3/3] report 저장")

    report = {
        "collection": args.collection,
        "processed": processed,
        "matched": matched,
        "unmatched": len(unmatched),
        "ambiguous": len(ambiguous),
        "unsupported_video": unsupported_video,
        "match_rate": matched / processed if processed else 0.0,
        "applied": args.apply,
        "tracking_lines": loaded_lines,
        "usable_tracking_lines": usable_lines,
    }

    (out_dir / "attach_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "unmatched.json").write_text(
        json.dumps(unmatched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ambiguous.json").write_text(
        json.dumps(ambiguous, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("COMPLETE")
    print(f"Processed        : {processed:,}")
    print(f"Matched          : {matched:,}")
    print(f"Unmatched        : {len(unmatched):,}")
    print(f"Ambiguous        : {len(ambiguous):,}")
    print(f"Unsupported video: {unsupported_video:,}")
    print(f"Match rate       : {(matched / processed * 100) if processed else 0:.2f}%")
    print(f"Applied          : {args.apply}")
    print(f"Report           : {out_dir / 'attach_report.json'}")
    print("=" * 80)

    client.close()


if __name__ == "__main__":
    main()
