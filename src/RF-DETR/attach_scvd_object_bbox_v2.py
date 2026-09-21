from __future__ import annotations
import argparse, json, re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QDRANT = ROOT / "data" / "qdrant_local"
DEFAULT_COLLECTION = "forensic_object_g14"
DEFAULT_META_ROOT = ROOT / "data" / "video_tracks" / "object"
DEFAULT_OUTPUT = ROOT / "outputs" / "attach_scvd_object_bbox_v2"

def norm_label(s):
    s = str(s or "").strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", s)

def parse_crop_video(name):
    m = re.fullmatch(r"n(\d+)_converted", name, re.I)
    return int(m.group(1)) if m else None

def parse_track_video(name):
    m = re.fullmatch(r"Normal_Videos_(\d+)_x264", name, re.I)
    return int(m.group(1)) if m else None

def parse_frame(path):
    m = re.search(r"frame_(\d+)", Path(path).stem, re.I)
    return int(m.group(1)) if m else -1

def parse_conf(path):
    m = re.search(r"frame_\d+_([0-9]*\.?[0-9]+)$", Path(path).stem, re.I)
    return float(m.group(1)) if m else None

def parse_track_dir(name):
    m = re.fullmatch(r"(.+?)_(\d+)", name)
    return (norm_label(m.group(1)), int(m.group(2))) if m else (norm_label(name), None)

def image_size(path):
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None

def load_meta(root):
    by_key = defaultdict(list)
    videos = set()
    lines = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        vn = parse_track_video(d.name)
        if vn is None:
            continue
        jf = d / "tracks.jsonl"
        if not jf.is_file():
            continue
        videos.add(vn)
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                lines += 1
                try:
                    r = json.loads(line)
                    bbox = r.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    label = norm_label(r.get("final_class_name") or r.get("class_name") or r.get("raw_class_name"))
                    frame = int(r["frame_idx"])
                    item = {
                        "video_id": str(r.get("video_id", d.name)),
                        "track_id": int(r.get("track_id", -1)),
                        "frame_idx": frame,
                        "label": label,
                        "confidence": float(r.get("confidence", 0.0)),
                        "bbox": [float(x) for x in bbox],
                        "width": int(r.get("width", max(0, bbox[2]-bbox[0]))),
                        "height": int(r.get("height", max(0, bbox[3]-bbox[1]))),
                        "timestamp_sec": float(r["timestamp_sec"]) if r.get("timestamp_sec") is not None else None,
                    }
                    by_key[(vn, frame, label)].append(item)
                except Exception:
                    pass
    return by_key, videos, lines

def choose(cands, crop_conf, crop_wh, crop_tid, conf_tol, size_tol, ambiguity_margin):
    if not cands:
        return None, "not_found", None
    scored = []
    for m in cands:
        cd = abs(m["confidence"] - crop_conf) if crop_conf is not None else 1.0
        sd = (abs(m["width"]-crop_wh[0]) + abs(m["height"]-crop_wh[1])) if crop_wh else 999999
        bonus = -0.25 if crop_tid is not None and crop_tid == m["track_id"] else 0.0
        score = cd*1000.0 + min(sd,1000)*0.01 + bonus
        scored.append((score, cd, sd, m))
    scored.sort(key=lambda x: x[0])
    best = scored[0]
    if crop_conf is not None and best[1] > conf_tol:
        return None, "confidence_mismatch", {"best_conf_diff":best[1], "best_size_diff":best[2]}
    if crop_wh is not None and best[2] > size_tol:
        return None, "size_mismatch", {"best_conf_diff":best[1], "best_size_diff":best[2]}
    if len(scored) >= 2 and scored[1][0]-best[0] < ambiguity_margin:
        return None, "ambiguous", {"best_score":best[0], "second_score":scored[1][0]}
    reason = "track_conf_size_match" if crop_tid == best[3]["track_id"] else "conf_size_match"
    return best[3], reason, {"best_conf_diff":best[1], "best_size_diff":best[2], "score":best[0]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--conf-tol", type=float, default=0.00015)
    ap.add_argument("--size-tol", type=int, default=4)
    ap.add_argument("--ambiguity-margin", type=float, default=0.05)
    args = ap.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ATTACH SCVD OBJECT BBOX V2")
    print("Collection :", args.collection)
    print("Apply      :", args.apply)
    print("="*80)

    by_key, available_videos, lines = load_meta(args.meta_root.resolve())
    print("[1/3] metadata lines=", f"{lines:,}", "videos=", len(available_videos))

    client = QdrantClient(path=str(args.qdrant_path.resolve()))
    if not client.collection_exists(args.collection):
        raise SystemExit("collection 없음: "+args.collection)

    offset = None
    processed = 0
    counts = Counter()
    report = []

    print("[2/3] matching")
    while True:
        n = 256
        if args.limit:
            remain = args.limit - processed
            if remain <= 0: break
            n = min(n, remain)
        pts, offset = client.scroll(args.collection, limit=n, offset=offset, with_payload=True, with_vectors=False)

        for p in pts:
            payload = dict(p.payload or {})
            qpath = str(payload.get("path") or payload.get("crop_path") or "")
            pp = Path(qpath)
            video_dir = pp.parent.parent.name if len(pp.parts) >= 3 else ""
            track_dir = pp.parent.name if len(pp.parts) >= 2 else ""
            vn = parse_crop_video(video_dir)
            label_dir, tid = parse_track_dir(track_dir)
            label = norm_label(payload.get("label") or label_dir)
            try:
                frame = int(payload.get("frame_idx"))
            except Exception:
                frame = parse_frame(qpath)
            conf = parse_conf(qpath)
            wh = image_size(qpath)

            row = {"point_id":str(p.id),"path":qpath,"video_dir":video_dir,"frame_idx":frame,
                   "label":label,"crop_track_id":tid,"crop_confidence":conf,"crop_size":list(wh) if wh else None}

            if vn is None:
                row.update(status="SKIP_NO_METADATA_FAMILY", reason="non_normal_family")
                counts["SKIP_NO_METADATA_FAMILY"] += 1
                report.append(row); processed += 1; continue

            if vn not in available_videos:
                row.update(status="SKIP_NO_SOURCE_TRACKS", reason="normal_tracks_missing")
                counts["SKIP_NO_SOURCE_TRACKS"] += 1
                report.append(row); processed += 1; continue

            cands = by_key.get((vn, frame, label), [])
            meta, reason, detail = choose(cands, conf, wh, tid, args.conf_tol, args.size_tol, args.ambiguity_margin)

            if meta is None:
                status = "AMBIGUOUS" if reason=="ambiguous" else "UNMATCHED"
                row.update(status=status, reason=reason, detail=detail, candidate_count=len(cands))
                counts[status]+=1
            else:
                row.update(status="MATCHED", reason=reason, detail=detail, candidate_count=len(cands),
                           matched_video_id=meta["video_id"], matched_track_id=meta["track_id"],
                           matched_confidence=meta["confidence"], matched_bbox=meta["bbox"],
                           matched_size=[meta["width"],meta["height"]], timestamp_sec=meta["timestamp_sec"])
                counts["MATCHED"] += 1
                if args.apply:
                    client.set_payload(
                        collection_name=args.collection,
                        payload={
                            "bbox": meta["bbox"],
                            "bbox_space": "frame",
                            "confidence": meta["confidence"],
                            "source_video_id": meta["video_id"],
                            "source_track_id": meta["track_id"],
                            "frame_idx": meta["frame_idx"],
                            "timestamp_sec": meta["timestamp_sec"],
                            "bbox_attach_source": "video_tracks/object/tracks.jsonl",
                            "bbox_attach_version": "v2",
                            "bbox_attach_match": reason,
                        },
                        points=[str(p.id)],
                        wait=True,
                    )
            report.append(row)
            processed += 1
            if processed % 250 == 0:
                print(f"      {processed:,} / matched={counts['MATCHED']:,}")
        if offset is None: break

    eligible = counts["MATCHED"]+counts["UNMATCHED"]+counts["AMBIGUOUS"]
    summary = {
        "collection": args.collection,
        "processed": processed,
        "counts": dict(counts),
        "eligible_normal_points": eligible,
        "eligible_match_rate": counts["MATCHED"]/max(1,eligible),
        "applied": args.apply,
    }
    (out/"summary_v2.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"match_report_v2.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    print("[3/3] complete")
    print("Matched               :", f"{counts['MATCHED']:,}")
    print("Unmatched             :", f"{counts['UNMATCHED']:,}")
    print("Ambiguous             :", f"{counts['AMBIGUOUS']:,}")
    print("Non-Normal skipped    :", f"{counts['SKIP_NO_METADATA_FAMILY']:,}")
    print("Normal tracks missing :", f"{counts['SKIP_NO_SOURCE_TRACKS']:,}")
    print("Eligible match rate   :", f"{summary['eligible_match_rate']*100:.2f}%")
    print("Summary               :", out/"summary_v2.json")
    client.close()

if __name__ == "__main__":
    main()
