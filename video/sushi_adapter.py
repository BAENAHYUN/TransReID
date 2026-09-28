#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sushi_adapter.py

Convert this project's BoT-SORT tracks.jsonl + source video into
SUSHI-compatible person detection / FastReID embedding artifacts.

- person: source video bbox crop -> official SUSHI FastReID
  (fastreid_msmt_BOT_R50_ibn) -> 2048-D features -> frame-wise .pt
- non-person objects: bypass SUSHI and preserve original BoT-SORT tracks

This script prepares SUSHI input artifacts only. It does NOT run
mot17private.pth inference yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

REID_ARCH = "fastreid_msmt_BOT_R50_ibn"
REID_DIM = 2048
PERSON_CLASS_NAME = "person"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--tracklets", default=None)
    p.add_argument("--sushi-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--det-file", default="forensic_botsort")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--person-class", default=PERSON_CLASS_NAME)
    p.add_argument("--save-crops", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_track_row(row: Dict[str, Any], idx: int) -> None:
    required = {"frame_idx", "track_id", "bbox", "confidence", "class_id", "class_name", "timestamp_sec"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"track row #{idx} missing fields: {sorted(missing)}")
    if not isinstance(row["bbox"], (list, tuple)) or len(row["bbox"]) != 4:
        raise ValueError(f"track row #{idx} invalid bbox: {row['bbox']!r}")


def clamp_bbox(bbox: Iterable[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, bbox)
    x1i = max(0, min(width - 1, int(np.floor(x1))))
    y1i = max(0, min(height - 1, int(np.floor(y1))))
    x2i = max(0, min(width, int(np.ceil(x2))))
    y2i = max(0, min(height, int(np.ceil(y2))))
    if x2i <= x1i or y2i <= y1i:
        raise ValueError(f"Degenerate bbox after clamp: {bbox}")
    return x1i, y1i, x2i, y2i


def load_fastreid(sushi_root: Path, device: str):
    sushi_root = sushi_root.resolve()
    sys.path.insert(0, str(sushi_root))
    sys.path.insert(0, str(sushi_root / "fast-reid"))
    from src.models.reid.fastreid_models import load_fastreid_model
    model, transforms = load_fastreid_model(REID_ARCH)
    model = model.to(device).eval()
    return model, transforms


def apply_transform(transform, crop_bgr: np.ndarray) -> torch.Tensor:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    out = transform(Image.fromarray(crop_rgb))
    if isinstance(out, dict):
        out = out.get("image")
    if isinstance(out, np.ndarray):
        out = torch.from_numpy(out)
    if not torch.is_tensor(out) or out.ndim != 3:
        raise TypeError(f"Unexpected FastReID transform output: {type(out)}, shape={getattr(out, 'shape', None)}")
    return out


@torch.inference_mode()
def infer_batch(model, tensors: List[torch.Tensor], device: str) -> torch.Tensor:
    x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
    y = model(x)
    if not torch.is_tensor(y):
        raise TypeError(f"Expected Tensor from FastReID, got {type(y)}")
    if y.ndim != 2 or y.shape[1] != REID_DIM:
        raise ValueError(f"Expected [N,{REID_DIM}], got {tuple(y.shape)}")
    return y.detach().float().cpu()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    tracks_path = Path(args.tracks).resolve()
    sushi_root = Path(args.sushi_root).resolve()
    output_root = Path(args.output_root).resolve()

    for p in (video_path, tracks_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    if not sushi_root.is_dir():
        raise FileNotFoundError(sushi_root)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows = read_jsonl(tracks_path)
    if not rows:
        raise RuntimeError("tracks.jsonl is empty")
    for i, row in enumerate(rows):
        validate_track_row(row, i)

    person_rows = [r for r in rows if str(r["class_name"]).strip().lower() == args.person_class.lower()]
    object_rows = [r for r in rows if str(r["class_name"]).strip().lower() != args.person_class.lower()]
    person_rows.sort(key=lambda r: (int(r["frame_idx"]), int(r["track_id"]), float(r["bbox"][0]), float(r["bbox"][1])))

    if args.tracklets:
        tracklets_path = Path(args.tracklets).resolve()
        with tracklets_path.open("r", encoding="utf-8") as f:
            tracklets = json.load(f)
        if isinstance(tracklets, dict):
            tracklets = tracklets.get("tracklets", [])
        tracklet_ids = {int(t["track_id"]) for t in tracklets if str(t.get("class_name", "")).lower() == args.person_class.lower()}
        row_ids = {int(r["track_id"]) for r in person_rows}
        if tracklet_ids and tracklet_ids != row_ids:
            print(f"[WARN] track-id mismatch: tracks-only={sorted(row_ids-tracklet_ids)}, tracklets-only={sorted(tracklet_ids-row_ids)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    seq_name = video_path.stem
    seq_root = output_root / seq_name
    processed_root = seq_root / "processed_data"
    det_dir = processed_root / "det"
    emb_base = processed_root / "embeddings" / args.det_file
    reid_dir = emb_base / f"reid_{REID_ARCH}"
    node_dir = emb_base / f"node_{REID_ARCH}"
    crops_dir = seq_root / "person_crops"
    for d in (det_dir, reid_dir, node_dir):
        d.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(seq_root / "object_tracks.jsonl", object_rows)

    if not person_rows:
        cap.release()
        (seq_root / "manifest.json").write_text(json.dumps({
            "video": str(video_path), "tracks": str(tracks_path), "sequence": seq_name,
            "person_detections": 0, "object_bypass_records": len(object_rows),
            "reid_arch": REID_ARCH, "reid_dim": REID_DIM
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[DONE] no person tracks")
        return

    print(f"[INFO] loading FastReID: {REID_ARCH}")
    model, transforms = load_fastreid(sushi_root, args.device)

    records = []
    for detection_id, r in enumerate(person_rows):
        x1, y1, x2, y2 = map(float, r["bbox"])
        w, h = x2 - x1, y2 - y1
        records.append({
            "detection_id": detection_id,
            "frame": int(r["frame_idx"]),
            "id": int(r["track_id"]),
            "bb_left": x1,
            "bb_top": y1,
            "bb_width": w,
            "bb_height": h,
            "bb_right": x2,
            "bb_bot": y2,
            "feet_x": x1 + 0.5 * w,
            "feet_y": y1 + h,
            "confidence": float(r["confidence"]),
            "class_id": int(r["class_id"]),
            "class_name": str(r["class_name"]),
            "timestamp_sec": float(r["timestamp_sec"]),
            "source_track_id": int(r["track_id"]),
        })

    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_frame[int(rec["frame"])].append(rec)
    requested_frames = sorted(by_frame)
    max_requested = requested_frames[-1]
    if frame_count > 0 and max_requested >= frame_count:
        cap.release()
        raise ValueError(f"frame_idx={max_requested} but video has {frame_count} frames")

    pending_tensors: List[torch.Tensor] = []
    pending_meta: List[Tuple[int, int]] = []
    features_by_frame: Dict[int, List[Tuple[int, torch.Tensor]]] = defaultdict(list)

    def flush_batch() -> None:
        nonlocal pending_tensors, pending_meta
        if not pending_tensors:
            return
        feats = infer_batch(model, pending_tensors, args.device)
        for (frame_idx, det_id), feat in zip(pending_meta, feats):
            features_by_frame[frame_idx].append((det_id, feat))
        pending_tensors = []
        pending_meta = []

    requested_set = set(requested_frames)
    current = 0
    while current <= max_requested:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Video ended at frame {current}; max requested={max_requested}")
        if current in requested_set:
            for rec in by_frame[current]:
                x1, y1, x2, y2 = clamp_bbox([rec["bb_left"], rec["bb_top"], rec["bb_right"], rec["bb_bot"]], frame_width, frame_height)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    raise RuntimeError(f"Empty crop frame={current}, detection_id={rec['detection_id']}")
                if args.save_crops:
                    cv2.imwrite(str(crops_dir / f"f{current:08d}_d{rec['detection_id']:08d}_t{rec['source_track_id']:06d}.jpg"), crop)
                pending_tensors.append(apply_transform(transforms, crop))
                pending_meta.append((current, int(rec["detection_id"])))
                if len(pending_tensors) >= args.batch_size:
                    flush_batch()
        current += 1
    cap.release()
    flush_batch()

    total_features = sum(len(v) for v in features_by_frame.values())
    if total_features != len(records):
        raise RuntimeError(f"Feature count mismatch: detections={len(records)}, features={total_features}")

    for frame_idx in requested_frames:
        pairs = sorted(features_by_frame[frame_idx], key=lambda x: x[0])
        ids = torch.tensor([[float(det_id)] for det_id, _ in pairs], dtype=torch.float32)
        feats = torch.stack([feat for _, feat in pairs], dim=0).float()
        tensor = torch.cat([ids, feats], dim=1)
        if tensor.shape[1] != 1 + REID_DIM:
            raise RuntimeError(f"Bad tensor shape at frame {frame_idx}: {tuple(tensor.shape)}")
        torch.save(tensor, reid_dir / f"{frame_idx}.pt")
        torch.save(tensor.clone(), node_dir / f"{frame_idx}.pt")

    sys.path.insert(0, str(sushi_root))
    from src.data.seq_processor import DataFrameWSeqInfo
    det_df = DataFrameWSeqInfo(pd.DataFrame.from_records(records))
    det_df.sort_values(["frame", "detection_id"], inplace=True)
    det_df.reset_index(drop=True, inplace=True)
    det_df.seq_info_dict = {
        "seq_name": seq_name,
        "seq_path": str(seq_root),
        "fps": int(round(fps)),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "seq_len": frame_count,
        "has_gt": False,
        "is_gt": False,
        "det_file_name": args.det_file,
        "source_video": str(video_path),
        "frame_index_origin": 0,
    }

    det_pkl = det_dir / f"{args.det_file}.pkl"
    det_df.to_pickle(det_pkl)
    det_df.to_csv(det_dir / f"{args.det_file}.csv", index=False)
    write_jsonl(seq_root / "person_tracks.jsonl", person_rows)

    manifest = {
        "video": str(video_path),
        "tracks": str(tracks_path),
        "tracklets": str(Path(args.tracklets).resolve()) if args.tracklets else None,
        "sequence": seq_name,
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frame_count": frame_count,
        "frame_index_origin": 0,
        "person_detections": len(records),
        "person_source_track_ids": sorted({int(r["source_track_id"]) for r in records}),
        "object_bypass_records": len(object_rows),
        "reid_arch": REID_ARCH,
        "reid_dim": REID_DIM,
        "det_file": args.det_file,
        "det_pickle": str(det_pkl),
        "reid_embeddings_dir": str(reid_dir),
        "node_embeddings_dir": str(node_dir),
        "object_bypass_file": str(seq_root / "object_tracks.jsonl"),
        "sushi_checkpoint_expected": str(sushi_root / "pretrained_models" / "mot17private.pth"),
        "notes": [
            "Only person detections are prepared for SUSHI MOT17 Private.",
            "Non-person object tracks bypass SUSHI and retain BoT-SORT IDs.",
            "FastReID features are 2048-D using SUSHI's official MSMT17 BoT R50-IBN loader.",
            "Frame numbering preserves the project's original 0-based frame_idx.",
            "This adapter prepares artifacts only; it does not execute HICL inference."
        ],
    }
    (seq_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SUSHI INPUT ADAPTER COMPLETE")
    print("=" * 72)
    print(f"sequence              : {seq_name}")
    print(f"person detections     : {len(records)}")
    print(f"person short track IDs: {manifest['person_source_track_ids']}")
    print(f"object bypass records : {len(object_rows)}")
    print(f"FastReID              : {REID_ARCH}")
    print(f"FastReID dim          : {REID_DIM}")
    print(f"det dataframe         : {det_pkl}")
    print(f"reid frame tensors    : {reid_dir}")
    print(f"node frame tensors    : {node_dir}")
    print(f"manifest              : {seq_root / 'manifest.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
