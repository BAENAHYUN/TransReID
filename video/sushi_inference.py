#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
sushi_inference_v2.py

Custom local SUSHI inference runner for the prepared forensic person tracks.

Key fixes vs v1:
1) Split detections into <=512-frame windows, matching the MOT17 Private hierarchy.
2) Use the actual HierarchicalGraph API from this SUSHI checkout:
   - update_maps_and_depth(labels)
   - update_maps_and_depth_wo_labels()
   - get_labels()
3) Avoid constructing MOTSceneDataset by using experiment_mode="custom".
4) Person -> SUSHI; non-person -> BoT-SORT bypass.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


REID_ARCH = "fastreid_msmt_BOT_R50_ibn"
REID_DIM = 2048
FRAMES_PER_GRAPH = 512


def box_iou(a, b) -> float:
    """(x1, y1, x2, y2) 두 박스의 IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def link_windows(windows, det_to_long_id: Dict[int, int], max_gap: int = 2, min_iou: float = 0.3):
    """512 프레임 창을 독립 처리한 SUSHI 의 long id 를 창 경계에서 잇는다.

    근거: BoT-SORT 의 raw track(source_track_id) 이 창 경계를 넘어 이어지고(프레임 간격 <= max_gap) 경계 양쪽 박스가 겹치면(IoU >= min_iou)
    같은 사람이다 — 추적기는 끊지 않았고 창만 끊은 것. 같은 경계에서 한 long id 는 한 번만 잇는다(IoU 큰 후보 우선, 나머지는 conflict).
    반환 (remap: 옛 long id -> 새 long id (1부터 등장 순), stats).
    """
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    stats = {"boundaries": max(len(windows) - 1, 0), "candidates": 0, "linked": 0, "rejected_gap": 0, "rejected_iou": 0, "conflicts": 0,
             "max_gap": int(max_gap), "min_iou": float(min_iou)}
    records = []
    for wi in range(len(windows) - 1):
        a = windows[wi][2].sort_values(["frame", "detection_id"])
        b = windows[wi + 1][2].sort_values(["frame", "detection_id"])
        last_a = {int(r.source_track_id): r for r in a.itertuples()}          # 뒤에 오는 행이 덮어씀 -> 마지막 프레임
        first_b = {}
        for r in b.itertuples():
            first_b.setdefault(int(r.source_track_id), r)                     # 처음 나온 행 -> 첫 프레임
        proposals = []
        for tid, ra in last_a.items():
            rb = first_b.get(tid)
            if rb is None:
                continue
            stats["candidates"] += 1
            gap = int(rb.frame) - int(ra.frame)
            rec = {"boundary": wi, "track_id": tid, "frame_a": int(ra.frame), "frame_b": int(rb.frame), "gap": gap,
                   "long_a": int(det_to_long_id[int(ra.detection_id)]), "long_b": int(det_to_long_id[int(rb.detection_id)]), "iou": None, "decision": None}
            records.append(rec)
            if gap < 1 or gap > max_gap:
                stats["rejected_gap"] += 1
                rec["decision"] = "rejected_gap"
                continue
            iou = box_iou((float(ra.bb_left), float(ra.bb_top), float(ra.bb_right), float(ra.bb_bot)),
                          (float(rb.bb_left), float(rb.bb_top), float(rb.bb_right), float(rb.bb_bot)))
            rec["iou"] = round(iou, 4)
            if iou < min_iou:
                stats["rejected_iou"] += 1
                rec["decision"] = "rejected_iou"
                continue
            proposals.append((iou, int(det_to_long_id[int(ra.detection_id)]), int(det_to_long_id[int(rb.detection_id)]), rec))
        used_a, used_b = set(), set()
        used_a_pair, used_b_pair = {}, {}
        for iou, la, lb, rec in sorted(proposals, key=lambda x: (-x[0], x[1], x[2])):
            if (la in used_a and used_a_pair.get(la) != lb) or (lb in used_b and used_b_pair.get(lb) != la):
                stats["conflicts"] += 1
                rec["decision"] = "conflict"
                continue
            if la in used_a:
                rec["decision"] = "duplicate"
                continue                                                      # 같은 쌍을 다른 raw track 이 다시 제안
            rec["decision"] = "linked"
            used_a.add(la)
            used_b.add(lb)
            used_a_pair[la] = lb
            used_b_pair[lb] = la
            union(la, lb)
            stats["linked"] += 1
    remap: Dict[int, int] = {}
    next_id = 1
    root_to_new: Dict[int, int] = {}
    for old in sorted(set(det_to_long_id.values())):
        root = find(old)
        if root not in root_to_new:
            root_to_new[root] = next_id
            next_id += 1
        remap[old] = root_to_new[root]
    stats["long_ids_before"] = len(remap)
    stats["long_ids_after"] = len(root_to_new)
    stats["links"] = records                                                          # 연결 근거 (raw id · 양쪽 프레임 · IoU · 결정)
    stats["remap"] = {int(k): int(v) for k, v in remap.items()}
    return remap, stats



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--sushi-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--det-file", default="forensic_botsort")
    p.add_argument("--device", default="cuda")
    p.add_argument("--link-windows", action="store_true",
                   help="512 프레임 창 경계에서 raw track(BoT-SORT id) 이 이어지고 박스가 겹치면 long id 를 잇는다 (창 독립 처리로 갈라진 같은 사람 복구)")
    p.add_argument("--link-max-gap", type=int, default=2, help="창 경계 양쪽 검출의 최대 프레임 간격")
    p.add_argument("--link-min-iou", type=float, default=0.3, help="창 경계 양쪽 박스의 최소 IoU")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _read_config_default(config_py: Path, name: str, fallback):
    """
    Best-effort extraction of argparse default values from configs/config.py.
    Falls back only if the exact option is not found.
    """
    text = config_py.read_text(encoding="utf-8", errors="ignore")

    # add_argument('--name', ..., default=...)
    patt = rf"add_argument\(\s*['\"]--{re.escape(name)}['\"][\s\S]*?default\s*=\s*([^,\)\n]+)"
    m = re.search(patt, text)
    if m:
        raw = m.group(1).strip()
        try:
            return ast.literal_eval(raw)
        except Exception:
            if raw in ("True", "False"):
                return raw == "True"
            return raw.strip("'\"")

    # parser.set_defaults(name=...)
    patt2 = rf"set_defaults\([\s\S]*?\b{re.escape(name)}\s*=\s*([^,\)\n]+)"
    m = re.search(patt2, text)
    if m:
        raw = m.group(1).strip()
        try:
            return ast.literal_eval(raw)
        except Exception:
            if raw in ("True", "False"):
                return raw == "True"
            return raw.strip("'\"")

    print(f"[WARN] could not read config default for '{name}', fallback={fallback!r}")
    return fallback


def build_config(sushi_root: Path, input_root: Path, checkpoint: Path, device: str):
    cfg_py = sushi_root / "configs" / "config.py"

    connectivity = _read_config_default(cfg_py, "connectivity", "full")
    share_weights = _read_config_default(cfg_py, "share_weights", "all_but_first")
    node_level_embed = _read_config_default(cfg_py, "node_level_embed", False)

    print(f"[INFO] config connectivity={connectivity!r}")
    print(f"[INFO] config share_weights={share_weights!r}")
    print(f"[INFO] config node_level_embed={node_level_embed!r}")

    return SimpleNamespace(
        # IMPORTANT: not "test", otherwise HICLTracker constructs MOTSceneDataset.
        experiment_mode="custom",
        device=device,
        hicl_model_path=str(checkpoint),
        experiment_path=str(input_root / "_sushi_run"),

        # Model
        node_dim=2048,
        hicl_depth=9,
        do_hicl_feats=False,
        hicl_feats_args={},
        share_weights=share_weights,
        edge_level_embed=True,
        node_level_embed=node_level_embed,

        # Hierarchical graph
        connectivity=connectivity,
        node_id_min_ratio=0.5,
        frames_per_graph=512,
        frames_per_level=[2, 4, 8, 16, 32, 64, 128, 256, 512],
        top_k_nns=10,
        symmetric_edges=True,

        # ReID
        reid_arch=REID_ARCH,
        reid_sim_fn="l2",
        edge_sim_fn="l2",
        l2_norm_reid=False,
        reid_embeddings_dir=f"reid_{REID_ARCH}",
        node_embeddings_dir=f"node_{REID_ARCH}",
        zero_nodes=True,

        # Official MOT17 Private style motion settings
        do_motion=True,
        interpolate_motion=True,
        linear_center_only=True,
        motion_max_length=[2, 4, 8, 16, 32, 64, 128, 256],
        motion_pred_length=[2, 4, 8, 16, 32, 64, 128, 256],
        pruning_method=[
            "geometry",
            "motion_005", "motion_005", "motion_005", "motion_005",
            "motion_005", "motion_005", "motion_005", "motion_005"
        ],
        mpn_use_motion=[False, True, True, True, True, True, True, True, True],
        mpn_use_reid_edge=[True] * 9,
        mpn_use_pos_edge=[True] * 9,

        # Projection
        rounding_method="exact",
        solver_backend="pulp",

        # Constructor / unused placeholders
        depth_pretrain_iteration=500,
        gamma=1,
        lr=0.0003,
        weight_decay=0.0001,
        num_batch=1,
        num_workers=0,
        train_dataset_frame_overlap=20,
        augmentation=False,
        no_fp_loss=False,
    )


def split_windows(df: pd.DataFrame, span: int = FRAMES_PER_GRAPH):
    """
    Non-overlapping sparse windows anchored at the earliest remaining detection.
    Every produced graph satisfies max(frame)-min(frame) < 512.
    """
    remaining = df.copy()
    windows = []
    while len(remaining):
        start = int(remaining["frame"].min())
        end_exclusive = start + span
        mask = (remaining["frame"] >= start) & (remaining["frame"] < end_exclusive)
        sub = remaining.loc[mask].copy().sort_values(["frame", "detection_id"]).reset_index(drop=True)
        if len(sub) == 0:
            raise RuntimeError("window splitter produced an empty window")
        windows.append((start, end_exclusive - 1, sub))
        remaining = remaining.loc[~mask].copy()
    return windows


def load_feature_rows(
    input_root: Path,
    det_file: str,
    emb_dir_name: str,
    sub_df: pd.DataFrame,
) -> torch.Tensor:
    root = input_root / "processed_data" / "embeddings" / det_file / emb_dir_name
    frames = sorted(int(x) for x in sub_df["frame"].unique())
    tensors = [torch.load(root / f"{f}.pt", map_location="cpu") for f in frames]
    all_emb = torch.cat(tensors, dim=0)

    wanted_ids = sub_df["detection_id"].astype(int).to_numpy()
    id_col = all_emb[:, 0].int().numpy()
    keep = np.isin(id_col, wanted_ids)
    all_emb = all_emb[keep]

    # Stored tensors were sorted by global detection_id.
    lookup = {int(row[0].item()): row for row in all_emb}
    ordered = []
    for did in wanted_ids:
        did = int(did)
        if did not in lookup:
            raise RuntimeError(f"missing embedding for detection_id={did}")
        ordered.append(lookup[did])
    return torch.stack(ordered, dim=0)


def build_graph(input_root: Path, det_file: str, sub_df: pd.DataFrame, seq_info, config):
    from src.data.graph import HierarchicalGraph

    x_reid_stored = load_feature_rows(
        input_root, det_file, config.reid_embeddings_dir, sub_df
    )
    x_node_stored = load_feature_rows(
        input_root, det_file, config.node_embeddings_dir, sub_df
    )

    expected_ids = sub_df["detection_id"].astype(int).to_numpy()
    if not np.array_equal(x_reid_stored[:, 0].int().numpy(), expected_ids):
        raise RuntimeError("ReID detection_id order mismatch")
    if not np.array_equal(x_node_stored[:, 0].int().numpy(), expected_ids):
        raise RuntimeError("Node detection_id order mismatch")

    x_reid = x_reid_stored[:, 1:].float()
    x_node = x_node_stored[:, 1:].float()

    if x_reid.shape[1] != REID_DIM:
        raise RuntimeError(f"expected ReID dim={REID_DIM}, got {x_reid.shape}")

    x_frame = torch.tensor(sub_df[["frame"]].values, dtype=torch.long)
    x_bbox = torch.tensor(
        sub_df[["bb_left", "bb_top", "bb_width", "bb_height"]].values,
        dtype=torch.float32,
    )
    x_feet = torch.tensor(
        sub_df[["feet_x", "feet_y"]].values,
        dtype=torch.float32,
    )
    y_id = torch.tensor(sub_df[["id"]].values, dtype=torch.long)
    x_center = x_bbox[:, :2] + 0.5 * x_bbox[:, 2:]

    start_frame = int(sub_df["frame"].min())
    end_frame = int(sub_df["frame"].max())

    if end_frame - start_frame >= FRAMES_PER_GRAPH:
        raise RuntimeError(
            f"window too large: {start_frame}..{end_frame} "
            f"(span={end_frame-start_frame+1})"
        )

    fps = int(round(float(seq_info.get("fps", 30))))

    graph = HierarchicalGraph(
        x_reid=x_reid,
        x_node=x_node,
        x_frame=x_frame,
        x_bbox=x_bbox,
        x_feet=x_feet,
        x_center=x_center,
        y_id=y_id,
        fps=torch.tensor(fps).long(),
        frames_total=torch.tensor(config.frames_per_graph).long(),
        frames_per_level=torch.tensor(config.frames_per_level).long(),
        start_frame=torch.tensor(start_frame).long(),
        end_frame=torch.tensor(end_frame).long(),
    )

    return graph


def _run_model(model, curr_graph_batch, depth: int):
    """
    SUSHI commits differ slightly in HICLNet.forward signature.
    Use signature-guided calls, not a hard-coded blind call.
    """
    sig = inspect.signature(model.forward)
    params = [
        p for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    print(f"[DEBUG] HICLNet.forward={sig}")

    if len(params) >= 2:
        result = model(curr_graph_batch, depth)
    else:
        result = model(curr_graph_batch)

    if isinstance(result, tuple) and len(result) and isinstance(result[0], dict):
        result = result[0]

    if not isinstance(result, dict):
        raise RuntimeError(
            f"unexpected HICLNet output type: {type(result)}; "
            f"forward signature={sig}"
        )

    if "classified_edges" not in result:
        raise RuntimeError(
            f"HICLNet output missing 'classified_edges'; keys={list(result.keys())}"
        )

    classified = result["classified_edges"]
    if isinstance(classified, (list, tuple)):
        logits = classified[-1]
    else:
        logits = classified

    return logits.view(-1)


def run_one_window(graph, tracker, config, window_index: int):
    graph.to(config.device)

    print(
        f"[WINDOW {window_index}] "
        f"start={int(graph.start_frame)} end={int(graph.end_frame)} "
        f"detections={int(graph.x_frame.shape[0])}"
    )

    with torch.inference_mode():
        for depth in range(config.hicl_depth):
            # graph.curr_depth is a scalar tensor
            curr_depth = int(graph.curr_depth.item())

            curr_batch, _motion_pred = tracker._hicl_to_curr([graph])

            # No valid edges -> no merges on this hierarchy level.
            if (
                curr_batch is None
                or not hasattr(curr_batch, "edge_index")
                or curr_batch.edge_index is None
                or curr_batch.edge_index.numel() == 0
            ):
                print(f"  depth={curr_depth}: no edges -> keep clusters")
                graph.update_maps_and_depth_wo_labels()
                continue

            logits = _run_model(tracker.model, curr_batch, curr_depth)
            probs = torch.sigmoid(logits)
            curr_batch.edge_preds = (probs >= 0.5).to(curr_batch.edge_preds.dtype)

            tracker._postprocess_graph(
                curr_batch,
                remove_negatives=True,
                decision_threshold=0.5,
            )

            # If positive pruning removed every edge, keep the current clusters.
            if (
                curr_batch.edge_index is None
                or curr_batch.edge_index.numel() == 0
            ):
                print(f"  depth={curr_depth}: no positive edges")
                graph.update_maps_and_depth_wo_labels()
                continue

            projected = tracker._project_graph(curr_batch, decision_threshold=0.5)
            n_components, labels = tracker._assign_labels(projected)

            graph.update_maps_and_depth(labels)

            print(
                f"  depth={curr_depth}: "
                f"nodes={projected.num_nodes} components={n_components}"
            )
            if n_components <= 1:
                print(f"  depth={curr_depth}: single component -> early stop")
                break

    labels = graph.get_labels()
    if labels is None:
        # This should only happen if all update calls failed, but make it explicit.
        raise RuntimeError("HierarchicalGraph.get_labels() returned None")
    return np.asarray(labels).reshape(-1)


def merge_results(original_tracks, person_df, det_to_long_id, method: str = "sushi_mot17private"):
    # key -> detection_id
    keyed = {}
    loose = {}                                   # (frame, source_track_id) -> [(bbox, detection_id)]
    for _, r in person_df.iterrows():
        key = (
            int(r["frame"]),
            int(r["source_track_id"]),
            round(float(r["bb_left"]), 4),
            round(float(r["bb_top"]), 4),
            round(float(r["bb_right"]), 4),
            round(float(r["bb_bot"]), 4),
        )
        keyed[key] = int(r["detection_id"])
        loose.setdefault((int(r["frame"]), int(r["source_track_id"])), []).append((key[2:], int(r["detection_id"])))

    def loose_lookup(frame, tid, bbox, min_iou=0.5):
        """정확한 키가 없을 때(어댑터의 화면 경계 클리핑·반올림 차이) 같은 (frame, track) 의 검출로 대응.
        그 (frame, track) 에 검출이 하나뿐이면 그것(경계에서 잘린 좁은 박스는 IoU 가 크게 떨어질 수 있음), 여럿이면 IoU 최대(min_iou 이상)."""
        cands = loose.get((frame, tid), [])
        if len(cands) == 1:
            return cands[0][1]
        best = None
        for cand_box, did in cands:
            iou = box_iou(cand_box, bbox)
            if iou >= min_iou and (best is None or iou > best[0]):
                best = (iou, did)
        return None if best is None else best[1]

    merged = []
    for row in original_tracks:
        out = dict(row)
        if str(row.get("class_name", "")).lower() == "person":
            x1, y1, x2, y2 = map(float, row["bbox"])
            key = (
                int(row["frame_idx"]),
                int(row["track_id"]),
                round(x1, 4),
                round(y1, 4),
                round(x2, 4),
                round(y2, 4),
            )
            did = keyed.get(key)
            if did is None:
                did = loose_lookup(key[0], key[1], key[2:])
            if did is None:
                raise RuntimeError(f"cannot map person record: {key}")
            out["short_track_id"] = int(row["track_id"])
            out["long_track_id"] = int(det_to_long_id[did])
            out["stitch_method"] = method
        else:
            out["short_track_id"] = int(row["track_id"])
            out["long_track_id"] = int(row["track_id"])
            out["stitch_method"] = "botsort_bypass"
        merged.append(out)
    return merged


def main():
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    sushi_root = Path(args.sushi_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    tracks_path = Path(args.tracks).resolve()
    output_path = Path(args.output).resolve()

    for p in (input_root, sushi_root):
        if not p.exists():
            raise FileNotFoundError(p)
    for p in (checkpoint, tracks_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # SUSHI imports and relative config path expect repo root as CWD.
    sys.path.insert(0, str(sushi_root))

    det_pkl = input_root / "processed_data" / "det" / f"{args.det_file}.pkl"
    # No-person video bypass
    from pathlib import Path as _Path
    import json as _json

    if not _Path(det_pkl).exists():
        _tracks_path = _Path(args.tracks)
        _text = _tracks_path.read_text(encoding="utf-8")
        try:
            _rows = _json.loads(_text)
            if isinstance(_rows, dict):
                _rows = _rows.get("tracks", [])
        except Exception:
            _rows = [_json.loads(line) for line in _text.splitlines() if line.strip()]

        _person_rows = [r for r in _rows if str(r.get("class_name", "")).lower() == "person"]
        if _person_rows:
            raise FileNotFoundError(f"SUSHI det pickle missing despite person tracks: {det_pkl}")

        _out_rows = []
        for _r in _rows:
            _x = dict(_r)
            _tid = _x.get("track_id")
            _x["short_track_id"] = _x.get("short_track_id", _tid)
            _x["long_track_id"] = _x.get("long_track_id", _tid)
            _x["stitch_method"] = "botsort_bypass_no_person"
            _out_rows.append(_x)

        _output_path = _Path(args.output)
        _output_path.parent.mkdir(parents=True, exist_ok=True)
        _output_path.write_text(_json.dumps(_out_rows, indent=2, ensure_ascii=False), encoding="utf-8")

        print("=" * 72)
        print("SUSHI BYPASS: NO PERSON TRACKS")
        print("=" * 72)
        print("person detections : 0")
        print(f"passthrough rows  : {len(_out_rows)}")
        print(f"output            : {_output_path.resolve()}")
        print("=" * 72)
        return

    person_df = pd.read_pickle(det_pkl).sort_values(
        ["frame", "detection_id"]
    ).reset_index(drop=True)

    seq_info = getattr(person_df, "seq_info_dict", {}) or {}
    config = build_config(sushi_root, input_root, checkpoint, args.device)

    windows = split_windows(person_df, FRAMES_PER_GRAPH)

    print("=" * 72)
    print("SUSHI CUSTOM INFERENCE V2")
    print("=" * 72)
    print(f"person detections : {len(person_df)}")
    print(f"source frame range: {int(person_df.frame.min())}..{int(person_df.frame.max())}")
    print(f"graph windows     : {len(windows)}")
    for i, (a, b, sdf) in enumerate(windows, 1):
        print(
            f"  window {i}: allowed={a}..{b}, "
            f"actual={int(sdf.frame.min())}..{int(sdf.frame.max())}, "
            f"detections={len(sdf)}"
        )

    # HICLTracker._get_model opens configs/mpntrack_cfg.yaml using a relative path.
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(sushi_root)

        from src.tracker.hicl_tracker import HICLTracker

        tracker = HICLTracker(
            config=config,
            seqs={"train": {}, "val": {}, "test": {}},
            splits=(None, None, None),
        )

        state = torch.load(checkpoint, map_location=args.device)
        tracker.model.load_state_dict(state)
        tracker.model.eval()

        det_to_long_id = {}
        global_id_offset = 0

        for wi, (_window_start, _window_end, sub_df) in enumerate(windows, 1):
            # A hierarchy requires at least two unique frames to be meaningful.
            if sub_df["frame"].nunique() < 2:
                local_labels = np.arange(len(sub_df), dtype=int)
            else:
                graph = build_graph(
                    input_root,
                    args.det_file,
                    sub_df,
                    seq_info,
                    config,
                )
                local_labels = run_one_window(graph, tracker, config, wi)

            if len(local_labels) != len(sub_df):
                raise RuntimeError(
                    f"window {wi}: labels={len(local_labels)}, detections={len(sub_df)}"
                )

            # Labels are local to each 512-frame graph, so offset them globally.
            unique_local = sorted(set(int(x) for x in local_labels.tolist()))
            remap = {
                lab: global_id_offset + j + 1
                for j, lab in enumerate(unique_local)
            }
            global_id_offset += len(unique_local)

            for did, lab in zip(
                sub_df["detection_id"].astype(int).tolist(),
                local_labels.tolist(),
            ):
                det_to_long_id[int(did)] = int(remap[int(lab)])

    finally:
        import os
        os.chdir(old_cwd)

    link_stats = None
    if args.link_windows:
        remap, link_stats = link_windows(windows, det_to_long_id, args.link_max_gap, args.link_min_iou)
        det_to_long_id = {d: remap.get(l, l) for d, l in det_to_long_id.items()}
        print(f"window link       : {link_stats}")
    original_tracks = read_jsonl(tracks_path)
    merged = merge_results(original_tracks, person_df, det_to_long_id, "sushi_mot17private+winlink" if args.link_windows else "sushi_mot17private")
    if link_stats is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.with_suffix(".link.json").write_text(json.dumps(link_stats, indent=2), encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    long_ids = sorted({
        int(x["long_track_id"])
        for x in merged
        if str(x.get("class_name", "")).lower() == "person"
    })

    print()
    print("=" * 72)
    print("SUSHI MOT17 PRIVATE INFERENCE COMPLETE")
    print("=" * 72)
    print(f"person detections : {len(person_df)}")
    print(f"person long IDs   : {long_ids}")
    print(f"output            : {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
