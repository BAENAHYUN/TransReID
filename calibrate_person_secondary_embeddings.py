from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"
PERSON_ROOT = ROOT / "data" / "video_tracks" / "person"
OUT_ROOT = (
    ROOT / "data" / "validation" / "person_secondary_embedding_calibration"
)


def normalize(v) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n <= 0:
        return x
    return x / n


def cosine(a, b) -> float:
    aa = normalize(a)
    bb = normalize(b)
    return float(np.dot(aa, bb))


def safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"JSONL parse error: {path}:{line_no}: {exc}"
                ) from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def frame_idx(row: dict) -> int:
    for key in ("frame_idx", "frame_number", "frame"):
        if row.get(key) is not None:
            return safe_int(row.get(key), -1)
    return -1


def timestamp(row: dict) -> float:
    for key in ("timestamp_sec", "time_sec"):
        if row.get(key) is not None:
            return safe_float(row.get(key), 0.0)
    return 0.0


def resolve_crop_path(row: dict) -> Path | None:
    raw = row.get("crop_path") or row.get("path")
    if not raw:
        return None

    p = Path(str(raw))
    if not p.is_absolute():
        p = ROOT / p

    return p


def evenly_sample(rows: List[dict], k: int) -> List[dict]:
    if len(rows) <= k:
        return rows[:]

    if k <= 1:
        return [rows[len(rows) // 2]]

    idxs = []
    for i in range(k):
        idx = round(i * (len(rows) - 1) / (k - 1))
        if idx not in idxs:
            idxs.append(idx)

    return [rows[i] for i in idxs]


def build_track_rows(original_rows: List[dict]) -> Dict[int, List[dict]]:
    out = defaultdict(list)

    for row in original_rows:
        tid = safe_int(row.get("track_id"), -1)
        if tid >= 0:
            out[tid].append(row)

    for tid in out:
        out[tid].sort(key=lambda r: (timestamp(r), frame_idx(r)))

    return dict(out)


def actual_merge_edges(stitched_rows: List[dict]) -> List[dict]:
    """
    stitched metadata repeats for every frame row.
    Keep one metadata row per original_track_id.
    """
    one_per_track = {}

    for row in stitched_rows:
        tid = safe_int(row.get("original_track_id"), -1)
        if tid < 0:
            continue
        one_per_track.setdefault(tid, row)

    edges = []

    for tid, row in sorted(one_per_track.items()):
        predecessor = row.get("stitched_from_track_id")
        rule = row.get("stitch_rule")

        if predecessor is None or not rule:
            continue

        edges.append({
            "person_id": safe_int(row.get("person_id"), -1),
            "track_id": tid,
            "predecessor_track_id": safe_int(predecessor, -1),
            "rule": str(rule),
            "gap_sec": row.get("stitch_gap_sec"),
            "solider_global": row.get("stitch_global_similarity"),
            "solider_fused": row.get("stitch_fused_score"),
            "clothing_similarity": row.get("stitch_clothing_similarity"),
            "clothing_upper_min": row.get(
                "stitch_clothing_upper_min_similarity"
            ),
        })

    return edges


def embed_track(
    router: Router,
    track_id: int,
    rows: List[dict],
    *,
    sample_count: int,
) -> dict[str, np.ndarray]:
    selected = evenly_sample(rows, sample_count)

    vectors = defaultdict(list)
    used = 0

    for row in selected:
        p = resolve_crop_path(row)
        if p is None or not p.exists():
            continue

        # Existing project Router:
        # person scope returns SigLIP2 + IRRA + SOLIDER.
        vector_map = router.embed_query_image(
            str(p),
            scope="person",
        )

        for name in ("siglip2", "irra", "solider"):
            if name in vector_map:
                vectors[name].append(
                    normalize(vector_map[name])
                )

        used += 1

    if used == 0:
        raise RuntimeError(
            f"track {track_id}: no readable representative crops"
        )

    out = {}

    for name, vals in vectors.items():
        if not vals:
            continue
        mean = np.mean(np.stack(vals, axis=0), axis=0)
        out[name] = normalize(mean)

    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Calibrate independent person appearance embeddings on visually "
            "reviewed V4.5 merge edges. No stitching output is modified."
        )
    )
    ap.add_argument(
        "--video",
        default="Normal_Videos_439_x264",
    )
    ap.add_argument(
        "--true-person-ids",
        nargs="+",
        type=int,
        default=[5, 10, 14, 16, 17, 18],
        help="visually confirmed normal merged identity IDs",
    )
    ap.add_argument(
        "--false-person-ids",
        nargs="+",
        type=int,
        default=[19],
        help="visually confirmed false-merge identity IDs",
    )
    ap.add_argument(
        "--samples-per-track",
        type=int,
        default=5,
    )
    args = ap.parse_args()

    video_dir = PERSON_ROOT / args.video
    original_path = video_dir / "tracks.jsonl"
    stitched_path = video_dir / "stitched_tracks_v4_5.jsonl"

    if not original_path.exists():
        raise FileNotFoundError(original_path)

    if not stitched_path.exists():
        raise FileNotFoundError(stitched_path)

    original = load_jsonl(original_path)
    stitched = load_jsonl(stitched_path)

    track_rows = build_track_rows(original)
    edges = actual_merge_edges(stitched)

    true_ids = set(args.true_person_ids)
    false_ids = set(args.false_person_ids)
    reviewed_ids = true_ids | false_ids

    edges = [
        e for e in edges
        if e["person_id"] in reviewed_ids
    ]

    if not edges:
        raise RuntimeError(
            "No reviewed final merge edges found for requested person IDs."
        )

    needed_tracks = set()
    for e in edges:
        needed_tracks.add(e["track_id"])
        needed_tracks.add(e["predecessor_track_id"])

    print("=" * 100)
    print("PERSON SECONDARY EMBEDDING CALIBRATION")
    print("=" * 100)
    print("video              :", args.video)
    print("reviewed merge edges:", len(edges))
    print("unique tracks      :", len(needed_tracks))
    print("samples per track  :", args.samples_per_track)
    print("true person IDs    :", sorted(true_ids))
    print("false person IDs   :", sorted(false_ids))
    print()
    print("Loading existing Router models...")
    print("This computes only the small visually-reviewed subset.")

    cfg = PipelineConfig.load(CONFIG_PATH)
    registry = EmbedderRegistry(cfg)
    router = Router(cfg, registry, input_format="rgb")

    track_embeddings = {}

    for idx, tid in enumerate(sorted(needed_tracks), 1):
        if tid not in track_rows:
            raise RuntimeError(f"track {tid} missing from tracks.jsonl")

        print(f"[{idx}/{len(needed_tracks)}] embedding track {tid}")
        track_embeddings[tid] = embed_track(
            router,
            tid,
            track_rows[tid],
            sample_count=args.samples_per_track,
        )

    report = []

    for e in edges:
        a = e["predecessor_track_id"]
        b = e["track_id"]

        va = track_embeddings[a]
        vb = track_embeddings[b]

        label = (
            "FALSE"
            if e["person_id"] in false_ids
            else "TRUE"
        )

        row = {
            "label": label,
            "person_id": e["person_id"],
            "predecessor_track_id": a,
            "track_id": b,
            "rule": e["rule"],
            "gap_sec": e["gap_sec"],
            "solider_stitch_global": e["solider_global"],
            "solider_stitch_fused": e["solider_fused"],
            "hsv_clothing_similarity": e["clothing_similarity"],
            "hsv_clothing_upper_min": e["clothing_upper_min"],
            "irra_similarity": (
                cosine(va["irra"], vb["irra"])
                if "irra" in va and "irra" in vb
                else None
            ),
            "siglip2_similarity": (
                cosine(va["siglip2"], vb["siglip2"])
                if "siglip2" in va and "siglip2" in vb
                else None
            ),
            "solider_recomputed_similarity": (
                cosine(va["solider"], vb["solider"])
                if "solider" in va and "solider" in vb
                else None
            ),
        }

        report.append(row)

    # Sort so the false edge is easy to compare against true edges.
    report.sort(
        key=lambda r: (
            0 if r["label"] == "FALSE" else 1,
            r["person_id"],
            r["track_id"],
        )
    )

    out_dir = OUT_ROOT / args.video
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "reviewed_merge_embedding_scores.csv"

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        fields = [
            "label",
            "person_id",
            "predecessor_track_id",
            "track_id",
            "rule",
            "gap_sec",
            "solider_stitch_global",
            "solider_stitch_fused",
            "hsv_clothing_similarity",
            "hsv_clothing_upper_min",
            "irra_similarity",
            "siglip2_similarity",
            "solider_recomputed_similarity",
        ]

        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(report)

    print()
    print("=" * 100)
    print("REVIEWED FINAL MERGE EDGES")
    print("=" * 100)

    header = (
        f"{'GT':5s} {'EDGE':11s} "
        f"{'IRRA':>8s} {'SIGLIP2':>8s} "
        f"{'SOLIDER':>8s} {'HSV':>8s} {'GAP':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in report:
        edge = (
            f"T{r['predecessor_track_id']}"
            f"->T{r['track_id']}"
        )

        def fv(x):
            return "   N/A  " if x is None else f"{float(x):8.4f}"

        print(
            f"{r['label']:5s} {edge:11s} "
            f"{fv(r['irra_similarity'])} "
            f"{fv(r['siglip2_similarity'])} "
            f"{fv(r['solider_recomputed_similarity'])} "
            f"{fv(r['hsv_clothing_similarity'])} "
            f"{safe_float(r['gap_sec']):6.1f}"
        )

    print()
    print("output:", csv_path)
    print()
    print("Interpretation:")
    print("  - Do NOT set a new gate yet.")
    print("  - Compare the FALSE T27->T36 row with the TRUE rows.")
    print("  - A useful secondary model should place the false edge clearly")
    print("    below most/all visually confirmed true edges.")
    print("  - If IRRA/SigLIP2 also overlap heavily, keep V4.4's conservative")
    print("    split policy rather than forcing more automatic merges.")


if __name__ == "__main__":
    main()
