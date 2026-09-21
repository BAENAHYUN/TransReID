#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Detection, Router
from qdrant_store import QdrantStore


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"
CANDIDATE_ROOT = ROOT / "outputs" / "final_db_candidates"


def parse_args():
    p = argparse.ArgumentParser(
        description="Final DB candidates -> production Qdrant collections"
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--video-stem",
        default=None,
        help="Process one video stem only.",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Process every final_db_candidates.json under outputs/final_db_candidates.",
    )

    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos whose ingest summary already matches the target collections.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--person-collection",
        default="forensic_person",
    )
    p.add_argument(
        "--object-collection",
        default="forensic_object",
    )
    p.add_argument(
        "--recreate-person",
        action="store_true",
        help="DANGEROUS: recreate only the person collection before ingest.",
    )
    p.add_argument(
        "--recreate-object",
        action="store_true",
        help="DANGEROUS: recreate only the object collection before ingest.",
    )
    return p.parse_args()


def load_candidates(video_stem: str):
    path = CANDIDATE_ROOT / video_stem / "final_db_candidates.json"
    if not path.is_file():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("candidates", [])
    if not isinstance(rows, list):
        raise TypeError("candidates must be a list")
    return path, data.get("summary", {}), rows


def stable_id(video_stem: str, row: dict) -> str:
    scope = str(row["candidate_scope"])
    track_id = int(row.get("selected_track_id", row.get("long_track_id", -1)))
    frame_idx = int(row["frame_idx"])
    rank = int(row.get("selected_rank", 0))

    key = f"final-ingest|{video_stem}|{scope}|{track_id}|{frame_idx}|{rank}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def make_collection_configs(
    cfg: PipelineConfig,
    person_collection: str,
    object_collection: str,
):
    person_retrievers = {
        name: spec
        for name, spec in cfg.retrievers.items()
        if spec.accepts_person()
    }
    object_retrievers = {
        name: spec
        for name, spec in cfg.retrievers.items()
        if spec.accepts_object()
    }

    person_cfg = replace(
        cfg,
        collection=person_collection,
        retrievers=person_retrievers,
    )
    object_cfg = replace(
        cfg,
        collection=object_collection,
        retrievers=object_retrievers,
    )
    return person_cfg, object_cfg


def row_to_detection(row: dict, video_stem: str) -> Detection:
    scope = str(row["candidate_scope"]).lower()
    label = "person" if scope == "person" else str(
        row.get("selected_label")
        or row.get("object_track_label")
        or row.get("class_name")
        or "object"
    )

    track_id = int(
        row.get("selected_track_id",
                row.get("long_track_id",
                        row.get("track_id", -1)))
    )

    crop_path = str(Path(row["selected_crop_path"]).resolve())
    det_id = stable_id(video_stem, row)

    # qdrant_store.py reserves core payload keys such as
    # label / frame_idx / bbox. They are supplied by Detection itself,
    # so do NOT duplicate them inside det.extra.
    extra = {
        "media_type": "video",
        "source": "final_db_candidates",
        "video_stem": video_stem,
        "video": Path(row.get("video", video_stem)).name,
        "video_path": row.get("video_path", ""),
        "detection_id": det_id,
        "crop_id": det_id,
        "crop_path": crop_path,
        "timestamp_sec": float(row.get("timestamp_sec") or 0.0),
        "track_id": track_id,
        "long_track_id": int(row.get("long_track_id", track_id)),
        "selected_track_id": track_id,
        "selected_rank": int(row.get("selected_rank", 0)),
        "track_key": f"{video_stem}/{scope}_{track_id}",
        # Explicit final identity payloads for forensic traceability.
        "canonical_person_id": (
            int(row.get("canonical_person_id", track_id))
            if scope == "person" else None
        ),
        "canonical_person_members": (
            row.get("canonical_person_members")
            if scope == "person" else None
        ),
        "merged_object_track_id": (
            int(row.get("merged_object_track_id", track_id))
            if scope == "object" else None
        ),
        "confidence": float(row.get("confidence") or 0.0),
        "quality_score": float(row.get("quality_score") or 0.0),
        "final_person_score": row.get("final_person_score"),
        "object_semantic_score": row.get("object_semantic_score"),
        "object_semantic_decision": row.get("object_semantic_decision"),
        "merged_from_track_ids": row.get("merged_from_track_ids"),
        "candidate_scope": scope,
    }

    return Detection(
        crop=crop_path,
        label=label,
        score=float(row.get("confidence") or 0.0),
        bbox=tuple(float(x) for x in row.get("bbox_clamped", row.get("bbox"))),
        image_id=video_stem,
        frame_idx=int(row["frame_idx"]),
        track_id=track_id,
        extra=extra,
    )


def validate_vectors(dets, vec_maps, cfg):
    person_labels = {str(x).lower() for x in cfg.person_labels}
    dims = {}

    if len(dets) != len(vec_maps):
        raise RuntimeError(
            f"router count mismatch {len(dets)} != {len(vec_maps)}"
        )

    for i, (det, vec_map) in enumerate(zip(dets, vec_maps)):
        scope = str((getattr(det, "extra", None) or {}).get("candidate_scope", "")).lower()

        if scope == "person":
            is_person = True
        elif scope == "object":
            is_person = False
        else:
            is_person = str(det.label).lower() in person_labels

        expected = {
            name
            for name, spec in cfg.retrievers.items()
            if (
                spec.scope == "all"
                or (spec.scope == "person" and is_person)
                or (spec.scope == "object" and not is_person)
            )
        }
        actual = set(vec_map)

        if actual != expected:
            raise RuntimeError(
                f"routing mismatch index={i} label={det.label} "
                f"expected={sorted(expected)} actual={sorted(actual)}"
            )

        for name, vec in vec_map.items():
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)

            if not np.isfinite(arr).all():
                raise RuntimeError(
                    f"{name}: NaN/Inf vector at index={i}"
                )

            spec_dim = int(cfg.retrievers[name].dim)
            if arr.size != spec_dim:
                raise RuntimeError(
                    f"{name}: runtime dim={arr.size}, "
                    f"pipeline.yaml dim={spec_dim}"
                )

            dims[name] = int(arr.size)

    return dims


def make_html(
    out_path: Path,
    video_stem: str,
    candidate_summary: dict,
    ingest_summary: dict,
):
    dims = "".join(
        f'<span class="chip">{html.escape(k)} {v}D</span>'
        for k, v in sorted(ingest_summary["vector_dims"].items())
    )

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TEST Qdrant Ingest Report</title>
<style>
:root{{--bg:#09101d;--panel:#121b2d;--line:#293753;--text:#eef4ff;--muted:#9aabc9;--green:#77eab7}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}
main{{max-width:1200px;margin:auto;padding:34px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:18px}}
.k{{font-size:12px;color:var(--muted);text-transform:uppercase}}
.v{{font-size:28px;font-weight:800;margin-top:5px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:20px}}
.chips{{display:flex;gap:8px;flex-wrap:wrap}}
.chip{{background:#0a111e;border:1px solid var(--line);padding:8px 11px;border-radius:999px}}
.ok{{color:var(--green);font-weight:800}}
code{{background:#080d17;padding:3px 6px;border-radius:6px}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<main>
<h1>Production Qdrant Ingest 쨌 {html.escape(video_stem)}</h1>
<p class="ok">Production collections</p>

<div class="grid">
<div class="card"><div class="k">Person tracks</div><div class="v">{candidate_summary.get("person_tracks",0)}</div></div>
<div class="card"><div class="k">Object tracks</div><div class="v">{candidate_summary.get("object_tracks",0)}</div></div>
<div class="card"><div class="k">Person points</div><div class="v">{ingest_summary["person_upserts"]}</div></div>
<div class="card"><div class="k">Object points</div><div class="v">{ingest_summary["object_upserts"]}</div></div>
</div>

<section>
<h2>Embedding dimensions</h2>
<div class="chips">{dims}</div>
</section>

<section>
<h2>Collections</h2>
<p>Person: <code>{html.escape(ingest_summary["person_collection"])}</code></p>
<p>Object: <code>{html.escape(ingest_summary["object_collection"])}</code></p>
<p>Recreated person: <b>{ingest_summary["recreated_person"]}</b></p>
<p>Recreated object: <b>{ingest_summary["recreated_object"]}</b></p>
</section>

<section>
<h2>Timing</h2>
<p>Total elapsed: <b>{ingest_summary["elapsed_sec"]:.2f}s</b></p>
</section>
</main>
</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")



def discover_video_stems() -> list[str]:
    stems = [
        p.parent.name
        for p in CANDIDATE_ROOT.glob("*/final_db_candidates.json")
        if p.is_file()
    ]
    return sorted(set(stems), key=str.lower)


def resume_summary_matches(
    summary_path: Path,
    person_collection: str,
    object_collection: str,
) -> bool:
    if not summary_path.is_file():
        return False

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        data.get("person_collection") == person_collection
        and data.get("object_collection") == object_collection
    )


def process_video(
    video_stem: str,
    *,
    cfg,
    router,
    person_store,
    object_store,
    person_cfg,
    object_cfg,
    batch_size: int,
):
    started = time.time()

    candidate_path, candidate_summary, rows = load_candidates(video_stem)

    person_rows = [
        r for r in rows
        if str(r.get("candidate_scope", "")).lower() == "person"
    ]
    object_rows = [
        r for r in rows
        if str(r.get("candidate_scope", "")).lower() == "object"
    ]

    if not rows:
        raise RuntimeError("No final DB candidates found")

    print()
    print("-" * 88)
    print("video stem      :", video_stem)
    print("candidate file  :", candidate_path)
    print("person rows     :", len(person_rows))
    print("object rows     :", len(object_rows))
    print("-" * 88)

    person_dets = []
    person_vecs = []
    object_dets = []
    object_vecs = []
    vector_dims = {}

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]

        dets = [
            row_to_detection(r, video_stem)
            for r in batch_rows
        ]

        vec_maps = router.embed(dets)
        dims = validate_vectors(dets, vec_maps, cfg)
        vector_dims.update(dims)

        for det, vec_map, row in zip(dets, vec_maps, batch_rows):
            scope = str(row["candidate_scope"]).lower()

            if scope == "person":
                person_dets.append(det)
                person_vecs.append({
                    k: v
                    for k, v in vec_map.items()
                    if k in person_cfg.retrievers
                })
            else:
                object_dets.append(det)
                object_vecs.append({
                    k: v
                    for k, v in vec_map.items()
                    if k in object_cfg.retrievers
                })

        done = min(start + batch_size, len(rows))
        print(
            f"[EMBED] {done}/{len(rows)} "
            f"| dims={vector_dims}"
        )

    person_upserts = 0
    object_upserts = 0

    if person_dets:
        person_upserts = person_store.upsert(
            person_dets,
            person_vecs,
            batch_size=batch_size,
        )

    if object_dets:
        object_upserts = object_store.upsert(
            object_dets,
            object_vecs,
            batch_size=batch_size,
        )

    out_dir = CANDIDATE_ROOT / video_stem
    summary = {
        "video_stem": video_stem,
        "person_collection": person_cfg.collection,
        "object_collection": object_cfg.collection,
        "person_candidates": len(person_rows),
        "object_candidates": len(object_rows),
        "person_upserts": int(person_upserts),
        "object_upserts": int(object_upserts),
        "vector_dims": vector_dims,
        "recreated_person": False,
        "recreated_object": False,
        "elapsed_sec": round(time.time() - started, 3),
    }

    summary_path = out_dir / "qdrant_ingest_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    html_path = out_dir / "qdrant_ingest_report.html"
    make_html(
        html_path,
        video_stem,
        candidate_summary,
        summary,
    )

    print(
        f"[DONE] {video_stem} "
        f"| person={person_upserts} "
        f"object={object_upserts} "
        f"| {summary['elapsed_sec']:.1f}s"
    )

    return summary


def main():
    args = parse_args()
    total_started = time.time()

    cfg = PipelineConfig.load(CONFIG_PATH)
    person_cfg, object_cfg = make_collection_configs(
        cfg,
        args.person_collection,
        args.object_collection,
    )

    if args.all:
        video_stems = discover_video_stems()
    else:
        video_stems = [args.video_stem]

    if not video_stems:
        raise RuntimeError(
            f"No final_db_candidates.json found under {CANDIDATE_ROOT}"
        )

    print("=" * 88)
    print("PRODUCTION QDRANT BULK INGEST")
    print("=" * 88)
    print("videos          :", len(video_stems))
    print("person coll     :", person_cfg.collection)
    print("object coll     :", object_cfg.collection)
    print("batch size      :", args.batch_size)
    print("resume          :", args.resume)
    print("recreate person :", args.recreate_person)
    print("recreate object :", args.recreate_object)
    print("=" * 88)

    person_store = QdrantStore(person_cfg)
    object_store = QdrantStore(object_cfg)

    # Recreate, when explicitly requested, happens only once.
    person_store.ensure_collection(recreate=args.recreate_person)
    object_store.ensure_collection(recreate=args.recreate_object)

    # Critical speed improvement:
    # load all embedders once and reuse them for every video.
    registry = EmbedderRegistry(cfg)
    router = Router(cfg, registry, input_format="rgb")

    completed = 0
    skipped = 0
    failed = 0
    total_person_upserts = 0
    total_object_upserts = 0

    try:
        total = len(video_stems)

        for index, video_stem in enumerate(video_stems, start=1):
            out_dir = CANDIDATE_ROOT / video_stem
            summary_path = out_dir / "qdrant_ingest_summary.json"

            if (
                args.resume
                and resume_summary_matches(
                    summary_path,
                    person_cfg.collection,
                    object_cfg.collection,
                )
            ):
                skipped += 1
                print(f"[{index}/{total}] SKIP {video_stem}")
                continue

            print()
            print(f"[{index}/{total}] START {video_stem}")

            try:
                summary = process_video(
                    video_stem,
                    cfg=cfg,
                    router=router,
                    person_store=person_store,
                    object_store=object_store,
                    person_cfg=person_cfg,
                    object_cfg=object_cfg,
                    batch_size=args.batch_size,
                )

                completed += 1
                total_person_upserts += int(summary["person_upserts"])
                total_object_upserts += int(summary["object_upserts"])

            except Exception as exc:
                failed += 1
                print(
                    f"[FAILED] {video_stem}: "
                    f"{type(exc).__name__}: {exc}"
                )

        elapsed = round(time.time() - total_started, 3)

        print()
        print("=" * 88)
        print("BULK INGEST COMPLETE")
        print("=" * 88)
        print("candidate videos :", total)
        print("completed        :", completed)
        print("skipped          :", skipped)
        print("failed           :", failed)
        print("person upserts   :", total_person_upserts)
        print("object upserts   :", total_object_upserts)
        print("elapsed_sec      :", elapsed)
        print("=" * 88)

    finally:
        try:
            registry.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()

