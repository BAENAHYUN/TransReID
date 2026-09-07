from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Detection, Router
from qdrant_store import QdrantStore


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"

VIDEO_ROOT = ROOT / "data" / "videos"
PERSON_TRACK_ROOT = ROOT / "data" / "video_tracks" / "person"
OBJECT_TRACK_ROOT = ROOT / "data" / "video_tracks" / "object"
STATE_DIR = ROOT / "data" / "video_embedding_checkpoint_v2"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

PERSON_STITCH_FILES = (
    "stitched_tracks_v4_1.jsonl",
    "stitched_tracks_v4.jsonl",
    "stitched_tracks_v3.jsonl",
    "stitched_tracks_v2.jsonl",
    "stitched_tracks.jsonl",
)
OBJECT_STITCH_FILES = (
    "stitched_tracks_v4_2.jsonl",
    "stitched_tracks_v4_1.jsonl",
    "stitched_tracks_v4.jsonl",
    "stitched_tracks_v3.jsonl",
    "stitched_tracks_v2.jsonl",
    "stitched_tracks.jsonl",
)
PERSON_CACHE_FILES = (
    "stitch_embeddings_solider_v4_1.npz",
    "stitch_embeddings_solider_v4.npz",
    "stitch_embeddings_solider_v3.npz",
    "stitch_embeddings_solider_v2.npz",
    "stitch_embeddings_solider.npz",
)
OBJECT_CACHE_FILES = (
    "stitch_embeddings_dinov2_v4_2.npz",
    "stitch_embeddings_dinov2_v4_1.npz",
    "stitch_embeddings_dinov2_v4.npz",
    "stitch_embeddings_dinov2_v3.npz",
    "stitch_embeddings_dinov2_v2.npz",
    "stitch_embeddings_dinov2.npz",
)


def first_existing(directory: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        p = directory / name
        if p.exists() and p.is_file():
            return p
    return None


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL: {path} line={line_no}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row must be object: {path} line={line_no}")
            rows.append(row)
    return rows


def build_video_index() -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not VIDEO_ROOT.exists():
        return index
    for p in sorted(VIDEO_ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        if p.stem in index and index[p.stem].resolve() != p.resolve():
            raise RuntimeError(
                "Duplicate video stem detected:\n"
                f"  stem={p.stem}\n  1={index[p.stem]}\n  2={p}"
            )
        index[p.stem] = p
    return index


def make_collection_configs(cfg: PipelineConfig):
    person_retrievers = {
        name: spec for name, spec in cfg.retrievers.items() if spec.accepts_person()
    }
    object_retrievers = {
        name: spec for name, spec in cfg.retrievers.items() if spec.accepts_object()
    }
    return (
        replace(cfg, collection="forensic_person", retrievers=person_retrievers),
        replace(cfg, collection="forensic_object", retrievers=object_retrievers),
    )


def identity_id(kind: str, row: dict) -> Optional[int]:
    value = row.get("person_id" if kind == "person" else "object_id")
    return None if value is None else int(value)


def original_track_id(row: dict) -> Optional[int]:
    value = row.get("original_track_id")
    if value is None:
        value = row.get("track_id")
    return None if value is None else int(value)


def parse_bbox(row: dict) -> Tuple[float, float, float, float]:
    value = row.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"bbox must contain 4 values, got: {value!r}")
    bbox = tuple(float(x) for x in value)
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid bbox: {bbox}")
    return bbox


def parse_crop_path(row: dict) -> Path:
    value = row.get("crop_path") or row.get("path")
    if not value:
        raise ValueError("stitched row has no crop_path/path")
    p = Path(str(value))
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"crop not found: {p}")
    return p


def parse_timestamp_sec(row: dict) -> float:
    value = row.get("timestamp_sec")
    if value is None:
        value = row.get("time_sec")
    return 0.0 if value is None else float(value)


def mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minute = int(seconds // 60)
    second = seconds - minute * 60
    return f"{minute:02d}:{second:04.1f}"


def stable_detection_id(
    kind: str,
    video_stem: str,
    stitched_identity_id: int,
    original_tid: Optional[int],
    frame_idx: int,
    bbox: Tuple[float, float, float, float],
    crop_path: Path,
) -> str:
    bbox_key = ",".join(f"{float(x):.3f}" for x in bbox)
    key = (
        f"video|current|{kind}|video={video_stem}|identity={stitched_identity_id}|"
        f"original_track={original_tid}|frame={frame_idx}|bbox={bbox_key}|"
        f"crop={crop_path.name}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def make_record(
    kind: str,
    video_dir: Path,
    source_jsonl: Path,
    cache_path: Optional[Path],
    row: dict,
    video_index: Dict[str, Path],
) -> dict:
    iid = identity_id(kind, row)
    if iid is None:
        raise ValueError(
            f"{source_jsonl}: stitched row has no "
            f"{'person_id' if kind == 'person' else 'object_id'}"
        )

    orig_tid = original_track_id(row)
    frame_idx = int(row.get("frame_idx", 0))
    bbox = parse_bbox(row)
    crop_path = parse_crop_path(row)
    timestamp_sec = parse_timestamp_sec(row)

    video_stem = video_dir.name
    video_path = video_index.get(video_stem)

    if kind == "person":
        label = "person"
        stitched_id = str(row.get("stitched_id") or f"person_{iid:04d}")
    else:
        label = str(
            row.get("final_class_name")
            or row.get("class_name")
            or row.get("raw_class_name")
            or "unknown"
        )
        stitched_id = str(row.get("stitched_id") or f"object_{iid:04d}")

    det_id = stable_detection_id(
        kind,
        video_stem,
        iid,
        orig_tid,
        frame_idx,
        bbox,
        crop_path,
    )

    return {
        "kind": kind,
        "video_stem": video_stem,
        "video_name": video_path.name if video_path is not None else video_stem,
        "video_path": str(video_path) if video_path is not None else "",
        "source_jsonl": str(source_jsonl),
        "cache_path": str(cache_path) if cache_path is not None else "",
        "identity_id": iid,
        "stitched_id": stitched_id,
        "original_track_id": orig_tid,
        "frame_idx": frame_idx,
        "timestamp_sec": timestamp_sec,
        "bbox": bbox,
        "crop_path": crop_path,
        "label": label,
        "confidence": float(row.get("confidence", row.get("score", 1.0))),
        "detection_id": det_id,
        "stitch_model": row.get("stitch_model"),
        "stitch_rule": row.get("stitch_rule"),
        "stitched_from_track_id": row.get("stitched_from_track_id"),
        "stitch_gap_sec": row.get("stitch_gap_sec"),
        "stitch_fused_score": row.get("stitch_fused_score"),
        "stitch_global_similarity": row.get("stitch_global_similarity"),
        "stitch_gallery_global_topk_mean": row.get("stitch_gallery_global_topk_mean"),
        "raw_class_name": row.get("raw_class_name"),
        "class_name": row.get("class_name"),
        "final_class_name": row.get("final_class_name"),
    }


def discover_kind_records(
    kind: str,
    video_index: Dict[str, Path],
    max_videos: Optional[int],
) -> List[dict]:
    if kind == "person":
        root = PERSON_TRACK_ROOT
        stitch_names = PERSON_STITCH_FILES
        cache_names = PERSON_CACHE_FILES
    elif kind == "object":
        root = OBJECT_TRACK_ROOT
        stitch_names = OBJECT_STITCH_FILES
        cache_names = OBJECT_CACHE_FILES
    else:
        raise ValueError(kind)

    if not root.exists():
        return []

    selected = []
    for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stitched = first_existing(video_dir, stitch_names)
        if stitched is None:
            continue
        cache = first_existing(video_dir, cache_names)
        selected.append((video_dir, stitched, cache))

    if max_videos is not None:
        selected = selected[:max_videos]

    records: List[dict] = []
    for video_dir, stitched, cache in selected:
        for row in load_jsonl(stitched):
            try:
                records.append(
                    make_record(kind, video_dir, stitched, cache, row, video_index)
                )
            except Exception as exc:
                raise RuntimeError(
                    "failed to convert stitched row\n"
                    f"  kind={kind}\n  source={stitched}\n  row={row}"
                ) from exc
    return records


def record_to_detection(record: dict) -> Detection:
    kind = record["kind"]
    iid = int(record["identity_id"])
    identity_payload = {"person_id": iid} if kind == "person" else {"object_id": iid}
    t = float(record["timestamp_sec"])

    extra = {
        "media_type": "video",
        "detection_id": record["detection_id"],
        "crop_id": record["detection_id"],
        "crop_path": str(record["crop_path"]),
        "bbox_space": "frame",
        "video": record["video_name"],
        "video_name": record["video_name"],
        "video_path": record["video_path"],
        "frame_number": int(record["frame_idx"]),
        "timestamp_sec": t,
        "time_sec": t,
        "time_mmss": mmss(t),
        **identity_payload,
        "stitched_id": record["stitched_id"],
        "original_track_id": record["original_track_id"],
        "track_key": f"{record['video_stem']}/{record['stitched_id']}",
        "source": "video_tracks_stitched",
        "source_jsonl": record["source_jsonl"],
        "stitch_model": record["stitch_model"],
        "stitch_rule": record["stitch_rule"],
        "stitched_from_track_id": record["stitched_from_track_id"],
        "stitch_gap_sec": record["stitch_gap_sec"],
        "stitch_fused_score": record["stitch_fused_score"],
        "stitch_global_similarity": record["stitch_global_similarity"],
        "stitch_gallery_global_topk_mean": record["stitch_gallery_global_topk_mean"],
        "raw_class_name": record["raw_class_name"],
        "class_name": record["class_name"],
        "final_class_name": record["final_class_name"],
        "stitch_cache_file": record["cache_path"],
    }
    extra = {k: v for k, v in extra.items() if v is not None}

    # canonical track_id = final stitched identity.
    # ByteTrack 원본 ID는 original_track_id에 보존한다.
    return Detection(
        crop=str(record["crop_path"]),
        label=str(record["label"]),
        score=float(record["confidence"]),
        bbox=tuple(float(x) for x in record["bbox"]),
        image_id=record["video_path"] or record["video_name"],
        frame_idx=int(record["frame_idx"]),
        track_id=iid,
        extra=extra,
    )


class StitchCache:
    def __init__(self):
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _load(self, path: str) -> Dict[str, np.ndarray]:
        if path in self._cache:
            return self._cache[path]
        if not path:
            self._cache[path] = {}
            return self._cache[path]
        p = Path(path)
        if not p.exists():
            self._cache[path] = {}
            return self._cache[path]
        with np.load(p, allow_pickle=False) as data:
            loaded = {
                key: np.asarray(data[key], dtype=np.float32).copy()
                for key in data.files
            }
        self._cache[path] = loaded
        return loaded

    @staticmethod
    def candidate_keys(kind: str, iid: int) -> Tuple[str, ...]:
        if kind == "object":
            return (f"object_{iid}_global", f"object_{iid}")
        return (f"person_{iid}_global", f"person_{iid}")

    def get_identity_vector(
        self,
        record: dict,
        vector_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        kind = record["kind"]
        if kind == "object" and vector_name != "dinov2":
            return None, None
        if kind == "person" and vector_name != "solider":
            return None, None

        values = self._load(record["cache_path"])
        for key in self.candidate_keys(kind, int(record["identity_id"])):
            if key in values:
                return np.asarray(values[key], dtype=np.float32).reshape(-1), key
        return None, None


def make_router(cfg: PipelineConfig, retriever_names: Iterable[str]) -> Optional[Router]:
    names = [name for name in retriever_names if name in cfg.retrievers]
    if not names:
        return None
    sub_cfg = replace(
        cfg,
        retrievers={name: cfg.retrievers[name] for name in names},
    )
    registry = EmbedderRegistry(sub_cfg)
    return Router(sub_cfg, registry, input_format="rgb")


def expected_vector_names(det: Detection, cfg: PipelineConfig) -> set:
    person_labels = {str(x).lower() for x in cfg.person_labels}
    is_person = str(det.label).lower() in person_labels
    return {
        name
        for name, spec in cfg.retrievers.items()
        if (
            spec.scope == "all"
            or (spec.scope == "person" and is_person)
            or (spec.scope == "object" and not is_person)
        )
    }


def validate_vectors(detections: List[Detection], vectors: List[dict], cfg: PipelineConfig):
    if len(detections) != len(vectors):
        raise RuntimeError("detections/vectors length mismatch")
    for i, (det, vector_map) in enumerate(zip(detections, vectors)):
        expected = expected_vector_names(det, cfg)
        actual = set(vector_map)
        if actual != expected:
            raise RuntimeError(
                f"Router/cache routing mismatch index={i}, label={det.label}, "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )


class EmbeddingEngine:
    """
    Stitch 단계 identity-level cache를 재사용한다.

      object -> DINOv2 object_<id>_global
      person -> SOLIDER person_<id>[_global]

    SigLIP2 / IRRA 등 나머지는 현재 crop에서 계산한다.
    cache miss만 해당 named vector 하나를 crop에서 재계산한다.
    """

    CACHE_NAMES = {"dinov2", "solider"}

    def __init__(self, cfg: PipelineConfig, use_stitch_cache: bool = True):
        self.cfg = cfg
        self.use_stitch_cache = use_stitch_cache
        compute_names = [
            name
            for name in cfg.retrievers
            if (not use_stitch_cache or name not in self.CACHE_NAMES)
        ]
        self.compute_router = make_router(cfg, compute_names)
        self.single_routers: Dict[str, Router] = {}
        self.cache = StitchCache()
        self.cache_hits = 0
        self.cache_misses = 0
        self.recomputed_cache_misses = 0

    def single_router(self, name: str) -> Router:
        if name not in self.single_routers:
            router = make_router(self.cfg, [name])
            if router is None:
                raise RuntimeError(f"cannot build router for {name}")
            self.single_routers[name] = router
        return self.single_routers[name]

    def embed_batch(
        self,
        records: List[dict],
        detections: List[Detection],
    ) -> List[Dict[str, np.ndarray]]:
        if not detections:
            return []

        if self.compute_router is None:
            vector_maps: List[Dict[str, np.ndarray]] = [{} for _ in detections]
        else:
            vector_maps = self.compute_router.embed(detections)

        for vector_name in ("dinov2", "solider"):
            if vector_name not in self.cfg.retrievers:
                continue

            missing_indices: List[int] = []
            for i, (record, det) in enumerate(zip(records, detections)):
                if vector_name not in expected_vector_names(det, self.cfg):
                    continue
                if not self.use_stitch_cache:
                    continue

                vec, cache_key = self.cache.get_identity_vector(record, vector_name)
                if vec is not None:
                    vector_maps[i][vector_name] = vec
                    self.cache_hits += 1
                    det.extra[f"vector_source_{vector_name}"] = "stitch_identity_global"
                    det.extra[f"stitch_cache_key_{vector_name}"] = cache_key
                else:
                    self.cache_misses += 1
                    missing_indices.append(i)

            if missing_indices:
                router = self.single_router(vector_name)
                miss_dets = [detections[i] for i in missing_indices]
                miss_maps = router.embed(miss_dets)
                for idx, vec_map in zip(missing_indices, miss_maps):
                    if vector_name not in vec_map:
                        raise RuntimeError(
                            f"{vector_name} fallback returned no vector for index={idx}"
                        )
                    vector_maps[idx][vector_name] = vec_map[vector_name]
                    detections[idx].extra[f"vector_source_{vector_name}"] = "recomputed_crop"
                    self.recomputed_cache_misses += 1

        validate_vectors(detections, vector_maps, self.cfg)
        return vector_maps


def state_path(ingest_type: str, use_stitch_cache: bool) -> Path:
    mode = "cache" if use_stitch_cache else "recompute"
    return STATE_DIR / f"state_{ingest_type}_{mode}.json"


def record_signature(records: List[dict], ingest_type: str) -> str:
    parts = [f"type={ingest_type}", f"total={len(records)}"]
    if records:
        indexes = {0, len(records) // 2, len(records) - 1}
        for idx in sorted(indexes):
            parts.append(records[idx]["detection_id"])
            parts.append(records[idx]["source_jsonl"])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(parts)))


def save_state(path: Path, next_index: int, total: int, signature: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {"next_index": next_index, "total": total, "signature": signature},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_state(
    path: Path,
    total: int,
    signature: str,
    fresh: bool,
) -> int:
    if fresh or not path.exists():
        return 0
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if int(state.get("total", -1)) != total:
        print("[!] record 총수가 달라 checkpoint를 0부터 시작합니다.")
        return 0
    if str(state.get("signature", "")) != signature:
        print("[!] stitched input signature가 달라 checkpoint를 0부터 시작합니다.")
        return 0
    return max(0, int(state.get("next_index", 0)))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Current stitched video tracks -> forensic_person / forensic_object Qdrant"
        )
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-crops", type=int, default=None)
    ap.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="maximum stitched video directories per selected kind",
    )
    ap.add_argument(
        "--type",
        choices=["all", "person", "object"],
        default="all",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="ignore builder checkpoint; does NOT recreate/delete Qdrant collections",
    )
    ap.add_argument(
        "--no-stitch-cache",
        action="store_true",
        help="recompute SOLIDER/DINOv2 per crop instead of stitch identity cache",
    )
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.max_crops is not None and args.max_crops <= 0:
        raise ValueError("--max-crops must be >= 1")
    if args.max_videos is not None and args.max_videos <= 0:
        raise ValueError("--max-videos must be >= 1")

    use_stitch_cache = not args.no_stitch_cache
    cfg = PipelineConfig.load(CONFIG_PATH)
    person_cfg, object_cfg = make_collection_configs(cfg)

    print("=" * 88)
    print("CURRENT STITCHED VIDEO DB BUILDER")
    print("=" * 88)
    print("video root        :", VIDEO_ROOT)
    print("person track root :", PERSON_TRACK_ROOT)
    print("object track root :", OBJECT_TRACK_ROOT)
    print("person collection :", person_cfg.collection)
    print("person vectors    :", sorted(person_cfg.retrievers))
    print("object collection :", object_cfg.collection)
    print("object vectors    :", sorted(object_cfg.retrievers))
    print("stitch cache reuse:", use_stitch_cache)
    print("IMPORTANT         : existing image points are NOT deleted")

    person_store = None
    object_store = None
    if args.type in ("all", "person"):
        person_store = QdrantStore(person_cfg)
        person_store.ensure_collection(recreate=False)
    if args.type in ("all", "object"):
        object_store = QdrantStore(object_cfg)
        object_store.ensure_collection(recreate=False)

    video_index = build_video_index()
    print("video files       :", f"{len(video_index):,}")

    records: List[dict] = []
    if args.type in ("all", "person"):
        person_records = discover_kind_records(
            "person", video_index, args.max_videos
        )
        records.extend(person_records)
        print("person rows       :", f"{len(person_records):,}")
    if args.type in ("all", "object"):
        object_records = discover_kind_records(
            "object", video_index, args.max_videos
        )
        records.extend(object_records)
        print("object rows       :", f"{len(object_records):,}")

    if args.max_crops is not None:
        records = records[: args.max_crops]

    total = len(records)
    if total == 0:
        print("No stitched video rows found.")
        print("Object requires e.g. stitched_tracks_v4_1.jsonl")
        print("Person requires stitched_tracks*.jsonl")
        return

    signature = record_signature(records, args.type)
    checkpoint = state_path(args.type, use_stitch_cache)
    start_index = load_state(checkpoint, total, signature, args.fresh)
    if start_index > 0:
        start_index = max(0, start_index - args.batch_size)
        print("resume safety replay:", f"{start_index:,}/{total:,}")

    embedding_engine = EmbeddingEngine(cfg, use_stitch_cache=use_stitch_cache)
    person_labels = {str(x).lower() for x in cfg.person_labels}

    run_start = time.time()
    processed = 0
    uploaded_person_total = 0
    uploaded_object_total = 0

    try:
        for start in range(start_index, total, args.batch_size):
            end = min(start + args.batch_size, total)
            batch_records = records[start:end]
            detections = [record_to_detection(r) for r in batch_records]
            vectors = embedding_engine.embed_batch(batch_records, detections)

            person_dets, person_vecs = [], []
            object_dets, object_vecs = [], []

            for det, vec_map in zip(detections, vectors):
                if str(det.label).lower() in person_labels:
                    person_dets.append(det)
                    person_vecs.append(
                        {k: v for k, v in vec_map.items() if k in person_cfg.retrievers}
                    )
                else:
                    object_dets.append(det)
                    object_vecs.append(
                        {k: v for k, v in vec_map.items() if k in object_cfg.retrievers}
                    )

            uploaded_person = 0
            uploaded_object = 0

            if person_dets:
                if person_store is None:
                    raise RuntimeError("person_store is not initialized")
                uploaded_person = person_store.upsert(
                    person_dets,
                    person_vecs,
                    batch_size=args.batch_size,
                )

            if object_dets:
                if object_store is None:
                    raise RuntimeError("object_store is not initialized")
                uploaded_object = object_store.upsert(
                    object_dets,
                    object_vecs,
                    batch_size=args.batch_size,
                )

            uploaded_person_total += uploaded_person
            uploaded_object_total += uploaded_object
            save_state(checkpoint, end, total, signature)

            processed += end - start
            elapsed = time.time() - run_start
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = total - end
            eta = remaining / rate if rate > 0 else 0.0
            h, rem = divmod(int(eta), 3600)
            m, s = divmod(rem, 60)

            print(
                f"[{end:,}/{total:,}] {end / total * 100:6.2f}% | "
                f"{rate:6.2f} crop/s | ETA {h:02d}:{m:02d}:{s:02d} | "
                f"Qdrant person={uploaded_person} object={uploaded_object} | "
                f"cache hit={embedding_engine.cache_hits:,} "
                f"miss={embedding_engine.cache_misses:,}"
            )

    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint was saved after the last successful batch.")
        return

    print("\n" + "=" * 88)
    print("VIDEO DB BUILD COMPLETE")
    print("processed              :", f"{total:,}")
    print("person upserts         :", f"{uploaded_person_total:,}")
    print("object upserts         :", f"{uploaded_object_total:,}")
    print("stitch cache hits      :", f"{embedding_engine.cache_hits:,}")
    print("stitch cache misses    :", f"{embedding_engine.cache_misses:,}")
    print("cache miss recomputes  :", f"{embedding_engine.recomputed_cache_misses:,}")
    print("image points           : preserved")
    print("=" * 88)


if __name__ == "__main__":
    main()