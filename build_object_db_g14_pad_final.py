#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
새 Qdrant collection에 Object crop을 재색인한다.

최종 구성
- SigLIP2 : 768D
- DINOv2  : Giant + Registers + AR-Pad224 + CLS + 1536D + L2
- Qdrant named vectors:
    siglip2 : 768D
    dinov2  : 1536D

안전 원칙
- 기존 collection 삭제/재생성 안 함.
- 기본 새 collection: forensic_object_g14_pad
- collection이 이미 있으면 vector schema를 검증한 뒤 이어서 upsert.
- 같은 crop_path는 deterministic UUID5를 사용하므로 재실행해도 같은 point를 덮어쓴다.

권장 실행:
python .\build_object_db_g14_pad_final.py `
  --selected-json ".\outputs\object_crop_quality_v3_fixed\selected_crops_v3.json" `
  --collection forensic_object_g14_pad `
  --batch-size 4

필요 패키지:
pip install qdrant-client pyyaml opencv-python
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import yaml
from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "pipeline.yaml"
DEFAULT_SELECTED = ROOT / "outputs" / "object_crop_quality_v3_fixed" / "selected_crops_v3.json"
DEFAULT_COLLECTION = "forensic_object_g14_pad"

SIGLIP_NAME = "siglip2"
DINO_NAME = "dinov2"
SIGLIP_DIM = 768
DINO_DIM = 1536


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def import_class(module_name: str, class_name: str):
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_from_cfg(cfg: dict, name: str, batch_size_override: int | None = None):
    spec = cfg["retrievers"][name]
    cls = import_class(spec["module"], spec["class"])
    params = dict(spec.get("params") or {})

    if batch_size_override is not None and "batch_size" in inspect.signature(cls).parameters:
        params["batch_size"] = batch_size_override
    elif batch_size_override is not None and "batch_size" in params:
        params["batch_size"] = batch_size_override

    print(f"[LOAD] {name}: {spec['module']}.{spec['class']}")
    obj = cls(**params)
    return obj, spec


def resolve_crop_path(item: Any, selected_json: Path) -> Path | None:
    if isinstance(item, str):
        raw = item
    elif isinstance(item, dict):
        raw = None
        for key in (
            "crop_path", "path", "image_path", "file_path", "filepath",
            "crop", "image", "filename"
        ):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                raw = v.strip()
                break
        if raw is None:
            return None
    else:
        return None

    p = Path(raw).expanduser()
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates += [
            (ROOT / p),
            (selected_json.parent / p),
        ]

    for c in candidates:
        if c.is_file():
            return c.resolve()

    # 원래 경로를 반환해 missing으로 집계
    return candidates[0].resolve() if candidates else p.resolve()


def normalize_items(data: Any) -> List[Any]:
    """
    selected_crops_v3.json 지원 구조:

    1) list
       [item, item, ...]

    2) wrapper dict
       {"selected": [...]}, {"crops": [...]}, ...

    3) 현재 SCVD 구조
       {
         "n001_converted/bench_0001": [...],
         "n001_converted/chair_0001": [...],
         ...
       }

       즉 track/group key -> crop list.
       이 경우 모든 list를 flatten하고 group_id를 payload에 보존한다.

    4) path -> metadata dict
       {"path/to/crop.jpg": {...}, ...}
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # 일반 wrapper 구조
        for key in (
            "selected",
            "selected_crops",
            "crops",
            "items",
            "data",
            "records",
        ):
            v = data.get(key)
            if isinstance(v, list):
                return v

        # 현재 selected_crops_v3.json:
        # group/track key -> list
        if data and all(isinstance(v, list) for v in data.values()):
            out = []

            for group_id, rows in data.items():
                for row in rows:
                    if isinstance(row, dict):
                        item = dict(row)
                        item.setdefault("group_id", str(group_id))
                        item.setdefault("track_key", str(group_id))
                        out.append(item)

                    elif isinstance(row, str):
                        out.append({
                            "crop_path": row,
                            "group_id": str(group_id),
                            "track_key": str(group_id),
                        })

                    else:
                        # 예상치 못한 primitive가 있어도 path resolver에서
                        # 안전하게 skip될 수 있도록 원형을 유지한다.
                        out.append(row)

            return out

        # path -> metadata dict 구조
        if data and all(isinstance(v, dict) for v in data.values()):
            out = []

            for k, v in data.items():
                row = dict(v)
                row.setdefault("crop_path", k)
                out.append(row)

            return out

    raise ValueError(
        "selected JSON 구조를 인식하지 못했습니다. "
        f"type={type(data).__name__}"
    )


def sanitize_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return sanitize_payload(value.item())

    if isinstance(value, dict):
        return {str(k): sanitize_payload(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [sanitize_payload(v) for v in value]

    return str(value)


def make_payload(item: Any, crop_path: Path) -> dict:
    if isinstance(item, dict):
        payload = sanitize_payload(dict(item))
    else:
        payload = {}

    payload["crop_path"] = str(crop_path)
    payload["path"] = str(crop_path)
    payload["is_person"] = False
    payload["embedding_pipeline"] = "siglip2+dino_g14_registers_ar_pad224_cls"
    payload["dino_dim"] = DINO_DIM
    payload["siglip_dim"] = SIGLIP_DIM
    return payload


def deterministic_id(crop_path: Path) -> str:
    # machine-independent logical relative path when possible
    try:
        logical = crop_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        logical = crop_path.resolve().as_posix()

    return str(uuid.uuid5(uuid.NAMESPACE_URL, "forensic-object:" + logical))


def read_bgr(paths: List[Path]) -> Tuple[List[np.ndarray], List[Path]]:
    images = []
    ok_paths = []

    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] cv2.imread failed: {p}")
            continue
        images.append(img)
        ok_paths.append(p)

    return images, ok_paths


def call_embedder(embedder, images_bgr: List[np.ndarray]) -> np.ndarray:
    """
    프로젝트 embedder API 차이를 최대한 흡수한다.
    우선순위:
      embed_crops(images, input_format='bgr')
      embed_crops(images)
      embed_images(images, input_format='bgr')
      embed_images(images)
      encode(images, input_format='bgr')
      encode(images)
    """
    candidates = ("embed_crops", "embed_images", "encode")

    last_error = None
    for name in candidates:
        fn = getattr(embedder, name, None)
        if fn is None:
            continue

        try:
            sig = inspect.signature(fn)
            if "input_format" in sig.parameters:
                out = fn(images_bgr, input_format="bgr")
            else:
                out = fn(images_bgr)
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            return arr
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"지원 가능한 embed API를 찾지 못했습니다. "
        f"마지막 오류={last_error}"
    )


def get_collection_vectors(client: QdrantClient, collection: str) -> dict | None:
    try:
        info = client.get_collection(collection)
    except Exception:
        return None

    vectors = info.config.params.vectors
    if not isinstance(vectors, dict):
        raise RuntimeError(
            f"{collection}은 named-vector collection이 아닙니다: {vectors}"
        )
    return vectors


def ensure_collection(
    client: QdrantClient,
    collection: str,
    on_disk: bool,
):
    existing = get_collection_vectors(client, collection)

    if existing is None:
        print(f"[CREATE] new collection: {collection}")
        client.create_collection(
            collection_name=collection,
            vectors_config={
                SIGLIP_NAME: models.VectorParams(
                    size=SIGLIP_DIM,
                    distance=models.Distance.COSINE,
                    on_disk=on_disk,
                ),
                DINO_NAME: models.VectorParams(
                    size=DINO_DIM,
                    distance=models.Distance.COSINE,
                    on_disk=on_disk,
                ),
            },
        )
        return

    print(f"[FOUND] collection already exists: {collection}")

    for name, expected in (
        (SIGLIP_NAME, SIGLIP_DIM),
        (DINO_NAME, DINO_DIM),
    ):
        if name not in existing:
            raise RuntimeError(
                f"기존 collection에 named vector '{name}'가 없습니다."
            )
        actual = int(existing[name].size)
        if actual != expected:
            raise RuntimeError(
                f"{name} dim mismatch: expected={expected}, actual={actual}"
            )

    print("[OK] existing vector schema matches")


def validate_vectors(name: str, arr: np.ndarray, dim: int):
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise RuntimeError(f"{name} shape mismatch: {arr.shape}, expected (*,{dim})")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name}: NaN/Inf 발견")

    norms = np.linalg.norm(arr, axis=1)
    print(
        f"[CHECK] {name}: shape={arr.shape} "
        f"norm mean={norms.mean():.6f} "
        f"min={norms.min():.6f} max={norms.max():.6f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--selected-json", default=str(DEFAULT_SELECTED))
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--qdrant-url", default=None)
    ap.add_argument("--on-disk", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    selected_json = Path(args.selected_json).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not selected_json.is_file():
        raise FileNotFoundError(selected_json)

    cfg = load_yaml(config_path)

    # SSOT 검증
    dino_spec = cfg["retrievers"]["dinov2"]
    siglip_spec = cfg["retrievers"]["siglip2"]

    if int(dino_spec["dim"]) != DINO_DIM:
        raise RuntimeError(f"pipeline.yaml dinov2.dim != {DINO_DIM}")
    if int(siglip_spec["dim"]) != SIGLIP_DIM:
        raise RuntimeError(f"pipeline.yaml siglip2.dim != {SIGLIP_DIM}")

    qdrant_url = args.qdrant_url or cfg["qdrant"]["url"]

    raw = json.loads(selected_json.read_text(encoding="utf-8"))
    items = normalize_items(raw)

    records = []
    missing = 0
    for item in items:
        p = resolve_crop_path(item, selected_json)
        if p is None or not p.is_file():
            missing += 1
            continue
        records.append((item, p))

    if args.limit > 0:
        records = records[: args.limit]

    print("=" * 88)
    print("OBJECT DB REINDEX — FINAL DINO G14 REGISTERS AR-PAD224 CLS")
    print("=" * 88)
    print("config      :", config_path)
    print("selected    :", selected_json)
    print("json items  :", len(items))
    print("valid crops :", len(records))
    print("missing     :", missing)
    print("collection  :", args.collection)
    print("qdrant url  :", qdrant_url)
    print("batch size  :", args.batch_size)
    print("dry run     :", args.dry_run)

    if not records:
        raise RuntimeError("색인할 crop이 없습니다.")

    # Embedder는 YAML에서 로드하여 색인/검색 계약을 일치시킨다.
    siglip, _ = instantiate_from_cfg(
        cfg, "siglip2",
        batch_size_override=max(args.batch_size, 1),
    )
    dino, _ = instantiate_from_cfg(
        cfg, "dinov2",
        batch_size_override=args.batch_size,
    )

    client = QdrantClient(url=qdrant_url)

    if not args.dry_run:
        ensure_collection(
            client,
            args.collection,
            on_disk=bool(args.on_disk or cfg.get("qdrant", {}).get("on_disk", False)),
        )

    inserted = 0
    failed_read = 0

    for start in range(0, len(records), args.batch_size):
        chunk = records[start : start + args.batch_size]
        paths = [p for _, p in chunk]

        images, ok_paths = read_bgr(paths)
        if not images:
            failed_read += len(paths)
            continue

        # read 실패 항목이 있으면 path 기준으로 원 record 복원
        by_path = {str(p): item for item, p in chunk}
        ok_items = [(by_path[str(p)], p) for p in ok_paths]

        sig_vec = call_embedder(siglip, images)
        dino_vec = call_embedder(dino, images)

        validate_vectors(SIGLIP_NAME, sig_vec, SIGLIP_DIM)
        validate_vectors(DINO_NAME, dino_vec, DINO_DIM)

        if len(sig_vec) != len(ok_items) or len(dino_vec) != len(ok_items):
            raise RuntimeError(
                f"batch length mismatch: items={len(ok_items)}, "
                f"siglip={len(sig_vec)}, dino={len(dino_vec)}"
            )

        points = []
        for i, (item, path) in enumerate(ok_items):
            points.append(
                models.PointStruct(
                    id=deterministic_id(path),
                    vector={
                        SIGLIP_NAME: sig_vec[i].tolist(),
                        DINO_NAME: dino_vec[i].tolist(),
                    },
                    payload=make_payload(item, path),
                )
            )

        if not args.dry_run:
            client.upsert(
                collection_name=args.collection,
                points=points,
                wait=True,
            )

        inserted += len(points)

        print(
            f"[{inserted:,}/{len(records):,}] "
            f"{inserted / len(records) * 100:6.2f}%"
        )

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)
    print("selected :", len(records))
    print("inserted :", inserted)
    print("read fail:", failed_read)

    if not args.dry_run:
        count = client.count(
            collection_name=args.collection,
            exact=True,
        ).count
        print("qdrant count:", count)
        print("collection  :", args.collection)

    print("\n기존 collection은 삭제/재생성하지 않았습니다.")


if __name__ == "__main__":
    main()
