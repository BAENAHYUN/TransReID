"""
ingest_selected_objects_g14_reg_fixed.py

object_crop_quality_v3_fixed.py가 만든 selected_crops_v3.json만 사용해
최종 object DB를 재구축한다.

파이프라인
----------
selected crop
  ├─ SigLIP2 base              -> 768D
  └─ DINOv2 ViT-g/14 Registers -> 1536D CLS
          ↓
      L2 normalize
          ↓
Qdrant forensic_object

기본 경로
---------
selected:
  outputs/object_crop_quality_v3_fixed/selected_crops_v3.json
Qdrant:
  data/qdrant_local
collection:
  forensic_object

중요
----
기존 forensic_object는 dino=768D이므로 g/14 1536D와 호환되지 않는다.
최종 재구축 시 --recreate를 사용해야 기존 object collection을 삭제하고
siglip=768, dino=1536 schema로 다시 만든다.

실행
----
python .\src\RF-DETR\ingest_selected_objects_g14_reg.py --recreate

테스트
------
python .\src\RF-DETR\ingest_selected_objects_g14_reg.py --recreate --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoModel, AutoProcessor

THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from embedders.object.dinov2_embedder_g14_reg import (
    DINOv2G14RegistersEmbedder,
    OUTPUT_DIM as DINO_DIM,
)

COLLECTION = "forensic_object_g14"
SIGLIP_MODEL = "google/siglip2-base-patch16-224"
SIGLIP_DIM = 768

DEFAULT_SELECTED = (
    ROOT_DIR
    / "outputs"
    / "object_crop_quality_v3_fixed"
    / "selected_crops_v3.json"
)
DEFAULT_QDRANT = ROOT_DIR / "data" / "qdrant_local"


def l2(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1)


class SigLIP2ImageEncoder:
    def __init__(self, device: str = "cuda", batch_size: int = 32) -> None:
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.batch_size = int(batch_size)

        print(f"[*] SigLIP2 로드 중: {SIGLIP_MODEL}")
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)

        dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else (torch.float16 if self.device.type == "cuda" else torch.float32)
        )

        self.model = AutoModel.from_pretrained(
            SIGLIP_MODEL,
            torch_dtype=dtype,
        ).to(self.device).eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        print(f"[+] SigLIP2 준비 완료: device={self.device}, dim={SIGLIP_DIM}")

    @torch.inference_mode()
    def encode(self, paths: Sequence[Path]) -> np.ndarray:
        outputs: List[np.ndarray] = []

        for s in range(0, len(paths), self.batch_size):
            batch_paths = paths[s:s + self.batch_size]
            images: List[Image.Image] = []

            for p in batch_paths:
                with Image.open(p) as im:
                    images.append(im.convert("RGB").copy())

            inputs = self.processor(
                images=images,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
                if torch.is_tensor(v)
            }

            if hasattr(self.model, "get_image_features"):
                feats = self.model.get_image_features(**inputs)
            else:
                feats = self.model(**inputs)

            # transformers 버전에 따라 get_image_features()가 Tensor가 아니라
            # BaseModelOutputWithPooling / ModelOutput을 반환할 수 있다.
            if not torch.is_tensor(feats):
                if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                    feats = feats.image_embeds
                elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                elif isinstance(feats, dict) and feats.get("image_embeds") is not None:
                    feats = feats["image_embeds"]
                elif isinstance(feats, dict) and feats.get("pooler_output") is not None:
                    feats = feats["pooler_output"]
                else:
                    raise RuntimeError(
                        "SigLIP2 image embedding Tensor를 찾지 못했습니다. "
                        f"returned={type(feats)!r}"
                    )

            feats = l2(feats)
            arr = feats.cpu().numpy().astype(np.float32, copy=False)

            if arr.ndim != 2 or arr.shape[1] != SIGLIP_DIM:
                raise RuntimeError(
                    f"SigLIP2 shape 불일치: expected=(N,{SIGLIP_DIM}), got={arr.shape}"
                )

            outputs.append(arr)

        if not outputs:
            return np.empty((0, SIGLIP_DIM), dtype=np.float32)

        return np.concatenate(outputs, axis=0)


class DINOEncoder:
    def __init__(self, device: str = "cuda", batch_size: int = 4) -> None:
        print("[*] DINOv2 ViT-g/14 + Registers 로드 중")
        self.embedder = DINOv2G14RegistersEmbedder(
            device=device,
            batch_size=batch_size,
        )
        print(f"[+] DINOv2 준비 완료: dim={DINO_DIM}")

    def encode(self, paths: Sequence[Path]) -> np.ndarray:
        images: List[np.ndarray] = []

        for p in paths:
            with Image.open(p) as im:
                images.append(
                    np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
                )

        vecs = self.embedder.embed_crops(
            images,
            input_format="rgb",
        )
        vecs = np.asarray(vecs, dtype=np.float32)

        if vecs.ndim != 2 or vecs.shape[1] != DINO_DIM:
            raise RuntimeError(
                f"DINO shape 불일치: expected=(N,{DINO_DIM}), got={vecs.shape}"
            )

        norms = np.linalg.norm(vecs, axis=1)
        if not np.all(np.isfinite(vecs)):
            raise RuntimeError("DINO embedding에 NaN/Inf가 있습니다.")
        if len(norms) and np.max(np.abs(norms - 1.0)) > 1e-3:
            raise RuntimeError(
                f"DINO L2 norm 불일치: min={norms.min():.6f}, max={norms.max():.6f}"
            )

        return vecs


def load_selected(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []

    if isinstance(raw, dict):
        for track_key, items in raw.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row.setdefault("track_key", track_key)
                rows.append(row)
    elif isinstance(raw, list):
        rows = [dict(x) for x in raw if isinstance(x, dict)]
    else:
        raise ValueError("selected JSON 구조를 이해할 수 없습니다.")

    clean: List[Dict[str, Any]] = []
    seen = set()

    for row in rows:
        p = Path(str(row.get("path", "")))

        if not p.is_absolute():
            p = ROOT_DIR / p

        try:
            p = p.resolve()
        except Exception:
            continue

        if not p.is_file():
            print(f"[!] crop 없음, skip: {p}")
            continue

        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)

        row["path"] = str(p)
        clean.append(row)

        if limit > 0 and len(clean) >= limit:
            break

    return clean


def stable_point_id(path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"transreid-object:{Path(path).as_posix()}"))


def collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        return bool(client.collection_exists(name))
    except AttributeError:
        try:
            client.get_collection(name)
            return True
        except Exception:
            return False


def prepare_collection(
    client: QdrantClient,
    *,
    recreate: bool,
) -> None:
    exists = collection_exists(client, COLLECTION)

    if exists and recreate:
        print(f"[*] 기존 {COLLECTION} 삭제")
        client.delete_collection(COLLECTION)
        exists = False

    if not exists:
        print(
            f"[*] {COLLECTION} 생성: "
            f"siglip={SIGLIP_DIM}D, dino={DINO_DIM}D"
        )
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "siglip": VectorParams(
                    size=SIGLIP_DIM,
                    distance=Distance.COSINE,
                ),
                "dino": VectorParams(
                    size=DINO_DIM,
                    distance=Distance.COSINE,
                ),
            },
        )
        return

    info = client.get_collection(COLLECTION)
    vectors = info.config.params.vectors

    def dim_of(name: str) -> int:
        spec = vectors[name]
        return int(spec.size)

    actual_siglip = dim_of("siglip")
    actual_dino = dim_of("dino")

    if actual_siglip != SIGLIP_DIM or actual_dino != DINO_DIM:
        raise RuntimeError(
            f"기존 {COLLECTION} schema가 호환되지 않습니다. "
            f"actual siglip={actual_siglip}, dino={actual_dino}; "
            f"required siglip={SIGLIP_DIM}, dino={DINO_DIM}. "
            "최종 DB 재구축은 --recreate로 실행하세요."
        )

    count = client.count(
        collection_name=COLLECTION,
        exact=True,
    ).count

    if count:
        raise RuntimeError(
            f"{COLLECTION}에 기존 point가 {count:,}개 있습니다. "
            "stale 768D DB와 섞이지 않도록 --recreate를 사용하세요."
        )


def make_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "path",
        "video",
        "track",
        "track_key",
        "label",
        "frame_idx",
        "width",
        "height",
        "area",
        "blur_var",
        "resolution_score",
        "sharpness_score",
        "person_count",
        "person_overlap",
        "person_max_conf",
        "overlap_score",
        "quality_score",
        "accepted",
        "reasons",
    }

    payload = {
        k: row[k]
        for k in keep
        if k in row
    }

    payload["embedding_model_dino"] = "facebook/dinov2-with-registers-giant"
    payload["embedding_feature_dino"] = "cls"
    payload["embedding_dim_dino"] = DINO_DIM
    payload["embedding_model_siglip"] = SIGLIP_MODEL
    payload["embedding_dim_siglip"] = SIGLIP_DIM

    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    ap.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--siglip-batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    selected_path = args.selected.resolve()
    qdrant_path = args.qdrant_path.resolve()

    if not selected_path.is_file():
        raise SystemExit(f"selected JSON 없음: {selected_path}")

    rows = load_selected(selected_path, args.limit)

    if not rows:
        raise SystemExit("적재할 selected crop이 없습니다.")

    print("=" * 78)
    print("FINAL OBJECT DB — DINOv2 ViT-g/14 + Registers")
    print(f"Selected JSON : {selected_path}")
    print(f"Crops         : {len(rows):,}")
    print(f"Qdrant        : {qdrant_path}")
    print(f"Collection    : {COLLECTION}")
    print(f"SigLIP2       : {SIGLIP_DIM}D")
    print(f"DINOv2 g/14 R : {DINO_DIM}D CLS")
    print(f"DINO batch    : {args.batch_size}")
    print("=" * 78)

    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    prepare_collection(client, recreate=args.recreate)

    siglip = SigLIP2ImageEncoder(
        device=args.device,
        batch_size=args.siglip_batch_size,
    )
    dino = DINOEncoder(
        device=args.device,
        batch_size=args.batch_size,
    )

    total = len(rows)
    ingest_batch = max(1, args.batch_size)

    inserted = 0

    for start in range(0, total, ingest_batch):
        chunk = rows[start:start + ingest_batch]
        paths = [Path(r["path"]) for r in chunk]

        sig_vecs = siglip.encode(paths)
        dino_vecs = dino.encode(paths)

        points: List[PointStruct] = []

        for row, sv, dv in zip(chunk, sig_vecs, dino_vecs):
            points.append(
                PointStruct(
                    id=stable_point_id(row["path"]),
                    vector={
                        "siglip": sv.tolist(),
                        "dino": dv.tolist(),
                    },
                    payload=make_payload(row),
                )
            )

        client.upsert(
            collection_name=COLLECTION,
            points=points,
            wait=True,
        )

        inserted += len(points)
        print(
            f"[{inserted:5d}/{total:5d}] "
            f"{inserted / total * 100:6.2f}%"
        )

    count = client.count(
        collection_name=COLLECTION,
        exact=True,
    ).count

    print()
    print("=" * 78)
    print("COMPLETE")
    print(f"Selected crops : {total:,}")
    print(f"Inserted       : {inserted:,}")
    print(f"Qdrant points  : {count:,}")
    print(f"SigLIP vector  : {SIGLIP_DIM}D")
    print(f"DINO vector    : {DINO_DIM}D")
    print("=" * 78)

    if count != total:
        raise RuntimeError(
            f"최종 point count 불일치: expected={total}, actual={count}"
        )


if __name__ == "__main__":
    main()
