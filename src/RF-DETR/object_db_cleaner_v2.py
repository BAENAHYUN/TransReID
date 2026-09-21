"""
object_db_cleaner_v2.py

V1 개선판:
- 단일 class centroid 대신 class 내부 multi-centroid(sub-cluster) 사용
- 정상적인 외형 다양성을 허용해서 false REVIEW를 줄임
- DINO kNN + SigLIP 의미 검증은 유지
- Spatial dedup은 bbox 있을 때만 수행
- 기본 실행은 DB 수정 없이 결과만 생성
- --apply-payload 시 clean_v2_* payload 기록

대상 기본 collection:
    forensic_object_g14

출력:
    outputs/object_db_cleaner_v2/
      object_cleaning_results_v2.json
      object_cleaning_results_v2.csv
      summary_v2.json

실행:
    python .\src\RF-DETR\object_db_cleaner_v2.py

payload 반영:
    python .\src\RF-DETR\object_db_cleaner_v2.py --apply-payload
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QDRANT = ROOT / "data" / "qdrant_local"
DEFAULT_COLLECTION = "forensic_object_g14"
DEFAULT_OUTPUT = ROOT / "outputs" / "object_db_cleaner_v2"
SIGLIP_MODEL = "google/siglip2-base-patch16-224"


@dataclass
class DBRow:
    idx: int
    point_id: str
    payload: Dict[str, Any]
    dino: np.ndarray
    siglip: np.ndarray


@dataclass
class Result:
    point_id: str
    path: str
    video: str
    frame_idx: int
    label: str
    status: str
    reasons: List[str]
    suggested_label: Optional[str]

    duplicate_of: Optional[str]
    duplicate_iou: Optional[float]
    duplicate_dino_cosine: Optional[float]

    knn_top_label: Optional[str]
    knn_top_ratio: float
    knn_same_label_ratio: float

    siglip_current_score: Optional[float]
    siglip_best_label: Optional[str]
    siglip_best_score: Optional[float]
    siglip_margin: Optional[float]

    nearest_subcluster_similarity: Optional[float]
    nearest_subcluster_id: Optional[int]
    same_label_knn_similarity: Optional[float]
    outlier_score: float


def norm(x: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(d, 1e-12)


def label_of(r: DBRow) -> str:
    return str(r.payload.get("label", "")).strip()


def video_of(r: DBRow) -> str:
    return str(r.payload.get("video", "")).strip()


def frame_of(r: DBRow) -> int:
    try:
        return int(r.payload.get("frame_idx", -1))
    except Exception:
        return -1


def confidence_of(r: DBRow) -> float:
    for key in ("confidence", "det_confidence", "quality_score"):
        try:
            v = r.payload.get(key)
            if v is not None:
                return float(v)
        except Exception:
            pass
    return 0.0


def bbox_of(r: DBRow) -> Optional[Tuple[float, float, float, float]]:
    for key in ("bbox", "xyxy", "bbox_xyxy"):
        v = r.payload.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            try:
                x1, y1, x2, y2 = map(float, v)
                if x2 > x1 and y2 > y1:
                    return x1, y1, x2, y2
            except Exception:
                pass
    return None


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def load_rows(client: QdrantClient, collection: str, limit: int) -> List[DBRow]:
    rows: List[DBRow] = []
    offset = None

    while True:
        n = 256
        if limit:
            remain = limit - len(rows)
            if remain <= 0:
                break
            n = min(n, remain)

        pts, offset = client.scroll(
            collection_name=collection,
            limit=n,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        for p in pts:
            if not isinstance(p.vector, dict):
                raise RuntimeError("named vector collection이 아닙니다.")

            dino = np.asarray(p.vector["dino"], dtype=np.float32)
            siglip = np.asarray(p.vector["siglip"], dtype=np.float32)

            rows.append(
                DBRow(
                    idx=len(rows),
                    point_id=str(p.id),
                    payload=dict(p.payload or {}),
                    dino=dino,
                    siglip=siglip,
                )
            )

        if offset is None:
            break

    if not rows:
        raise RuntimeError("collection에 point가 없습니다.")

    dmat = norm(np.stack([r.dino for r in rows]))
    smat = norm(np.stack([r.siglip for r in rows]))

    for i, r in enumerate(rows):
        r.dino = dmat[i]
        r.siglip = smat[i]

    return rows


def spatial_dedup(rows, iou_thr, dino_thr):
    result = {
        r.idx: {"duplicate_of": None, "iou": None, "cos": None}
        for r in rows
    }

    groups = defaultdict(list)

    for r in rows:
        if video_of(r) and frame_of(r) >= 0:
            groups[(video_of(r), frame_of(r))].append(r)

    for group in groups.values():
        if len(group) < 2:
            continue

        group = sorted(group, key=confidence_of, reverse=True)
        suppressed = set()

        for i, a in enumerate(group):
            if a.idx in suppressed or bbox_of(a) is None:
                continue

            for b in group[i + 1:]:
                if b.idx in suppressed or bbox_of(b) is None:
                    continue

                ov = bbox_iou(bbox_of(a), bbox_of(b))
                if ov < iou_thr:
                    continue

                cs = float(np.dot(a.dino, b.dino))
                if cs < dino_thr:
                    continue

                suppressed.add(b.idx)
                result[b.idx] = {
                    "duplicate_of": a.point_id,
                    "iou": ov,
                    "cos": cs,
                }

    return result


def global_knn(rows: List[DBRow], k: int):
    x = np.stack([r.dino for r in rows]).astype(np.float32)
    n = len(rows)

    k = min(max(1, k), max(1, n - 1))

    inds = np.empty((n, k), dtype=np.int32)
    sims = np.empty((n, k), dtype=np.float32)

    block = 128

    for st in range(0, n, block):
        ed = min(n, st + block)

        sim = x[st:ed] @ x.T

        for local, gi in enumerate(range(st, ed)):
            sim[local, gi] = -np.inf

        part = np.argpartition(sim, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(vals, axis=1)[:, ::-1]

        inds[st:ed] = np.take_along_axis(part, order, axis=1)
        sims[st:ed] = np.take_along_axis(vals, order, axis=1)

        print(f"    kNN {ed:,}/{n:,}")

    return inds, sims


def knn_label_stats(rows, inds):
    labels = [label_of(r) for r in rows]
    out = {}

    for i, r in enumerate(rows):
        neigh_labels = [labels[j] for j in inds[i]]

        counts = Counter(x for x in neigh_labels if x)
        top = counts.most_common(1)[0][0] if counts else None
        top_ratio = counts[top] / len(neigh_labels) if top else 0.0
        same_ratio = sum(x == labels[i] for x in neigh_labels) / len(neigh_labels)

        out[r.idx] = {
            "top": top,
            "top_ratio": float(top_ratio),
            "same_ratio": float(same_ratio),
        }

    return out


class SigLIPTextVerifier:
    def __init__(self, device="cuda"):
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )

        print(f"[*] SigLIP2 text verifier 로드: {SIGLIP_MODEL}")
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

    @torch.inference_mode()
    def encode(self, labels):
        labels = sorted(set(x for x in labels if x))
        prompts = [f"a photo of a {x.replace('_', ' ')}" for x in labels]

        inp = self.processor(
            text=prompts,
            padding="max_length",
            return_tensors="pt",
        )
        inp = {
            k: v.to(self.device)
            for k, v in inp.items()
            if torch.is_tensor(v)
        }

        out = (
            self.model.get_text_features(**inp)
            if hasattr(self.model, "get_text_features")
            else self.model(**inp)
        )

        if not torch.is_tensor(out):
            if getattr(out, "text_embeds", None) is not None:
                out = out.text_embeds
            elif getattr(out, "pooler_output", None) is not None:
                out = out.pooler_output
            elif isinstance(out, dict) and out.get("text_embeds") is not None:
                out = out["text_embeds"]
            elif isinstance(out, dict) and out.get("pooler_output") is not None:
                out = out["pooler_output"]
            else:
                raise RuntimeError(
                    f"SigLIP text tensor를 찾을 수 없음: {type(out)!r}"
                )

        out = F.normalize(out.float(), dim=-1)
        arr = out.cpu().numpy().astype(np.float32)

        return {
            label: arr[i]
            for i, label in enumerate(labels)
        }


def semantic_stats(rows, text_vecs):
    out = {}

    for r in rows:
        current = label_of(r)

        cur = (
            float(np.dot(r.siglip, text_vecs[current]))
            if current in text_vecs
            else None
        )

        best_label = None
        best_score = None

        for lab, tv in text_vecs.items():
            score = float(np.dot(r.siglip, tv))

            if best_score is None or score > best_score:
                best_label = lab
                best_score = score

        margin = (
            None
            if cur is None or best_score is None
            else best_score - cur
        )

        out[r.idx] = {
            "current": cur,
            "best_label": best_label,
            "best_score": best_score,
            "margin": margin,
        }

    return out


def choose_cluster_count(n: int, max_clusters: int) -> int:
    """
    클래스 샘플 수에 따라 sub-cluster 개수를 자동 선택.
    너무 작은 class는 1개 중심만 사용.
    """
    if n < 12:
        return 1

    # sqrt 기반. 100개면 3, 400개면 6 정도.
    k = int(round(math.sqrt(n) / 3.0))
    k = max(2, k)
    k = min(k, max_clusters)
    k = min(k, max(1, n // 8))

    return max(1, k)


def build_multi_centroids(
    rows: List[DBRow],
    *,
    max_clusters: int,
    random_state: int,
):
    labels = [label_of(r) for r in rows]
    x = np.stack([r.dino for r in rows]).astype(np.float32)

    by_label = defaultdict(list)

    for i, lab in enumerate(labels):
        if lab:
            by_label[lab].append(i)

    models = {}
    summary = {}

    for lab, idxs in sorted(by_label.items()):
        data = x[idxs]
        k = choose_cluster_count(len(idxs), max_clusters)

        if k == 1:
            center = norm(data.mean(axis=0))
            centers = center.reshape(1, -1)
        else:
            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=random_state,
                batch_size=min(256, max(32, len(idxs))),
                n_init=10,
                max_iter=200,
                reassignment_ratio=0.01,
            )
            km.fit(data)
            centers = norm(km.cluster_centers_.astype(np.float32))

        models[lab] = centers
        summary[lab] = {
            "samples": len(idxs),
            "subclusters": int(len(centers)),
        }

    return models, summary


def multi_centroid_outlier_stats(
    rows,
    inds,
    sims,
    centroids,
):
    labels = [label_of(r) for r in rows]
    out = {}

    for i, r in enumerate(rows):
        lab = labels[i]

        nearest_sim = None
        nearest_id = None

        if lab in centroids:
            cs = centroids[lab] @ r.dino
            nearest_id = int(np.argmax(cs))
            nearest_sim = float(cs[nearest_id])

        same_sims = [
            float(s)
            for j, s in zip(inds[i], sims[i])
            if labels[j] == lab
        ][:5]

        same_knn = (
            float(np.mean(same_sims))
            if same_sims
            else None
        )

        bad_cent = (
            0.0
            if nearest_sim is None
            else max(0.0, 1.0 - nearest_sim)
        )

        bad_knn = (
            0.0
            if same_knn is None
            else max(0.0, 1.0 - same_knn)
        )

        score = 0.60 * bad_cent + 0.40 * bad_knn

        out[r.idx] = {
            "nearest_sim": nearest_sim,
            "nearest_id": nearest_id,
            "same_knn": same_knn,
            "score": float(score),
        }

    return out


def decide(r, dedup, knn, sem, outlier, args):
    reasons = []
    status = "KEEP"
    suggested = None

    current = label_of(r)

    if dedup["duplicate_of"]:
        status = "REJECT"
        reasons.append("SPATIAL_DUPLICATE")

    label_conflict = (
        knn["top"]
        and knn["top"] != current
        and knn["top_ratio"] >= args.knn_conflict_ratio
        and knn["same_ratio"] <= args.knn_same_label_min
    )

    if label_conflict and status != "REJECT":
        status = "REVIEW"
        suggested = knn["top"]
        reasons.append("DINO_LABEL_CONFLICT")

    semantic_conflict = (
        sem["best_label"]
        and sem["best_label"] != current
        and sem["margin"] is not None
        and sem["margin"] >= args.semantic_margin
    )

    if (
        status != "REJECT"
        and label_conflict
        and semantic_conflict
        and knn["top"] == sem["best_label"]
    ):
        status = "RELABEL_CANDIDATE"
        suggested = knn["top"]
        reasons.append("SIGLIP_SUPPORTS_RELABEL")

    elif semantic_conflict and status == "KEEP":
        status = "REVIEW"
        suggested = sem["best_label"]
        reasons.append("SIGLIP_LABEL_CONFLICT")

    if (
        outlier["nearest_sim"] is not None
        and outlier["nearest_sim"] < args.subcluster_min
    ):
        reasons.append("LOW_SUBCLUSTER_SIM")

        if status == "KEEP":
            status = "REVIEW"

    if (
        outlier["same_knn"] is not None
        and outlier["same_knn"] < args.same_label_knn_min
    ):
        reasons.append("LOW_SAME_LABEL_KNN_SIM")

        if status == "KEEP":
            status = "REVIEW"

    return Result(
        point_id=r.point_id,
        path=str(r.payload.get("path", "")),
        video=video_of(r),
        frame_idx=frame_of(r),
        label=current,
        status=status,
        reasons=reasons,
        suggested_label=suggested,

        duplicate_of=dedup["duplicate_of"],
        duplicate_iou=dedup["iou"],
        duplicate_dino_cosine=dedup["cos"],

        knn_top_label=knn["top"],
        knn_top_ratio=knn["top_ratio"],
        knn_same_label_ratio=knn["same_ratio"],

        siglip_current_score=sem["current"],
        siglip_best_label=sem["best_label"],
        siglip_best_score=sem["best_score"],
        siglip_margin=sem["margin"],

        nearest_subcluster_similarity=outlier["nearest_sim"],
        nearest_subcluster_id=outlier["nearest_id"],
        same_label_knn_similarity=outlier["same_knn"],
        outlier_score=outlier["score"],
    )


def write_csv(results, path):
    if not results:
        return

    fields = list(asdict(results[0]).keys())

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in results:
            d = asdict(r)
            d["reasons"] = ";".join(r.reasons)
            writer.writerow(d)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=0)

    # dedup
    ap.add_argument("--dedup-iou", type=float, default=0.60)
    ap.add_argument("--dedup-dino", type=float, default=0.92)

    # label verification
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-conflict-ratio", type=float, default=0.70)
    ap.add_argument("--knn-same-label-min", type=float, default=0.20)

    # semantic
    ap.add_argument("--semantic-margin", type=float, default=0.04)

    # multi-centroid outlier
    ap.add_argument("--max-subclusters", type=int, default=8)
    ap.add_argument("--subcluster-min", type=float, default=0.45)
    ap.add_argument("--same-label-knn-min", type=float, default=0.50)
    ap.add_argument("--random-state", type=int, default=42)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-siglip-text", action="store_true")
    ap.add_argument("--apply-payload", action="store_true")

    args = ap.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(args.qdrant_path.resolve()))

    if not client.collection_exists(args.collection):
        raise SystemExit(f"collection 없음: {args.collection}")

    print("=" * 80)
    print("OBJECT DB CLEANER V2 — MULTI-CENTROID")
    print(f"Collection : {args.collection}")
    print("=" * 80)

    print("[1/7] DB load")
    rows = load_rows(client, args.collection, args.limit)
    print(f"      points={len(rows):,}")

    bbox_count = sum(bbox_of(r) is not None for r in rows)
    print(f"      bbox payload={bbox_count:,}/{len(rows):,}")

    print("[2/7] Spatial dedup")
    dedup = spatial_dedup(
        rows,
        args.dedup_iou,
        args.dedup_dino,
    )
    dup_count = sum(
        v["duplicate_of"] is not None
        for v in dedup.values()
    )
    print(f"      duplicate={dup_count:,}")

    if bbox_count == 0:
        print("      [!] bbox 없음 -> spatial dedup skip")

    print("[3/7] DINO kNN")
    inds, sims = global_knn(
        rows,
        args.knn_k,
    )
    knn = knn_label_stats(
        rows,
        inds,
    )

    print("[4/7] SigLIP semantic check")
    if args.skip_siglip_text:
        sem = {
            r.idx: {
                "current": None,
                "best_label": None,
                "best_score": None,
                "margin": None,
            }
            for r in rows
        }
    else:
        verifier = SigLIPTextVerifier(args.device)
        text_vectors = verifier.encode(
            [label_of(r) for r in rows]
        )
        sem = semantic_stats(
            rows,
            text_vectors,
        )
        del verifier

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[5/7] Class sub-clustering")
    centroids, cluster_summary = build_multi_centroids(
        rows,
        max_clusters=args.max_subclusters,
        random_state=args.random_state,
    )

    for lab, info in sorted(
        cluster_summary.items(),
        key=lambda x: x[1]["samples"],
        reverse=True,
    )[:20]:
        print(
            f"      {lab:20s} "
            f"samples={info['samples']:4d} "
            f"subclusters={info['subclusters']}"
        )

    print("[6/7] Multi-centroid outlier")
    outlier = multi_centroid_outlier_stats(
        rows,
        inds,
        sims,
        centroids,
    )

    print("[7/7] Decision")
    results = [
        decide(
            r,
            dedup[r.idx],
            knn[r.idx],
            sem[r.idx],
            outlier[r.idx],
            args,
        )
        for r in rows
    ]

    json_path = output_dir / "object_cleaning_results_v2.json"
    csv_path = output_dir / "object_cleaning_results_v2.csv"
    summary_path = output_dir / "summary_v2.json"

    json_path.write_text(
        json.dumps(
            [asdict(r) for r in results],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(results, csv_path)

    statuses = Counter(r.status for r in results)
    reasons = Counter(
        reason
        for r in results
        for reason in r.reasons
    )

    summary = {
        "collection": args.collection,
        "points": len(rows),
        "bbox_payload_points": bbox_count,
        "spatial_duplicates": dup_count,
        "status_counts": dict(statuses),
        "reason_counts": dict(reasons),
        "subclusters": cluster_summary,
        "thresholds": {
            "dedup_iou": args.dedup_iou,
            "dedup_dino": args.dedup_dino,
            "knn_k": args.knn_k,
            "knn_conflict_ratio": args.knn_conflict_ratio,
            "knn_same_label_min": args.knn_same_label_min,
            "semantic_margin": args.semantic_margin,
            "max_subclusters": args.max_subclusters,
            "subcluster_min": args.subcluster_min,
            "same_label_knn_min": args.same_label_knn_min,
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.apply_payload:
        print("[*] Qdrant clean_v2 payload update")

        for i, r in enumerate(results, 1):
            client.set_payload(
                collection_name=args.collection,
                payload={
                    "clean_v2_status": r.status,
                    "clean_v2_reasons": r.reasons,
                    "clean_v2_suggested_label": r.suggested_label,
                    "clean_v2_duplicate_of": r.duplicate_of,
                    "clean_v2_knn_top_label": r.knn_top_label,
                    "clean_v2_knn_top_ratio": r.knn_top_ratio,
                    "clean_v2_knn_same_label_ratio": r.knn_same_label_ratio,
                    "clean_v2_siglip_best_label": r.siglip_best_label,
                    "clean_v2_siglip_margin": r.siglip_margin,
                    "clean_v2_subcluster_similarity": r.nearest_subcluster_similarity,
                    "clean_v2_subcluster_id": r.nearest_subcluster_id,
                    "clean_v2_same_label_knn_similarity": r.same_label_knn_similarity,
                    "clean_v2_outlier_score": r.outlier_score,
                },
                points=[r.point_id],
                wait=True,
            )

            if i % 250 == 0 or i == len(results):
                print(
                    f"      payload {i:,}/{len(results):,}"
                )

    print()
    print("=" * 80)
    print("COMPLETE")
    print(f"KEEP              : {statuses.get('KEEP', 0):,}")
    print(f"REVIEW            : {statuses.get('REVIEW', 0):,}")
    print(f"RELABEL_CANDIDATE : {statuses.get('RELABEL_CANDIDATE', 0):,}")
    print(f"REJECT            : {statuses.get('REJECT', 0):,}")
    print(f"Spatial duplicate : {dup_count:,}")
    print(f"JSON              : {json_path}")
    print(f"CSV               : {csv_path}")
    print(f"Summary           : {summary_path}")
    print("=" * 80)

    client.close()


if __name__ == "__main__":
    main()
