"""
Market1501 evaluation for the current SOLIDER adapter.

Evaluates:
- mAP
- Rank-1
- Rank-5
- Rank-10
- mINP

Market1501 protocol:
- query:              <data-root>/query
- gallery:            <data-root>/bounding_box_test
- junk pid == -1 is ignored
- same pid + same camera gallery images are removed for each query
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedders.human.solider_embedder import SoliderEmbedder


_MARKET_RE = re.compile(r"^(-?\d+)_c(\d+)")


def parse_market_name(path: Path) -> tuple[int, int]:
    m = _MARKET_RE.match(path.name)
    if m is None:
        raise ValueError(f"Market1501 파일명을 해석할 수 없습니다: {path.name}")

    return int(m.group(1)), int(m.group(2))


def load_split(folder: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    paths = sorted(folder.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"JPG 파일이 없습니다: {folder}")

    pids = []
    camids = []

    for path in paths:
        pid, camid = parse_market_name(path)
        pids.append(pid)
        camids.append(camid)

    return (
        paths,
        np.asarray(pids, dtype=np.int32),
        np.asarray(camids, dtype=np.int32),
    )


def embed_paths(embedder: SoliderEmbedder, paths: list[Path]) -> np.ndarray:
    vecs = embedder.embed_crops([str(p) for p in paths])
    vecs = np.asarray(vecs, dtype=np.float32)

    if vecs.ndim != 2 or vecs.shape[1] != embedder.DIM:
        raise RuntimeError(
            f"embedding shape 오류: expected=(N,{embedder.DIM}), actual={vecs.shape}"
        )

    return vecs


def evaluate_market1501(
    q_vecs: np.ndarray,
    q_pids: np.ndarray,
    q_camids: np.ndarray,
    g_vecs: np.ndarray,
    g_pids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 10,
) -> dict[str, float]:
    if len(g_pids) == 0:
        raise RuntimeError("gallery가 비어 있습니다.")

    max_rank = min(max_rank, len(g_pids))

    cmc_sum = np.zeros(max_rank, dtype=np.float64)
    ap_sum = 0.0
    inp_sum = 0.0
    valid_queries = 0

    # SoliderEmbedder output is L2-normalized, so dot product == cosine similarity.
    sim = q_vecs @ g_vecs.T

    for i in range(len(q_pids)):
        order = np.argsort(-sim[i])

        q_pid = q_pids[i]
        q_cam = q_camids[i]

        ordered_pids = g_pids[order]
        ordered_camids = g_camids[order]

        remove = (
            (ordered_pids == -1)
            | ((ordered_pids == q_pid) & (ordered_camids == q_cam))
        )

        matches = (ordered_pids[~remove] == q_pid).astype(np.int32)

        if not np.any(matches):
            continue

        valid_queries += 1

        # CMC
        first_hit = int(np.flatnonzero(matches)[0])
        if first_hit < max_rank:
            cmc_sum[first_hit:] += 1.0

        # AP
        hit_positions = np.flatnonzero(matches)
        precisions = (
            np.arange(1, len(hit_positions) + 1, dtype=np.float64)
            / (hit_positions + 1)
        )
        ap_sum += float(precisions.mean())

        # INP
        last_hit_rank = int(hit_positions[-1]) + 1
        inp_sum += float(matches.sum() / last_hit_rank)

    if valid_queries == 0:
        raise RuntimeError("평가 가능한 query가 없습니다.")

    cmc = cmc_sum / valid_queries

    def rank_value(rank: int) -> float:
        idx = min(rank, len(cmc)) - 1
        return float(cmc[idx])

    return {
        "mAP": ap_sum / valid_queries,
        "mINP": inp_sum / valid_queries,
        "Rank-1": rank_value(1),
        "Rank-5": rank_value(5),
        "Rank-10": rank_value(10),
        "valid_queries": float(valid_queries),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate current SOLIDER adapter on Market1501"
    )

    ap.add_argument("--data-root", required=True)
    ap.add_argument("--solider-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--backbone",
        default="swin_base",
        choices=["swin_tiny", "swin_small", "swin_base"],
    )
    ap.add_argument("--semantic-weight", type=float, default=0.2)
    ap.add_argument(
        "--neck-feat",
        default="before",
        choices=["before", "after"],
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)

    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    query_dir = data_root / "query"
    gallery_dir = data_root / "bounding_box_test"

    if not query_dir.is_dir():
        raise FileNotFoundError(f"query 폴더가 없습니다: {query_dir}")

    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"gallery 폴더가 없습니다: {gallery_dir}")

    q_paths, q_pids, q_camids = load_split(query_dir)
    g_paths, g_pids, g_camids = load_split(gallery_dir)

    print(f"query   : {len(q_paths)}")
    print(f"gallery : {len(g_paths)}")

    embedder = SoliderEmbedder(
        solider_root=args.solider_root,
        ckpt_path=args.ckpt,
        backbone=args.backbone,
        semantic_weight=args.semantic_weight,
        neck_feat=args.neck_feat,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("\n[1/2] Query embedding...")
    q_vecs = embed_paths(embedder, q_paths)

    print("[2/2] Gallery embedding...")
    g_vecs = embed_paths(embedder, g_paths)

    print("\nEvaluating Market1501...")
    result = evaluate_market1501(
        q_vecs=q_vecs,
        q_pids=q_pids,
        q_camids=q_camids,
        g_vecs=g_vecs,
        g_pids=g_pids,
        g_camids=g_camids,
        max_rank=10,
    )

    print("\n========================================")
    print(" Market1501 SOLIDER Evaluation")
    print("========================================")
    print(f"valid queries : {int(result['valid_queries'])}")
    print(f"mAP           : {result['mAP'] * 100:.2f}%")
    print(f"mINP          : {result['mINP'] * 100:.2f}%")
    print(f"Rank-1        : {result['Rank-1'] * 100:.2f}%")
    print(f"Rank-5        : {result['Rank-5'] * 100:.2f}%")
    print(f"Rank-10       : {result['Rank-10'] * 100:.2f}%")
    print("========================================")


if __name__ == "__main__":
    main()
