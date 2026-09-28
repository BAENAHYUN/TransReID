"""
PRW GT 기반 임베딩 단독 Retrieval 평가
======================================
PRW 데이터셋의 GT bbox를 그대로 crop해서 gallery/query 임베딩만 비교한다.
검출(detection)/localization은 개입하지 않고 embedding retrieval만 측정하므로,
공식 PRW **Person Search** 프로토콜(query -> 전체 scene에서 검출부터 다시
수행하는 설정, Zheng et al. CVPR 2017 / Munjal et al. CVPR 2019 Query-Guided
End-to-End Person Search)과는 다르다. 여기서는 검출 오차를 배제하고 임베딩
모델 자체의 retrieval 성능만 비교하려는 목적이라 이 방식이 맞다 — 논문에
보고된 PRW Person Search 수치와 이 결과를 직접 비교하지 않을 것.

Gallery : frame_test.mat 의 6,112 test 프레임 x GT annotation bbox 크롭
           (pid == -2 는 junk -> 제외)
Query   : query_box/ 의 2,057 pre-cropped 이미지
Junk    : 같은 frame 에 같은 pid 인 gallery 항목 (동일 검출 제외)
Metric  : mAP / Rank-1 / Rank-5 (camera 제외 없음)

지원 모델
---------
  --model irra     IRRA ViT-B/16  (512-d)
  --model solider  SOLIDER Swin-B (1024-d)
  --model siglip2  SiGLIP2        (768-d)

사용 예
-------
  python eval/prw_eval.py --model irra --data-root ./data/PRW
  python eval/prw_eval.py --model solider --data-root ./data/PRW
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import scipy.io
except ImportError:
    print("[ERROR] scipy 가 없습니다: pip install scipy")
    sys.exit(1)

CAM_RE = re.compile(r"^c(\d+)")


def cam_from_frame(frame_name: str) -> int:
    m = CAM_RE.match(frame_name)
    return int(m.group(1)) if m else -1


def load_frame_list(mat_path: Path) -> List[str]:
    d = scipy.io.loadmat(str(mat_path))
    key = [k for k in d if not k.startswith("_")][0]
    arr = d[key].flatten()
    return [str(x[0]) if hasattr(x, "__len__") else str(x) for x in arr]


def load_annotation(ann_path: Path) -> np.ndarray:
    try:
        d = scipy.io.loadmat(str(ann_path))
        key = [k for k in d if not k.startswith("_")][0]
        arr = np.asarray(d[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr  # (N, 5): [pid, x, y, w, h]
    except Exception:
        return np.empty((0, 5), dtype=np.float32)


def safe_crop(img: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image:
    iw, ih = img.size
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(iw, int(x + w))
    y2 = min(ih, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return img.crop((0, 0, 64, 128))
    return img.crop((x1, y1, x2, y2))


def build_gallery(
    frame_names: List[str],
    frames_dir: Path,
    ann_dir: Path,
) -> Tuple[List[Image.Image], np.ndarray, np.ndarray, List[str]]:
    crops, pids, camids, frame_tags = [], [], [], []
    for i, frame_name in enumerate(frame_names):
        img_path = frames_dir / f"{frame_name}.jpg"
        ann_path = ann_dir / f"{frame_name}.jpg.mat"
        if not img_path.exists() or not ann_path.exists():
            continue
        ann = load_annotation(ann_path)
        if ann.shape[0] == 0:
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        cam = cam_from_frame(frame_name)
        for row in ann:
            pid_raw = int(row[0])
            if pid_raw == -2:
                continue
            x, y, w, h = row[1], row[2], row[3], row[4]
            crop = safe_crop(img, x, y, w, h)
            crops.append(crop)
            pids.append(pid_raw)
            camids.append(cam)
            frame_tags.append(frame_name)
        if (i + 1) % 500 == 0:
            print(f"  gallery frames: {i+1}/{len(frame_names)}", end="\r", flush=True)
    print()
    return (
        crops,
        np.asarray(pids, dtype=np.int32),
        np.asarray(camids, dtype=np.int32),
        frame_tags,
    )


def build_query(
    query_box_dir: Path,
    query_info_path: Path,
) -> Tuple[List[Image.Image], np.ndarray, np.ndarray, List[str]]:
    crops, pids, camids, frame_tags = [], [], [], []
    with open(query_info_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            pid = int(parts[0])
            frame_name = parts[5]
            img_path = query_box_dir / f"{pid}_{frame_name}.jpg"
            if not img_path.exists():
                continue
            try:
                crop = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            cam = cam_from_frame(frame_name)
            crops.append(crop)
            pids.append(pid)
            camids.append(cam)
            frame_tags.append(frame_name)
    return (
        crops,
        np.asarray(pids, dtype=np.int32),
        np.asarray(camids, dtype=np.int32),
        frame_tags,
    )


def embed_pil(embedder, images: List[Image.Image], batch_size: int = 64) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        vecs = embedder.embed_crops(batch)
        all_vecs.append(np.asarray(vecs, dtype=np.float32))
        if (i // batch_size) % 5 == 0:
            print(f"  {i + len(batch)}/{len(images)}", end="\r", flush=True)
    print()
    return np.vstack(all_vecs) if all_vecs else np.empty((0, embedder.DIM), dtype=np.float32)


def evaluate_prw(
    q_vecs: np.ndarray,
    q_pids: np.ndarray,
    q_frames: List[str],
    g_vecs: np.ndarray,
    g_pids: np.ndarray,
    g_frames: List[str],
    max_rank: int = 10,
) -> dict:
    sim = q_vecs @ g_vecs.T  # (Q, G)
    cmc_sum = np.zeros(max_rank, dtype=np.float64)
    ap_sum = 0.0
    valid = 0
    g_pids_arr = np.asarray(g_pids)
    g_frames_arr = np.asarray(g_frames)
    for qi in range(len(q_pids)):
        qpid = q_pids[qi]
        qframe = q_frames[qi]
        junk_mask = (g_pids_arr == qpid) & (g_frames_arr == qframe)
        n_pos = ((g_pids_arr == qpid) & (g_frames_arr != qframe)).sum()
        if n_pos == 0:
            continue
        valid += 1
        scores = sim[qi]
        sorted_idx = np.argsort(-scores)
        sorted_idx_clean = [idx for idx in sorted_idx if not junk_mask[idx]]
        for rank, idx in enumerate(sorted_idx_clean[:max_rank]):
            if g_pids_arr[idx] == qpid:
                cmc_sum[rank:] += 1.0
                break
        hits, ap = 0, 0.0
        for rank, idx in enumerate(sorted_idx_clean):
            if g_pids_arr[idx] == qpid:
                hits += 1
                ap += hits / (rank + 1)
                if hits == n_pos:
                    break
        ap_sum += ap / n_pos
    if valid == 0:
        raise RuntimeError("유효한 query가 없습니다.")
    cmc = cmc_sum / valid
    def r(k):
        return float(cmc[min(k, max_rank) - 1])
    return {
        "mAP": ap_sum / valid,
        "Rank-1": r(1),
        "Rank-5": r(5),
        "Rank-10": r(10),
        "valid_queries": valid,
    }


def load_embedder(model: str, args):
    """pipeline.yaml(retrievers)에서 실제 DB 구축에 쓴 model_id/checkpoint를
    읽어와 기본값으로 쓴다. 여기서 하드코딩된 리터럴을 따로 두면 pipeline.yaml
    이 바뀌었을 때 이 평가만 조용히 다른 모델/체크포인트를 쓰게 된다 --
    실제로 SigLIP2 는 클래스 기본값이 so400m(1152d)인데 pipeline.yaml/DB는
    base(768d)라서 차원 자체가 달랐고, SOLIDER 는 solider_root 기본값이
    존재하지 않는 경로(PROJECT_ROOT/'SOLIDER')였다 (실제는 third_party/SOLIDER).
    CLI 인자를 명시하면 그게 최우선이고, 없으면 pipeline.yaml 값을 쓴다.
    """
    from config import PipelineConfig
    cfg = PipelineConfig.load(args.config)
    if model not in cfg.retrievers:
        raise ValueError(
            f"pipeline.yaml(retrievers)에 '{model}' 이 없습니다. --config 를 확인하세요."
        )
    p = cfg.retrievers[model].params  # PipelineConfig.load 가 이미 절대경로로 resolve

    # CLI 로 checkpoint/옵션을 덮어쓰지 않았으면 yaml 의 module/class 그대로 만든다 (registry 경로) —
    # 그래야 yaml 로 임베더를 교체·추가했을 때 이 단독 평가도 같은 모델을 쓴다 (DB 구축·검색과 같은 로더).
    explicit = any(getattr(args, k, None) for k in ("irra_root", "irra_ckpt", "irra_cfg", "clip_pt", "solider_root", "solider_ckpt"))
    if model == "solider" and (args.backbone != "swin_base" or float(args.semantic_weight) != 0.2 or args.neck_feat != "before"):
        explicit = True
    if not explicit:
        from registry import EmbedderRegistry
        emb = EmbedderRegistry(cfg).get(model)
        try:
            emb._prw_eval_loader = "registry"   # 원장 params.loader 에 기록
        except Exception:
            pass
        return emb

    if model == "irra":
        from embedders.human.irra_embedder import IRRAEmbedder
        irra_root = args.irra_root or p.get("irra_root", str(PROJECT_ROOT / "IRRA"))
        ckpt = args.irra_ckpt or p.get("ckpt_path", str(PROJECT_ROOT / "weights/IRRA/cuhk_pedes/best.pth"))
        irra_cfg = args.irra_cfg or p.get("config_file", str(PROJECT_ROOT / "weights/IRRA/cuhk_pedes/configs.yaml"))
        clip_pt = args.clip_pt or p.get("clip_pretrained", str(PROJECT_ROOT / "weights/IRRA/ViT-B-16.pt"))
        return IRRAEmbedder(
            irra_root=irra_root,
            ckpt_path=ckpt,
            config_file=irra_cfg,
            clip_pretrained=clip_pt if Path(clip_pt).exists() else None,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif model == "solider":
        from embedders.human.solider_embedder import SoliderEmbedder
        solider_root = args.solider_root or p.get("solider_root", str(PROJECT_ROOT / "third_party/SOLIDER"))
        ckpt = args.solider_ckpt or p.get("ckpt_path", str(PROJECT_ROOT / "weights/SOLIDER/solider_market_swin_base.pth"))
        return SoliderEmbedder(
            solider_root=solider_root,
            ckpt_path=ckpt,
            backbone=p.get("backbone", args.backbone),
            semantic_weight=float(p.get("semantic_weight", args.semantic_weight)),
            neck_feat=p.get("neck_feat", args.neck_feat),
            device=args.device,
            batch_size=args.batch_size,
        )
    elif model == "siglip2":
        from embedders.siglip2_embedder import SigLIP2Embedder
        return SigLIP2Embedder(
            model_id=p.get("model_id", "google/siglip2-base-patch16-naflex"),
            max_num_patches=int(p.get("max_num_patches", 256)),
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"알 수 없는 모델: {model}")


def main():
    ap = argparse.ArgumentParser(description="PRW Person Retrieval Evaluation")
    ap.add_argument("--config", default="pipeline.yaml",
                    help="model_id/checkpoint 기본값을 여기서 읽는다 "
                         "(실제 forensic DB 를 구축한 그 설정)")
    ap.add_argument("--data-root", default="./data/PRW")
    ap.add_argument("--model", default="irra", help="pipeline.yaml retrievers 의 이름 (irra / solider / siglip2 또는 yaml 에 등록한 새 임베더)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--irra-root", default=None)
    ap.add_argument("--irra-ckpt", default=None)
    ap.add_argument("--irra-cfg",  default=None)
    ap.add_argument("--clip-pt",   default=None)
    ap.add_argument("--solider-root", default=None)
    ap.add_argument("--solider-ckpt", default=None)
    ap.add_argument("--backbone", default="swin_base", choices=["swin_tiny","swin_small","swin_base"])
    ap.add_argument("--semantic-weight", type=float, default=0.2)
    ap.add_argument("--neck-feat", default="before", choices=["before","after"])
    ap.add_argument("--save-result", default=None)
    ap.add_argument("--ledger", default=None, help="실행 원장 JSONL (기본 bench/ledger.jsonl; '' 또는 none 이면 기록 안 함)")
    ap.add_argument("--no-ledger", action="store_true", help="원장(bench/ledger.jsonl)에 기록하지 않음")
    args = ap.parse_args()

    data_root  = Path(args.data_root).expanduser().resolve()
    frames_dir = data_root / "frames"
    ann_dir    = data_root / "annotations"
    qbox_dir   = data_root / "query_box"
    qinfo_path = data_root / "query_info.txt"
    ft_path    = data_root / "frame_test.mat"

    for p in [frames_dir, ann_dir, qbox_dir, qinfo_path, ft_path]:
        if not p.exists():
            print(f"[ERROR] 경로가 없습니다: {p}")
            sys.exit(1)

    print("=" * 60)
    print(f" PRW Evaluation  |  model={args.model}")
    print("=" * 60)

    print("\n[1/4] Loading gallery (GT crops from test frames)...")
    t0 = time.time()
    test_frames = load_frame_list(ft_path)
    g_crops, g_pids, g_camids, g_frames = build_gallery(test_frames, frames_dir, ann_dir)
    print(f"  gallery crops  : {len(g_crops)}")
    print(f"  unique persons : {len(set(g_pids))}")
    print(f"  elapsed: {time.time()-t0:.1f}s")

    print("\n[2/4] Loading query images...")
    q_crops, q_pids, q_camids, q_frames = build_query(qbox_dir, qinfo_path)
    print(f"  queries        : {len(q_crops)}")
    print(f"  unique persons : {len(set(q_pids))}")

    print(f"\n[3/4] Loading embedder: {args.model}...")
    embedder = load_embedder(args.model, args)
    print(f"  DIM = {embedder.DIM}")

    print(f"\n  Embedding gallery ({len(g_crops)})...")
    t1 = time.time()
    g_vecs = embed_pil(embedder, g_crops, args.batch_size)
    print(f"  done ({time.time()-t1:.1f}s)")

    print(f"\n  Embedding queries ({len(q_crops)})...")
    t1 = time.time()
    q_vecs = embed_pil(embedder, q_crops, args.batch_size)
    print(f"  done ({time.time()-t1:.1f}s)")

    print("\n[4/4] Evaluating...")
    result = evaluate_prw(
        q_vecs=q_vecs, q_pids=q_pids, q_frames=q_frames,
        g_vecs=g_vecs, g_pids=g_pids, g_frames=g_frames,
        max_rank=10,
    )

    print("\n" + "=" * 60)
    print(f"  PRW Person Retrieval [{args.model.upper()}]")
    print("=" * 60)
    print(f"  valid queries  : {int(result['valid_queries'])} / {len(q_crops)}")
    print(f"  gallery size   : {len(g_crops)}")
    print(f"  mAP            : {result['mAP']*100:.2f}%")
    print(f"  Rank-1         : {result['Rank-1']*100:.2f}%")
    print(f"  Rank-5         : {result['Rank-5']*100:.2f}%")
    print(f"  Rank-10        : {result['Rank-10']*100:.2f}%")
    print("=" * 60)

    import json
    out = {
        "model": args.model, "gallery_size": len(g_crops),
        "query_total": len(q_crops), "valid_queries": int(result["valid_queries"]),
        "mAP": round(result["mAP"]*100, 4),
        "Rank-1": round(result["Rank-1"]*100, 4),
        "Rank-5": round(result["Rank-5"]*100, 4),
        "Rank-10": round(result["Rank-10"]*100, 4),
    }
    if args.save_result:
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {args.save_result}")

    # 실행 원장 (bench/ledger.jsonl) — 모델·설정·지표를 한 줄로
    from bench import ledger
    params = {"batch_size": args.batch_size, "loader": getattr(embedder, "_prw_eval_loader", "legacy")}
    if args.model == "solider":
        params.update(backbone=args.backbone, semantic_weight=args.semantic_weight, neck_feat=args.neck_feat)
    for key in ("irra_root", "irra_ckpt", "irra_cfg", "clip_pt", "solider_root", "solider_ckpt"):
        if getattr(args, key, None):
            params[key] = getattr(args, key)
    config_path = Path(args.config).expanduser().resolve()
    ledger.record(lambda: [ledger.entry_from_embedding_result(
        out, args.model, params=params, versions=ledger.versions_info(),
        config={"path": str(config_path), "sha256": ledger.file_sha256(config_path)},
        report=args.save_result, command=sys.argv[1:])], args.ledger, args.no_ledger)


if __name__ == "__main__":
    main()
