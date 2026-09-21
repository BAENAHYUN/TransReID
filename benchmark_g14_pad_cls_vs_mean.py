#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DINOv2 Giant+Registers + AR-Pad224: CLS vs Patch-Mean benchmark.

공통:
- model: facebook/dinov2-with-registers-giant
- input: aspect-ratio preserving resize -> 224x224 center pad
- padding RGB: ImageNet mean
- ImageNet normalization
- L2 normalization
- register tokens are excluded from patch mean

비교:
A = CLS, 1536D
B = Patch Mean, 1536D

Manifest CSV:
group_id,role,path
obj001,query,C:\...\query.jpg
obj001,positive,C:\...\same1.jpg
obj001,negative,C:\...\other1.jpg

Run:
python .\benchmark_g14_pad_cls_vs_mean.py `
  --manifest .\dinov2_benchmark_manifest.csv `
  --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel
import torchvision.transforms as T

MODEL_ID = "facebook/dinov2-with-registers-giant"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN_RGB = tuple(int(round(v * 255)) for v in IMAGENET_MEAN)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def load_manifest(path: Path):
    groups = defaultdict(lambda: {"query": [], "positive": [], "negative": []})
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"group_id", "role", "path"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"manifest columns must include {sorted(required)}")
        for row in reader:
            gid = str(row["group_id"]).strip()
            role = str(row["role"]).strip().lower()
            raw = str(row["path"]).strip()
            if not gid or role not in {"query", "positive", "negative"} or not raw:
                continue

            p = Path(raw).expanduser()
            p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()

            if not p.is_file():
                raise FileNotFoundError(p)
            if p.suffix.lower() not in EXTS:
                raise ValueError(f"unsupported image: {p}")

            groups[gid][role].append(p)

    if not groups:
        raise ValueError("manifest is empty")

    for gid, g in groups.items():
        if len(g["query"]) != 1:
            raise ValueError(f"{gid}: query must be exactly 1")
        if not g["positive"]:
            raise ValueError(f"{gid}: positive >= 1 required")
        if not g["negative"]:
            raise ValueError(f"{gid}: negative >= 1 required")

    return dict(groups)

def to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")

class Pad224:
    def __init__(self):
        self.norm = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = to_rgb(img)
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid image size: {(w, h)}")

        scale = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
        nw = max(1, min(IMAGE_SIZE, int(round(w * scale))))
        nh = max(1, min(IMAGE_SIZE, int(round(h * scale))))

        resized = img.resize((nw, nh), Image.Resampling.BICUBIC)

        canvas = Image.new(
            "RGB",
            (IMAGE_SIZE, IMAGE_SIZE),
            IMAGENET_MEAN_RGB,
        )

        left = (IMAGE_SIZE - nw) // 2
        top = (IMAGE_SIZE - nh) // 2
        canvas.paste(resized, (left, top))

        return self.norm(canvas)

def choose_device(raw):
    return torch.device(raw) if raw else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

def choose_dtype(device, fp32):
    if fp32 or device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

@torch.inference_mode()
def embed_paths(
    model,
    paths,
    preprocess,
    *,
    feature,
    num_prefix_tokens,
    device,
    dtype,
    batch_size,
):
    outs = []

    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]

        tensors = []
        for p in chunk:
            with Image.open(p) as im:
                tensors.append(preprocess(im.copy()))

        batch = torch.stack(tensors).to(device=device, dtype=dtype)

        tokens = model(pixel_values=batch).last_hidden_state

        if feature == "cls":
            feat = tokens[:, 0]

        elif feature == "mean":
            # [CLS][REG x N][PATCH ...]
            # register tokens must not be included in patch mean.
            patch_tokens = tokens[:, num_prefix_tokens:]
            feat = patch_tokens.mean(dim=1)

        else:
            raise ValueError(feature)

        feat = F.normalize(feat.float(), dim=-1)
        outs.append(feat.cpu().numpy())

        print(
            f"  {feature:4s} embedded "
            f"{min(start + batch_size, len(paths)):,}/{len(paths):,}",
            end="\r",
        )

    print()
    return np.concatenate(outs, axis=0).astype(np.float32, copy=False)

def group_metrics(q, pos, neg):
    ps = pos @ q
    ns = neg @ q

    labels = np.concatenate([
        np.ones(len(ps), dtype=np.int32),
        np.zeros(len(ns), dtype=np.int32),
    ])
    scores = np.concatenate([ps, ns])

    order = np.argsort(-scores)
    ranked = labels[order]

    positive_ranks = np.flatnonzero(ranked == 1) + 1
    first = int(positive_ranks[0])

    return {
        "R@1": float(first <= 1),
        "R@5": float(first <= 5),
        "MRR": float(1.0 / first),
        "first_positive_rank": first,
        "positive_mean": float(ps.mean()),
        "negative_mean": float(ns.mean()),
        "separation_gap": float(ps.mean() - ns.mean()),
        "strict_margin": float(ps.min() - ns.max()),
        "top1_positive": bool(ranked[0] == 1),
    }

def aggregate(per_group):
    vals = list(per_group.values())

    return {
        "groups": len(vals),
        "R@1": float(np.mean([v["R@1"] for v in vals])),
        "R@5": float(np.mean([v["R@5"] for v in vals])),
        "MRR": float(np.mean([v["MRR"] for v in vals])),
        "positive_mean": float(np.mean([v["positive_mean"] for v in vals])),
        "negative_mean": float(np.mean([v["negative_mean"] for v in vals])),
        "separation_gap_mean": float(
            np.mean([v["separation_gap"] for v in vals])
        ),
        "strict_margin_mean": float(
            np.mean([v["strict_margin"] for v in vals])
        ),
        "strict_margin_positive_rate": float(
            np.mean([v["strict_margin"] > 0 for v in vals])
        ),
    }

def evaluate(groups, vec_map):
    out = {}

    for gid, g in groups.items():
        q = vec_map[str(g["query"][0])]
        pos = np.stack([vec_map[str(p)] for p in g["positive"]])
        neg = np.stack([vec_map[str(p)] for p in g["negative"]])

        out[gid] = group_metrics(q, pos, neg)

    return out

def score_tuple(a):
    return (
        a["R@1"],
        a["R@5"],
        a["MRR"],
        a["separation_gap_mean"],
        a["strict_margin_mean"],
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--output-dir",
        default="outputs/dinov2_g14_pad_cls_vs_mean",
    )

    args = ap.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    groups = load_manifest(manifest)

    all_paths = []
    seen = set()

    for g in groups.values():
        for role in ("query", "positive", "negative"):
            for p in g[role]:
                sp = str(p)
                if sp not in seen:
                    seen.add(sp)
                    all_paths.append(p)

    device = choose_device(args.device)
    dtype = choose_dtype(device, args.fp32)

    print("=" * 88)
    print("DINOv2 G/14 REGISTERS — AR-PAD224 CLS vs MEAN")
    print("=" * 88)
    print("model        :", MODEL_ID)
    print("groups       :", len(groups))
    print("unique images:", len(all_paths))
    print("device       :", device)
    print("dtype        :", dtype)
    print("padding RGB  :", IMAGENET_MEAN_RGB)

    kwargs = {}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    if args.local_files_only:
        kwargs["local_files_only"] = True

    print("\n[LOAD MODEL]")
    model = AutoModel.from_pretrained(MODEL_ID, **kwargs)
    model = model.to(device=device, dtype=dtype).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    hidden = int(model.config.hidden_size)
    regs = int(getattr(model.config, "num_register_tokens", 0))
    patch = int(getattr(model.config, "patch_size", 14))
    num_prefix_tokens = 1 + regs

    print("hidden dim   :", hidden)
    print("registers    :", regs)
    print("patch size   :", patch)
    print("prefix tokens:", num_prefix_tokens)

    if hidden != 1536:
        raise RuntimeError(f"expected giant 1536-d, got {hidden}")
    if patch != 14:
        raise RuntimeError(f"expected patch size 14, got {patch}")

    # 224 / 14 = 16 -> 256 patch tokens expected.
    with torch.inference_mode():
        dummy = torch.zeros(
            1, 3, IMAGE_SIZE, IMAGE_SIZE,
            device=device,
            dtype=dtype,
        )
        seq = int(
            model(pixel_values=dummy)
            .last_hidden_state
            .shape[1]
        )

    expected_patch_tokens = (IMAGE_SIZE // patch) ** 2
    actual_prefix_tokens = seq - expected_patch_tokens

    if actual_prefix_tokens != num_prefix_tokens:
        raise RuntimeError(
            "token layout mismatch: "
            f"seq={seq}, patch={expected_patch_tokens}, "
            f"actual_prefix={actual_prefix_tokens}, "
            f"config_prefix={num_prefix_tokens}"
        )

    print("token layout  : verified")

    preprocess = Pad224()
    results = {}

    for feature in ("cls", "mean"):
        print(f"\n[RUN] pad224 + {feature.upper()}")

        vecs = embed_paths(
            model,
            all_paths,
            preprocess,
            feature=feature,
            num_prefix_tokens=num_prefix_tokens,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
        )

        if vecs.shape[1] != 1536:
            raise RuntimeError(
                f"{feature}: expected 1536D, got {vecs.shape}"
            )

        if not np.isfinite(vecs).all():
            raise RuntimeError(f"{feature}: NaN/Inf found")

        norms = np.linalg.norm(vecs, axis=1)

        print("  dim       :", vecs.shape[1])
        print("  norm mean :", float(norms.mean()))

        vec_map = {
            str(p): v
            for p, v in zip(all_paths, vecs)
        }

        per_group = evaluate(groups, vec_map)
        agg = aggregate(per_group)

        results[feature] = {
            "aggregate": agg,
            "per_group": per_group,
        }

        print(
            f"  R@1={agg['R@1']:.4f} "
            f"R@5={agg['R@5']:.4f} "
            f"MRR={agg['MRR']:.4f}"
        )
        print(
            f"  gap={agg['separation_gap_mean']:.6f} "
            f"margin={agg['strict_margin_mean']:.6f}"
        )

    cls = results["cls"]["aggregate"]
    mean = results["mean"]["aggregate"]

    winner = (
        "mean"
        if score_tuple(mean) > score_tuple(cls)
        else "cls"
    )

    keys = [
        "R@1",
        "R@5",
        "MRR",
        "positive_mean",
        "negative_mean",
        "separation_gap_mean",
        "strict_margin_mean",
        "strict_margin_positive_rate",
    ]

    delta = {
        k: float(mean[k] - cls[k])
        for k in keys
    }

    out = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_id": MODEL_ID,
        "hidden_dim": hidden,
        "num_register_tokens": regs,
        "patch_size": patch,
        "input": "AR preserve resize -> center pad 224 -> ImageNet normalize",
        "padding_rgb": IMAGENET_MEAN_RGB,
        "manifest": str(manifest),
        "groups": len(groups),
        "unique_images": len(all_paths),
        "results": results,
        "delta_mean_minus_cls": delta,
        "winner": winner,
    }

    (out / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "DINOv2 Giant+Registers — AR-Pad224 CLS vs Mean",
        "=" * 72,
        (
            f"CLS : R@1={cls['R@1']:.4f} "
            f"R@5={cls['R@5']:.4f} "
            f"MRR={cls['MRR']:.4f} "
            f"gap={cls['separation_gap_mean']:.6f} "
            f"margin={cls['strict_margin_mean']:.6f}"
        ),
        (
            f"MEAN: R@1={mean['R@1']:.4f} "
            f"R@5={mean['R@5']:.4f} "
            f"MRR={mean['MRR']:.4f} "
            f"gap={mean['separation_gap_mean']:.6f} "
            f"margin={mean['strict_margin_mean']:.6f}"
        ),
        "",
        "MEAN - CLS:",
        *[
            f"  {k}: {v:+.6f}"
            for k, v in delta.items()
        ],
        "",
        f"WINNER: {winner}",
    ]

    (out / "summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("FINAL")
    print("=" * 88)
    print(
        "CLS  R@1/R@5/MRR :",
        f"{cls['R@1']:.4f} / "
        f"{cls['R@5']:.4f} / "
        f"{cls['MRR']:.4f}",
    )
    print(
        "MEAN R@1/R@5/MRR :",
        f"{mean['R@1']:.4f} / "
        f"{mean['R@5']:.4f} / "
        f"{mean['MRR']:.4f}",
    )
    print(
        "delta MEAN-CLS gap           :",
        f"{delta['separation_gap_mean']:+.6f}",
    )
    print(
        "delta MEAN-CLS strict margin :",
        f"{delta['strict_margin_mean']:+.6f}",
    )
    print("WINNER                       :", winner)
    print("output                       :", out)

if __name__ == "__main__":
    main()
