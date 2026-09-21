#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final DINOv2 Giant+Registers pooling benchmark on AR-Pad224.

공통:
- model: facebook/dinov2-with-registers-giant
- aspect-ratio preserving resize -> 224x224 center pad
- padding RGB = ImageNet mean
- ImageNet normalization
- register tokens excluded from patch pooling
- final L2 normalization

비교:
1) CLS               : 현재 최강 baseline
2) SignedPower p=2   : ViT 음수 특징을 보존하는 power pooling 실험
3) SignedPower p=3   : 더 강한 peak weighting

주의:
SignedPower는 원래 CNN GeM을 그대로 쓴 것이 아니라,
ViT patch token의 음수 성분을 보존하도록 만든 실험용 pooling이다.

실행:
python .\benchmark_g14_pad_cls_vs_signedpower.py `
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_manifest(path: Path):
    groups = defaultdict(lambda: {"query": [], "positive": [], "negative": []})

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"group_id", "role", "path"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"manifest columns must include {sorted(required)}"
            )

        for row in reader:
            gid = str(row["group_id"]).strip()
            role = str(row["role"]).strip().lower()
            raw = str(row["path"]).strip()

            if (
                not gid
                or role not in {"query", "positive", "negative"}
                or not raw
            ):
                continue

            p = Path(raw).expanduser()
            p = (
                (path.parent / p).resolve()
                if not p.is_absolute()
                else p.resolve()
            )

            if not p.is_file():
                raise FileNotFoundError(p)

            if p.suffix.lower() not in IMAGE_EXTS:
                raise ValueError(f"unsupported image: {p}")

            groups[gid][role].append(p)

    if not groups:
        raise ValueError("manifest is empty")

    for gid, g in groups.items():
        if len(g["query"]) != 1:
            raise ValueError(
                f"{gid}: query must be exactly 1, got {len(g['query'])}"
            )

        if not g["positive"]:
            raise ValueError(f"{gid}: positive >= 1 required")

        if not g["negative"]:
            raise ValueError(f"{gid}: negative >= 1 required")

    return dict(groups)


def to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


class Pad224:
    def __init__(self):
        self.tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = to_rgb(img)

        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid image size: {(w, h)}")

        scale = min(IMAGE_SIZE / w, IMAGE_SIZE / h)

        new_w = max(
            1,
            min(
                IMAGE_SIZE,
                int(round(w * scale)),
            ),
        )

        new_h = max(
            1,
            min(
                IMAGE_SIZE,
                int(round(h * scale)),
            ),
        )

        resized = img.resize(
            (new_w, new_h),
            Image.Resampling.BICUBIC,
        )

        canvas = Image.new(
            "RGB",
            (IMAGE_SIZE, IMAGE_SIZE),
            IMAGENET_MEAN_RGB,
        )

        left = (IMAGE_SIZE - new_w) // 2
        top = (IMAGE_SIZE - new_h) // 2

        canvas.paste(
            resized,
            (left, top),
        )

        return self.tensor_norm(canvas)


def choose_device(raw: str | None) -> torch.device:
    if raw:
        return torch.device(raw)

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def choose_dtype(
    device: torch.device,
    fp32: bool,
) -> torch.dtype:
    if fp32 or device.type != "cuda":
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def signed_power_pool(
    patch_tokens: torch.Tensor,
    p: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    patch_tokens: [B, N, D]

    sign-preserving generalized power mean:
      y = mean(sign(x) * |x|^p)
      z = sign(y) * |y|^(1/p)

    음수 feature 성분을 0으로 clamp하지 않는다.
    """
    x = patch_tokens.float()

    powered = (
        torch.sign(x)
        * torch.clamp(
            torch.abs(x),
            min=eps,
        ).pow(p)
    )

    pooled = powered.mean(dim=1)

    out = (
        torch.sign(pooled)
        * torch.clamp(
            torch.abs(pooled),
            min=eps,
        ).pow(1.0 / p)
    )

    return out


@torch.inference_mode()
def embed_paths(
    model,
    paths,
    preprocess,
    *,
    mode: str,
    num_prefix_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:

    outputs = []

    for start in range(
        0,
        len(paths),
        batch_size,
    ):
        chunk = paths[
            start:start + batch_size
        ]

        tensors = []

        for path in chunk:
            with Image.open(path) as image:
                tensors.append(
                    preprocess(
                        image.copy()
                    )
                )

        batch = torch.stack(
            tensors
        ).to(
            device=device,
            dtype=dtype,
        )

        tokens = model(
            pixel_values=batch
        ).last_hidden_state

        if mode == "cls":
            feat = tokens[:, 0].float()

        else:
            patch_tokens = tokens[
                :,
                num_prefix_tokens:,
            ]

            if mode == "signed_p2":
                feat = signed_power_pool(
                    patch_tokens,
                    p=2.0,
                )

            elif mode == "signed_p3":
                feat = signed_power_pool(
                    patch_tokens,
                    p=3.0,
                )

            else:
                raise ValueError(
                    f"unknown mode: {mode}"
                )

        feat = F.normalize(
            feat,
            dim=-1,
        )

        outputs.append(
            feat.cpu().numpy()
        )

        print(
            f"  {mode:10s} embedded "
            f"{min(start + batch_size, len(paths)):,}"
            f"/{len(paths):,}",
            end="\r",
        )

    print()

    return np.concatenate(
        outputs,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )


def group_metrics(
    query,
    positives,
    negatives,
):
    pos_scores = positives @ query
    neg_scores = negatives @ query

    labels = np.concatenate([
        np.ones(
            len(pos_scores),
            dtype=np.int32,
        ),
        np.zeros(
            len(neg_scores),
            dtype=np.int32,
        ),
    ])

    scores = np.concatenate([
        pos_scores,
        neg_scores,
    ])

    order = np.argsort(
        -scores
    )

    ranked_labels = labels[
        order
    ]

    positive_ranks = (
        np.flatnonzero(
            ranked_labels == 1
        )
        + 1
    )

    first_rank = int(
        positive_ranks[0]
    )

    return {
        "R@1": float(
            first_rank <= 1
        ),
        "R@5": float(
            first_rank <= 5
        ),
        "MRR": float(
            1.0 / first_rank
        ),
        "first_positive_rank":
            first_rank,
        "positive_mean": float(
            pos_scores.mean()
        ),
        "negative_mean": float(
            neg_scores.mean()
        ),
        "separation_gap": float(
            pos_scores.mean()
            - neg_scores.mean()
        ),
        "strict_margin": float(
            pos_scores.min()
            - neg_scores.max()
        ),
        "top1_positive": bool(
            ranked_labels[0] == 1
        ),
    }


def aggregate(
    per_group,
):
    values = list(
        per_group.values()
    )

    return {
        "groups": len(values),

        "R@1": float(
            np.mean([
                x["R@1"]
                for x in values
            ])
        ),

        "R@5": float(
            np.mean([
                x["R@5"]
                for x in values
            ])
        ),

        "MRR": float(
            np.mean([
                x["MRR"]
                for x in values
            ])
        ),

        "positive_mean": float(
            np.mean([
                x["positive_mean"]
                for x in values
            ])
        ),

        "negative_mean": float(
            np.mean([
                x["negative_mean"]
                for x in values
            ])
        ),

        "separation_gap_mean": float(
            np.mean([
                x["separation_gap"]
                for x in values
            ])
        ),

        "strict_margin_mean": float(
            np.mean([
                x["strict_margin"]
                for x in values
            ])
        ),

        "strict_margin_positive_rate":
            float(
                np.mean([
                    x["strict_margin"] > 0
                    for x in values
                ])
            ),
    }


def evaluate(
    groups,
    vector_map,
):
    out = {}

    for gid, group in groups.items():

        query = vector_map[
            str(group["query"][0])
        ]

        positives = np.stack([
            vector_map[str(path)]
            for path in group["positive"]
        ])

        negatives = np.stack([
            vector_map[str(path)]
            for path in group["negative"]
        ])

        out[gid] = group_metrics(
            query,
            positives,
            negatives,
        )

    return out


def score_tuple(
    aggregate_result,
):
    return (
        aggregate_result["R@1"],
        aggregate_result["R@5"],
        aggregate_result["MRR"],
        aggregate_result[
            "separation_gap_mean"
        ],
        aggregate_result[
            "strict_margin_mean"
        ],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--fp32",
        action="store_true",
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "dinov2_g14_pad_"
            "cls_vs_signedpower"
        ),
    )

    args = parser.parse_args()

    manifest = Path(
        args.manifest
    ).expanduser().resolve()

    groups = load_manifest(
        manifest
    )

    all_paths = []
    seen = set()

    for group in groups.values():
        for role in (
            "query",
            "positive",
            "negative",
        ):
            for path in group[role]:

                key = str(path)

                if key not in seen:
                    seen.add(key)
                    all_paths.append(path)

    device = choose_device(
        args.device
    )

    dtype = choose_dtype(
        device,
        args.fp32,
    )

    print("=" * 92)
    print(
        "DINOv2 G/14 REGISTERS "
        "AR-PAD224 — CLS vs SIGNED POWER"
    )
    print("=" * 92)

    print("model        :", MODEL_ID)
    print("groups       :", len(groups))
    print("unique images:", len(all_paths))
    print("device       :", device)
    print("dtype        :", dtype)
    print(
        "padding RGB  :",
        IMAGENET_MEAN_RGB,
    )

    kwargs = {}

    if args.cache_dir:
        kwargs[
            "cache_dir"
        ] = args.cache_dir

    if args.local_files_only:
        kwargs[
            "local_files_only"
        ] = True

    print("\n[LOAD MODEL]")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        **kwargs,
    )

    model = model.to(
        device=device,
        dtype=dtype,
    ).eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    hidden = int(
        model.config.hidden_size
    )

    num_register_tokens = int(
        getattr(
            model.config,
            "num_register_tokens",
            0,
        )
    )

    patch_size = int(
        getattr(
            model.config,
            "patch_size",
            14,
        )
    )

    num_prefix_tokens = (
        1
        + num_register_tokens
    )

    print(
        "hidden dim   :",
        hidden,
    )

    print(
        "registers    :",
        num_register_tokens,
    )

    print(
        "patch size   :",
        patch_size,
    )

    print(
        "prefix tokens:",
        num_prefix_tokens,
    )

    if hidden != 1536:
        raise RuntimeError(
            f"expected 1536-d giant, "
            f"got {hidden}"
        )

    if patch_size != 14:
        raise RuntimeError(
            f"expected patch size 14, "
            f"got {patch_size}"
        )

    # Verify HF sequence layout count.
    with torch.inference_mode():

        dummy = torch.zeros(
            1,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=device,
            dtype=dtype,
        )

        sequence_length = int(
            model(
                pixel_values=dummy
            )
            .last_hidden_state
            .shape[1]
        )

    expected_patches = (
        IMAGE_SIZE
        // patch_size
    ) ** 2

    actual_prefix_tokens = (
        sequence_length
        - expected_patches
    )

    if (
        actual_prefix_tokens
        != num_prefix_tokens
    ):
        raise RuntimeError(
            "token layout mismatch: "
            f"seq={sequence_length}, "
            f"patch={expected_patches}, "
            f"actual_prefix="
            f"{actual_prefix_tokens}, "
            f"config_prefix="
            f"{num_prefix_tokens}"
        )

    print(
        "token layout  : verified"
    )

    preprocess = Pad224()

    results = {}

    modes = (
        "cls",
        "signed_p2",
        "signed_p3",
    )

    for mode in modes:

        print(
            f"\n[RUN] "
            f"Pad224 + {mode}"
        )

        vectors = embed_paths(
            model,
            all_paths,
            preprocess,
            mode=mode,
            num_prefix_tokens=
                num_prefix_tokens,
            device=device,
            dtype=dtype,
            batch_size=
                args.batch_size,
        )

        if vectors.shape[1] != 1536:
            raise RuntimeError(
                f"{mode}: expected 1536D, "
                f"got {vectors.shape}"
            )

        if not np.isfinite(
            vectors
        ).all():
            raise RuntimeError(
                f"{mode}: NaN/Inf found"
            )

        norms = np.linalg.norm(
            vectors,
            axis=1,
        )

        print(
            "  norm mean :",
            float(
                norms.mean()
            ),
        )

        vector_map = {
            str(path): vector
            for path, vector
            in zip(
                all_paths,
                vectors,
            )
        }

        per_group = evaluate(
            groups,
            vector_map,
        )

        agg = aggregate(
            per_group
        )

        results[mode] = {
            "aggregate": agg,
            "per_group": per_group,
        }

        print(
            f"  R@1="
            f"{agg['R@1']:.4f} "
            f"R@5="
            f"{agg['R@5']:.4f} "
            f"MRR="
            f"{agg['MRR']:.4f}"
        )

        print(
            f"  gap="
            f"{agg['separation_gap_mean']:.6f} "
            f"margin="
            f"{agg['strict_margin_mean']:.6f}"
        )

    ordered = sorted(
        modes,
        key=lambda mode:
            score_tuple(
                results[
                    mode
                ]["aggregate"]
            ),
        reverse=True,
    )

    winner = ordered[0]

    baseline = results[
        "cls"
    ]["aggregate"]

    deltas = {}

    for mode in (
        "signed_p2",
        "signed_p3",
    ):

        candidate = results[
            mode
        ]["aggregate"]

        deltas[
            f"{mode}_minus_cls"
        ] = {
            key: float(
                candidate[key]
                - baseline[key]
            )
            for key in (
                "R@1",
                "R@5",
                "MRR",
                "positive_mean",
                "negative_mean",
                "separation_gap_mean",
                "strict_margin_mean",
                "strict_margin_positive_rate",
            )
        }

    output_dir = (
        Path(
            args.output_dir
        )
        .expanduser()
        .resolve()
        / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model_id": MODEL_ID,
        "hidden_dim": hidden,
        "num_register_tokens":
            num_register_tokens,
        "patch_size":
            patch_size,
        "input": (
            "AR preserve resize -> "
            "center pad 224 -> "
            "ImageNet normalize"
        ),
        "padding_rgb":
            IMAGENET_MEAN_RGB,
        "manifest":
            str(manifest),
        "groups":
            len(groups),
        "unique_images":
            len(all_paths),
        "results":
            results,
        "deltas_vs_cls":
            deltas,
        "ranking":
            ordered,
        "winner":
            winner,
    }

    (
        output_dir
        / "results.json"
    ).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        (
            "DINOv2 Giant+Registers "
            "AR-Pad224 "
            "CLS vs SignedPower"
        ),
        "=" * 78,
    ]

    for rank, mode in enumerate(
        ordered,
        start=1,
    ):

        agg = results[
            mode
        ]["aggregate"]

        lines.append(
            f"#{rank} {mode:10s} | "
            f"R@1={agg['R@1']:.4f} "
            f"R@5={agg['R@5']:.4f} "
            f"MRR={agg['MRR']:.4f} "
            f"gap="
            f"{agg['separation_gap_mean']:.6f} "
            f"margin="
            f"{agg['strict_margin_mean']:.6f}"
        )

    lines += [
        "",
        f"WINNER: {winner}",
        "",
        (
            "판정 원칙: "
            "signed pooling이 CLS를 명확히 "
            "이기지 못하면 CLS를 운영값으로 유지."
        ),
    ]

    (
        output_dir
        / "summary.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 92
    )

    print("FINAL")

    print(
        "=" * 92
    )

    for rank, mode in enumerate(
        ordered,
        start=1,
    ):

        agg = results[
            mode
        ]["aggregate"]

        print(
            f"#{rank} "
            f"{mode:10s} "
            f"R@1/R@5/MRR="
            f"{agg['R@1']:.4f}/"
            f"{agg['R@5']:.4f}/"
            f"{agg['MRR']:.4f} "
            f"gap="
            f"{agg['separation_gap_mean']:.6f} "
            f"margin="
            f"{agg['strict_margin_mean']:.6f}"
        )

    print(
        "WINNER :",
        winner,
    )

    print(
        "output :",
        output_dir,
    )


if __name__ == "__main__":
    main()
