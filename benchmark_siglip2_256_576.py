import gc
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from embedders.siglip2_embedder import SigLIP2Embedder


# ============================================================
# 설정
# ============================================================

META = Path(r".\data\crops\filter_stats.json")

MODEL_ID = "google/siglip2-base-patch16-naflex"

MAX_CLASSES = 25
MAX_PER_CLASS = 20
MAX_TOTAL = 500

BATCH_SIZE = 16
SEED = 42


# ============================================================
# 데이터 로드
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

data = json.loads(META.read_text(encoding="utf-8"))
records = data["crops"]

valid = []

for r in records:
    p = Path(r["crop_path"])

    if not p.exists():
        continue

    label = str(r.get("class_name", "")).strip()

    if not label:
        continue

    valid.append({
        "path": str(p),
        "label": label,
    })

print("=" * 70)
print("SIGLIP2 256 vs 576 BENCHMARK")
print("=" * 70)
print("metadata records :", len(records))
print("valid crops      :", len(valid))


# ============================================================
# 클래스별 분류
# ============================================================

by_class = defaultdict(list)

for r in valid:
    by_class[r["label"]].append(r)

counts = Counter(r["label"] for r in valid)

# 너무 적은 클래스보다 샘플이 충분한 클래스 우선
selected_classes = [
    name
    for name, count in counts.most_common(MAX_CLASSES)
    if count >= 2
]

sampled = []

for label in selected_classes:
    items = by_class[label].copy()
    random.shuffle(items)

    sampled.extend(items[:MAX_PER_CLASS])

sampled = sampled[:MAX_TOTAL]

paths = [r["path"] for r in sampled]
labels = np.array([r["label"] for r in sampled])

# 실제 샘플에 남아 있는 클래스만 사용
query_classes = sorted(set(labels.tolist()))

print("selected classes :", len(query_classes))
print("sample crops     :", len(paths))
print()

print("=== CLASS DISTRIBUTION ===")
for c in query_classes:
    print(f"{c:20s} : {(labels == c).sum()}")


# ============================================================
# 평가
# ============================================================

def evaluate(max_num_patches):

    print()
    print("=" * 70)
    print(f"max_num_patches = {max_num_patches}")
    print("=" * 70)

    emb = SigLIP2Embedder(
        model_id=MODEL_ID,
        max_num_patches=max_num_patches,
        batch_size=BATCH_SIZE,
        l2_normalize=True,
        fp16=True,
    )

    print("device     :", emb.device)
    print("is_naflex :", emb.is_naflex)
    print("DIM        :", emb.DIM)

    # --------------------------------------------------------
    # GPU warmup
    # --------------------------------------------------------

    warmup_paths = paths[:min(8, len(paths))]

    _ = emb.embed_crops(
        warmup_paths,
        input_format="rgb",
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # 이미지 임베딩 시간 측정
    # --------------------------------------------------------

    start = time.perf_counter()

    image_vecs = emb.embed_crops(
        paths,
        input_format="rgb",
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    throughput = len(paths) / elapsed

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    queries = [
        f"a photo of a {c}"
        for c in query_classes
    ]

    text_vecs = emb.embed_text(queries)

    # L2 normalized 상태라 dot product = cosine
    scores = text_vecs @ image_vecs.T

    top1_hits = 0
    top5_hits = 0
    top10_hits = 0

    per_query = []

    for qi, cls in enumerate(query_classes):

        ranking = np.argsort(-scores[qi])

        top1 = ranking[:1]
        top5 = ranking[:5]
        top10 = ranking[:10]

        hit1 = bool(np.any(labels[top1] == cls))
        hit5 = bool(np.any(labels[top5] == cls))
        hit10 = bool(np.any(labels[top10] == cls))

        top1_hits += int(hit1)
        top5_hits += int(hit5)
        top10_hits += int(hit10)

        per_query.append({
            "class": cls,
            "top1": hit1,
            "top5": hit5,
            "top10": hit10,
            "best_score": float(scores[qi, ranking[0]]),
            "best_label": str(labels[ranking[0]]),
        })

    n_queries = len(query_classes)

    metrics = {
        "patches": max_num_patches,
        "n_images": len(paths),
        "n_queries": n_queries,
        "seconds": elapsed,
        "crop_per_sec": throughput,
        "top1": top1_hits / n_queries,
        "top5": top5_hits / n_queries,
        "top10": top10_hits / n_queries,
        "image_vecs": image_vecs,
        "per_query": per_query,
    }

    if torch.cuda.is_available():
        metrics["peak_vram_gb"] = (
            torch.cuda.max_memory_allocated() / 1024**3
        )
    else:
        metrics["peak_vram_gb"] = None

    print()
    print("=== RESULT ===")
    print(f"Images      : {len(paths)}")
    print(f"Queries     : {n_queries}")
    print(f"Time        : {elapsed:.2f} sec")
    print(f"Throughput  : {throughput:.2f} crop/sec")
    print(f"Top-1       : {metrics['top1']*100:.2f}%")
    print(f"Top-5       : {metrics['top5']*100:.2f}%")
    print(f"Top-10      : {metrics['top10']*100:.2f}%")

    if metrics["peak_vram_gb"] is not None:
        print(f"Peak VRAM   : {metrics['peak_vram_gb']:.2f} GB")

    del emb

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ============================================================
# 256
# ============================================================

r256 = evaluate(256)


# ============================================================
# 576
# ============================================================

r576 = evaluate(576)


# ============================================================
# 같은 이미지의 벡터 변화량
# ============================================================

same_image_cos = np.sum(
    r256["image_vecs"] * r576["image_vecs"],
    axis=1,
)

mean_same_cos = float(np.mean(same_image_cos))
median_same_cos = float(np.median(same_image_cos))


# ============================================================
# 최종 비교
# ============================================================

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"{'Metric':<20}"
    f"{'256':>15}"
    f"{'576':>15}"
)

print("-" * 50)

print(
    f"{'Time(sec)':<20}"
    f"{r256['seconds']:>15.2f}"
    f"{r576['seconds']:>15.2f}"
)

print(
    f"{'Crop/sec':<20}"
    f"{r256['crop_per_sec']:>15.2f}"
    f"{r576['crop_per_sec']:>15.2f}"
)

print(
    f"{'Top-1':<20}"
    f"{r256['top1']*100:>14.2f}%"
    f"{r576['top1']*100:>14.2f}%"
)

print(
    f"{'Top-5':<20}"
    f"{r256['top5']*100:>14.2f}%"
    f"{r576['top5']*100:>14.2f}%"
)

print(
    f"{'Top-10':<20}"
    f"{r256['top10']*100:>14.2f}%"
    f"{r576['top10']*100:>14.2f}%"
)

if (
    r256["peak_vram_gb"] is not None
    and r576["peak_vram_gb"] is not None
):
    print(
        f"{'Peak VRAM(GB)':<20}"
        f"{r256['peak_vram_gb']:>15.2f}"
        f"{r576['peak_vram_gb']:>15.2f}"
    )

print()
print("Same-image embedding cosine")
print(f"mean   : {mean_same_cos:.6f}")
print(f"median : {median_same_cos:.6f}")


# ============================================================
# Query별 차이 출력
# ============================================================

print()
print("=== QUERY DIFFERENCES ===")

d256 = {x["class"]: x for x in r256["per_query"]}
d576 = {x["class"]: x for x in r576["per_query"]}

for cls in query_classes:

    a = d256[cls]
    b = d576[cls]

    if (
        a["top1"] != b["top1"]
        or a["top5"] != b["top5"]
        or a["top10"] != b["top10"]
    ):
        print(
            f"{cls:20s} | "
            f"256={int(a['top1'])}/{int(a['top5'])}/{int(a['top10'])} "
            f"576={int(b['top1'])}/{int(b['top5'])}/{int(b['top10'])}"
        )


# ============================================================
# JSON 저장
# ============================================================

result = {
    "sample_images": len(paths),
    "classes": query_classes,
    "256": {
        k: v for k, v in r256.items()
        if k not in ("image_vecs",)
    },
    "576": {
        k: v for k, v in r576.items()
        if k not in ("image_vecs",)
    },
    "same_image_cosine_mean": mean_same_cos,
    "same_image_cosine_median": median_same_cos,
}

Path("siglip2_256_vs_576_result.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("saved: siglip2_256_vs_576_result.json")
