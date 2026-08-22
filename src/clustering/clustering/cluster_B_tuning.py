import json
import time
from pathlib import Path

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from qdrant_vectors import (
    get_client,
    load_reid_vectors,
    normalize_vectors,
)


# =========================================================
# Configuration
# =========================================================

ROOT_DIR = Path(
    __file__
).resolve().parents[2]

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "cluster_results"
    / "cluster_B"
)

SOURCE = "market1501"

MIN_SAMPLES = 5

# 너무 넓게 잡지 않고 단계적으로 확인
EPS_VALUES = [
    0.055,
    0.060,
    0.065,
    0.070,
    0.075,
    0.080,
    0.085,
    0.090,
    0.095,
]


# =========================================================
# Ground truth
# =========================================================

def get_ground_truth(
    payloads,
):

    labels = []

    for payload in payloads:

        market_id = payload.get(
            "market_id"
        )

        if market_id is None:

            raise ValueError(
                "Payload does not contain "
                "'market_id'."
            )

        labels.append(
            int(market_id)
        )

    return np.asarray(
        labels,
        dtype=np.int32,
    )


# =========================================================
# Weighted purity
# =========================================================

def calculate_weighted_purity(
    true_labels,
    cluster_labels,
):

    total = len(
        true_labels
    )

    if total == 0:
        return 0.0

    purity_sum = 0

    unique_clusters = np.unique(
        cluster_labels
    )

    for cluster_id in unique_clusters:

        # DBSCAN noise
        if cluster_id == -1:
            continue

        mask = (
            cluster_labels
            == cluster_id
        )

        cluster_truth = (
            true_labels[mask]
        )

        if len(cluster_truth) == 0:
            continue

        _, counts = np.unique(
            cluster_truth,
            return_counts=True,
        )

        purity_sum += counts.max()

    # Noise is not considered a correctly grouped cluster.
    clustered_count = np.sum(
        cluster_labels != -1
    )

    if clustered_count == 0:
        return 0.0

    return (
        purity_sum
        / clustered_count
    )


# =========================================================
# Cluster statistics
# =========================================================

def get_statistics(
    labels,
    true_labels,
):

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    cluster_ids = [
        int(x)
        for x in unique
        if x >= 0
    ]

    noise_count = int(
        np.sum(
            labels == -1
        )
    )

    clustered_count = int(
        np.sum(
            labels != -1
        )
    )

    if cluster_ids:

        cluster_sizes = [
            int(count)
            for label, count in zip(
                unique,
                counts,
            )
            if label >= 0
        ]

        min_cluster = min(
            cluster_sizes
        )

        max_cluster = max(
            cluster_sizes
        )

        avg_cluster = float(
            np.mean(
                cluster_sizes
            )
        )

    else:

        min_cluster = 0
        max_cluster = 0
        avg_cluster = 0.0

    purity = (
        calculate_weighted_purity(
            true_labels,
            labels,
        )
    )

    # ARI / NMI 전체 데이터 기준
    ari = adjusted_rand_score(
        true_labels,
        labels,
    )

    nmi = normalized_mutual_info_score(
        true_labels,
        labels,
    )

    return {
        "clusters": len(
            cluster_ids
        ),

        "noise": noise_count,

        "clustered": clustered_count,

        "noise_ratio": (
            noise_count
            / len(labels)
        ),

        "min_cluster": min_cluster,

        "max_cluster": max_cluster,

        "avg_cluster": avg_cluster,

        "weighted_purity": purity,

        "ARI": ari,

        "NMI": nmi,
    }


# =========================================================
# Run one DBSCAN
# =========================================================

def run_one(
    X,
    true_labels,
    eps,
):

    print()
    print(
        "-" * 70
    )

    print(
        f"DBSCAN eps={eps}"
    )

    start = time.time()

    model = DBSCAN(
        eps=eps,
        min_samples=MIN_SAMPLES,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    labels = model.fit_predict(
        X
    )

    elapsed = (
        time.time()
        - start
    )

    stats = get_statistics(
        labels,
        true_labels,
    )

    stats["eps"] = eps

    stats["min_samples"] = (
        MIN_SAMPLES
    )

    stats["time_seconds"] = (
        elapsed
    )

    print(
        f"Clusters       : "
        f"{stats['clusters']}"
    )

    print(
        f"Noise          : "
        f"{stats['noise']}"
    )

    print(
        f"Noise ratio    : "
        f"{stats['noise_ratio']:.4f}"
    )

    print(
        f"Min cluster    : "
        f"{stats['min_cluster']}"
    )

    print(
        f"Max cluster    : "
        f"{stats['max_cluster']}"
    )

    print(
        f"Avg cluster    : "
        f"{stats['avg_cluster']:.2f}"
    )

    print(
        f"Purity         : "
        f"{stats['weighted_purity']:.4f}"
    )

    print(
        f"ARI            : "
        f"{stats['ARI']:.4f}"
    )

    print(
        f"NMI            : "
        f"{stats['NMI']:.4f}"
    )

    print(
        f"Time           : "
        f"{elapsed / 60:.2f} min"
    )

    return stats


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 70)
    print(
        "DBSCAN PARAMETER TUNING"
    )
    print("=" * 70)

    print()
    print(
        f"eps values  : "
        f"{EPS_VALUES}"
    )

    print(
        f"min_samples : "
        f"{MIN_SAMPLES}"
    )

    # -----------------------------------------------------
    # Qdrant
    # -----------------------------------------------------

    print()
    print(
        "[1/3] Loading vectors from Qdrant..."
    )

    client = get_client()

    X, point_ids, payloads = (
        load_reid_vectors(
            client,
            source=SOURCE,
        )
    )

    print(
        f"Vectors: {X.shape}"
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    print()
    print(
        "[2/3] Normalizing vectors..."
    )

    X = normalize_vectors(
        X
    )

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    print(
        "Loading Market1501 ground truth..."
    )

    true_labels = (
        get_ground_truth(
            payloads
        )
    )

    print(
        f"Ground truth labels: "
        f"{len(true_labels)}"
    )

    print(
        f"Unique identities: "
        f"{len(np.unique(true_labels))}"
    )

    # -----------------------------------------------------
    # Tuning
    # -----------------------------------------------------

    print()
    print(
        "[3/3] Running DBSCAN experiments..."
    )

    results = []

    for eps in EPS_VALUES:

        stats = run_one(
            X,
            true_labels,
            eps,
        )

        results.append(
            stats
        )

    # -----------------------------------------------------
    # Sort by ARI
    # -----------------------------------------------------

    results_by_ari = sorted(
        results,
        key=lambda x: x["ARI"],
        reverse=True,
    )

    results_by_nmi = sorted(
        results,
        key=lambda x: x["NMI"],
        reverse=True,
    )

    results_by_purity = sorted(
        results,
        key=lambda x: x[
            "weighted_purity"
        ],
        reverse=True,
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print(
        "DBSCAN TUNING SUMMARY"
    )
    print("=" * 70)

    print()

    print(
        f"{'eps':>6} "
        f"{'clusters':>10} "
        f"{'noise':>8} "
        f"{'purity':>10} "
        f"{'ARI':>10} "
        f"{'NMI':>10}"
    )

    print(
        "-" * 70
    )

    for result in results:

        print(
            f"{result['eps']:>6.2f} "
            f"{result['clusters']:>10} "
            f"{result['noise']:>8} "
            f"{result['weighted_purity']:>10.4f} "
            f"{result['ARI']:>10.4f} "
            f"{result['NMI']:>10.4f}"
        )

    # -----------------------------------------------------
    # Best results
    # -----------------------------------------------------

    best_ari = results_by_ari[0]

    best_nmi = results_by_nmi[0]

    best_purity = (
        results_by_purity[0]
    )

    print()
    print("=" * 70)
    print(
        "BEST PARAMETERS"
    )
    print("=" * 70)

    print()
    print(
        "Best ARI:"
    )

    print(
        f"  eps       = "
        f"{best_ari['eps']}"
    )

    print(
        f"  ARI       = "
        f"{best_ari['ARI']:.4f}"
    )

    print(
        f"  clusters  = "
        f"{best_ari['clusters']}"
    )

    print(
        f"  noise     = "
        f"{best_ari['noise']}"
    )

    print()
    print(
        "Best NMI:"
    )

    print(
        f"  eps       = "
        f"{best_nmi['eps']}"
    )

    print(
        f"  NMI       = "
        f"{best_nmi['NMI']:.4f}"
    )

    print(
        f"  clusters  = "
        f"{best_nmi['clusters']}"
    )

    print(
        f"  noise     = "
        f"{best_nmi['noise']}"
    )

    print()
    print(
        "Best Purity:"
    )

    print(
        f"  eps       = "
        f"{best_purity['eps']}"
    )

    print(
        f"  Purity    = "
        f"{best_purity['weighted_purity']:.4f}"
    )

    print(
        f"  clusters  = "
        f"{best_purity['clusters']}"
    )

    print(
        f"  noise     = "
        f"{best_purity['noise']}"
    )

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "dbscan_tuning.json"
    )

    output = {

        "dataset": "Market-1501",

        "source": SOURCE,

        "samples": len(X),

        "min_samples": (
            MIN_SAMPLES
        ),

        "eps_values": EPS_VALUES,

        "results": results,

        "best_ARI": best_ari,

        "best_NMI": best_nmi,

        "best_purity": best_purity,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Tuning result saved:"
    )

    print(
        output_path
    )

    print()
    print("=" * 70)
    print(
        "DBSCAN TUNING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()