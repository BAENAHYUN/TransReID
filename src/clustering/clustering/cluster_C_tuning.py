import json
import time
from pathlib import Path

import numpy as np

from sklearn.cluster import HDBSCAN
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

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "cluster_results"
    / "cluster_C"
)

SOURCE = "market1501"

MIN_CLUSTER_SIZE_VALUES = [
    5,
    10,
    20,
    30,
    50,
]

MIN_SAMPLES_VALUES = [
    5,
    10,
    20,
]

CLUSTER_SELECTION_METHOD = "eom"


# =========================================================
# Ground Truth
# =========================================================

def get_ground_truth(payloads):

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
# Weighted Purity
# =========================================================

def calculate_weighted_purity(
    true_labels,
    cluster_labels,
):

    mask = (
        cluster_labels != -1
    )

    if not np.any(mask):
        return 0.0

    y_true = true_labels[mask]
    y_pred = cluster_labels[mask]

    total_correct = 0

    for cluster_id in np.unique(
        y_pred
    ):

        cluster_truth = y_true[
            y_pred == cluster_id
        ]

        if len(cluster_truth) == 0:
            continue

        _, counts = np.unique(
            cluster_truth,
            return_counts=True,
        )

        total_correct += int(
            counts.max()
        )

    return (
        total_correct
        / len(y_true)
    )


# =========================================================
# Statistics
# =========================================================

def calculate_statistics(
    labels,
    true_labels,
):

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    cluster_sizes = [
        int(count)
        for label, count in zip(
            unique,
            counts,
        )
        if label >= 0
    ]

    cluster_count = len(
        cluster_sizes
    )

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

    if cluster_sizes:

        min_cluster = min(
            cluster_sizes
        )

        max_cluster = max(
            cluster_sizes
        )

        avg_cluster = float(
            np.mean(cluster_sizes)
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

    ari = adjusted_rand_score(
        true_labels,
        labels,
    )

    nmi = normalized_mutual_info_score(
        true_labels,
        labels,
    )

    return {
        "clusters": cluster_count,
        "noise": noise_count,
        "clustered": clustered_count,
        "noise_ratio": (
            noise_count
            / len(labels)
        ),
        "min_cluster": min_cluster,
        "max_cluster": max_cluster,
        "avg_cluster": avg_cluster,
        "purity": purity,
        "ARI": ari,
        "NMI": nmi,
    }


# =========================================================
# Run one experiment
# =========================================================

def run_one(
    X,
    true_labels,
    min_cluster_size,
    min_samples,
):

    print()
    print("-" * 70)

    print(
        "HDBSCAN"
        f" min_cluster_size={min_cluster_size}"
        f" min_samples={min_samples}"
    )

    start = time.time()

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
        cluster_selection_method=(
            CLUSTER_SELECTION_METHOD
        ),
        n_jobs=-1,
    )

    labels = model.fit_predict(
        X
    )

    elapsed = (
        time.time()
        - start
    )

    stats = calculate_statistics(
        labels,
        true_labels,
    )

    result = {
        "min_cluster_size": (
            min_cluster_size
        ),
        "min_samples": (
            min_samples
        ),
        "clusters": stats["clusters"],
        "noise": stats["noise"],
        "clustered": stats["clustered"],
        "noise_ratio": stats[
            "noise_ratio"
        ],
        "min_cluster": stats[
            "min_cluster"
        ],
        "max_cluster": stats[
            "max_cluster"
        ],
        "avg_cluster": stats[
            "avg_cluster"
        ],
        "purity": stats[
            "purity"
        ],
        "ARI": stats[
            "ARI"
        ],
        "NMI": stats[
            "NMI"
        ],
        "time_seconds": elapsed,
    }

    print(
        f"Clusters : "
        f"{result['clusters']}"
    )

    print(
        f"Noise    : "
        f"{result['noise']}"
    )

    print(
        f"Purity   : "
        f"{result['purity']:.4f}"
    )

    print(
        f"ARI      : "
        f"{result['ARI']:.4f}"
    )

    print(
        f"NMI      : "
        f"{result['NMI']:.4f}"
    )

    print(
        f"Time     : "
        f"{elapsed / 60:.2f} min"
    )

    return result


# =========================================================
# Save tuning results
# =========================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Best by ARI
    # -----------------------------------------------------

    best_ari = max(
        results,
        key=lambda x: x["ARI"],
    )

    # -----------------------------------------------------
    # Best by NMI
    # -----------------------------------------------------

    best_nmi = max(
        results,
        key=lambda x: x["NMI"],
    )

    # -----------------------------------------------------
    # Best by Purity
    # -----------------------------------------------------

    best_purity = max(
        results,
        key=lambda x: x["purity"],
    )

    output = {

        "dataset": "Market-1501",

        "source": SOURCE,

        "samples": len(results),

        "cluster_selection_method": (
            CLUSTER_SELECTION_METHOD
        ),

        "min_cluster_size_values": (
            MIN_CLUSTER_SIZE_VALUES
        ),

        "min_samples_values": (
            MIN_SAMPLES_VALUES
        ),

        "results": results,

        "best_ARI": best_ari,

        "best_NMI": best_nmi,

        "best_purity": best_purity,
    }

    output_path = (
        OUTPUT_DIR
        / "hdbscan_tuning.json"
    )

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
        "Tuning result saved:"
    )

    print(
        output_path
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print(
        "HDBSCAN PARAMETER TUNING"
    )
    print("=" * 70)

    combinations = [
        (
            min_cluster_size,
            min_samples,
        )
        for min_cluster_size
        in MIN_CLUSTER_SIZE_VALUES
        for min_samples
        in MIN_SAMPLES_VALUES
    ]

    print()
    print(
        f"Total experiments: "
        f"{len(combinations)}"
    )

    print(
        f"min_cluster_size: "
        f"{MIN_CLUSTER_SIZE_VALUES}"
    )

    print(
        f"min_samples: "
        f"{MIN_SAMPLES_VALUES}"
    )

    # -----------------------------------------------------
    # Load Qdrant
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
    # Experiments
    # -----------------------------------------------------

    print()
    print(
        "[3/3] Running HDBSCAN experiments..."
    )

    results = []

    total = len(
        combinations
    )

    for index, (
        min_cluster_size,
        min_samples,
    ) in enumerate(
        combinations,
        start=1,
    ):

        print()
        print(
            f"[{index}/{total}]"
        )

        result = run_one(
            X,
            true_labels,
            min_cluster_size,
            min_samples,
        )

        results.append(
            result
        )

        # Save after EVERY experiment.
        save_results(
            results
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print(
        "HDBSCAN TUNING SUMMARY"
    )
    print("=" * 70)

    print()

    print(
        f"{'MCS':>5} "
        f"{'MS':>5} "
        f"{'Clusters':>10} "
        f"{'Noise':>8} "
        f"{'Purity':>9} "
        f"{'ARI':>9} "
        f"{'NMI':>9}"
    )

    print(
        "-" * 70
    )

    for result in results:

        print(
            f"{result['min_cluster_size']:>5} "
            f"{result['min_samples']:>5} "
            f"{result['clusters']:>10} "
            f"{result['noise']:>8} "
            f"{result['purity']:>9.4f} "
            f"{result['ARI']:>9.4f} "
            f"{result['NMI']:>9.4f}"
        )

    # -----------------------------------------------------
    # Best
    # -----------------------------------------------------

    best_ari = max(
        results,
        key=lambda x: x["ARI"],
    )

    best_nmi = max(
        results,
        key=lambda x: x["NMI"],
    )

    best_purity = max(
        results,
        key=lambda x: x["purity"],
    )

    print()
    print("=" * 70)
    print(
        "BEST PARAMETERS"
    )
    print("=" * 70)

    print()
    print("Best ARI:")

    print(
        f"  min_cluster_size = "
        f"{best_ari['min_cluster_size']}"
    )

    print(
        f"  min_samples      = "
        f"{best_ari['min_samples']}"
    )

    print(
        f"  ARI              = "
        f"{best_ari['ARI']:.4f}"
    )

    print(
        f"  Clusters         = "
        f"{best_ari['clusters']}"
    )

    print(
        f"  Noise            = "
        f"{best_ari['noise']}"
    )

    print()
    print("Best NMI:")

    print(
        f"  min_cluster_size = "
        f"{best_nmi['min_cluster_size']}"
    )

    print(
        f"  min_samples      = "
        f"{best_nmi['min_samples']}"
    )

    print(
        f"  NMI              = "
        f"{best_nmi['NMI']:.4f}"
    )

    print(
        f"  Clusters         = "
        f"{best_nmi['clusters']}"
    )

    print(
        f"  Noise            = "
        f"{best_nmi['noise']}"
    )

    print()
    print("Best Purity:")

    print(
        f"  min_cluster_size = "
        f"{best_purity['min_cluster_size']}"
    )

    print(
        f"  min_samples      = "
        f"{best_purity['min_samples']}"
    )

    print(
        f"  Purity           = "
        f"{best_purity['purity']:.4f}"
    )

    print(
        f"  Clusters         = "
        f"{best_purity['clusters']}"
    )

    print(
        f"  Noise            = "
        f"{best_purity['noise']}"
    )

    print()
    print("=" * 70)
    print(
        "HDBSCAN TUNING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()