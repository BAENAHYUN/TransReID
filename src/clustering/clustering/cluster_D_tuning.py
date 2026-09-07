import json
import time
from pathlib import Path

import numpy as np

from sklearn.cluster import AgglomerativeClustering
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
    / "cluster_D"
)

SOURCE = "market1501"

# 실제 identity 수가 1503이므로
# 그 주변을 중심으로 비교
N_CLUSTERS_VALUES = [
    500,
    750,
    1000,
    1250,
    1503,
    1750,
    2000,
]

LINKAGE_VALUES = [
    "average",
    "complete",
]


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
# Purity
# =========================================================

def calculate_purity(
    true_labels,
    cluster_labels,
):

    total_correct = 0

    for cluster_id in np.unique(
        cluster_labels
    ):

        cluster_truth = (
            true_labels[
                cluster_labels
                == cluster_id
            ]
        )

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
        / len(true_labels)
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
    ]

    purity = calculate_purity(
        true_labels,
        labels,
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
        "clusters": len(
            cluster_sizes
        ),
        "noise": 0,
        "min_cluster": min(
            cluster_sizes
        ),
        "max_cluster": max(
            cluster_sizes
        ),
        "avg_cluster": float(
            np.mean(cluster_sizes)
        ),
        "purity": purity,
        "ARI": ari,
        "NMI": nmi,
    }


# =========================================================
# One experiment
# =========================================================

def run_one(
    X,
    true_labels,
    n_clusters,
    linkage,
):

    print()
    print("-" * 70)

    print(
        "Agglomerative"
        f" n_clusters={n_clusters}"
        f" linkage={linkage}"
    )

    start = time.time()

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage=linkage,
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
        "n_clusters": n_clusters,
        "linkage": linkage,
        "clusters": stats[
            "clusters"
        ],
        "noise": stats[
            "noise"
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
# Save results
# =========================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    output = {
        "dataset": "Market-1501",

        "source": SOURCE,

        "n_clusters_values": (
            N_CLUSTERS_VALUES
        ),

        "linkage_values": (
            LINKAGE_VALUES
        ),

        "results": results,

        "best_ARI": best_ari,

        "best_NMI": best_nmi,

        "best_purity": best_purity,
    }

    output_path = (
        OUTPUT_DIR
        / "agglomerative_tuning.json"
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
        "AGGLOMERATIVE PARAMETER TUNING"
    )
    print("=" * 70)

    combinations = [
        (
            n_clusters,
            linkage,
        )
        for linkage
        in LINKAGE_VALUES
        for n_clusters
        in N_CLUSTERS_VALUES
    ]

    print()
    print(
        f"Total experiments: "
        f"{len(combinations)}"
    )

    print(
        f"n_clusters: "
        f"{N_CLUSTERS_VALUES}"
    )

    print(
        f"linkage: "
        f"{LINKAGE_VALUES}"
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
        "[3/3] Running experiments..."
    )

    results = []

    total = len(
        combinations
    )

    for index, (
        n_clusters,
        linkage,
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
            n_clusters,
            linkage,
        )

        results.append(
            result
        )

        # 매 실험 종료마다 저장
        save_results(
            results
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print(
        "AGGLOMERATIVE TUNING SUMMARY"
    )
    print("=" * 70)

    print()

    print(
        f"{'N':>6} "
        f"{'Linkage':>10} "
        f"{'Clusters':>10} "
        f"{'Noise':>7} "
        f"{'Purity':>9} "
        f"{'ARI':>9} "
        f"{'NMI':>9}"
    )

    print(
        "-" * 70
    )

    for result in results:

        print(
            f"{result['n_clusters']:>6} "
            f"{result['linkage']:>10} "
            f"{result['clusters']:>10} "
            f"{result['noise']:>7} "
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
        f"  n_clusters = "
        f"{best_ari['n_clusters']}"
    )

    print(
        f"  linkage    = "
        f"{best_ari['linkage']}"
    )

    print(
        f"  ARI        = "
        f"{best_ari['ARI']:.4f}"
    )

    print(
        f"  Purity     = "
        f"{best_ari['purity']:.4f}"
    )

    print(
        f"  NMI        = "
        f"{best_ari['NMI']:.4f}"
    )

    print()
    print("Best NMI:")

    print(
        f"  n_clusters = "
        f"{best_nmi['n_clusters']}"
    )

    print(
        f"  linkage    = "
        f"{best_nmi['linkage']}"
    )

    print(
        f"  NMI        = "
        f"{best_nmi['NMI']:.4f}"
    )

    print(
        f"  Purity     = "
        f"{best_nmi['purity']:.4f}"
    )

    print(
        f"  ARI        = "
        f"{best_nmi['ARI']:.4f}"
    )

    print()
    print("Best Purity:")

    print(
        f"  n_clusters = "
        f"{best_purity['n_clusters']}"
    )

    print(
        f"  linkage    = "
        f"{best_purity['linkage']}"
    )

    print(
        f"  Purity     = "
        f"{best_purity['purity']:.4f}"
    )

    print(
        f"  ARI        = "
        f"{best_purity['ARI']:.4f}"
    )

    print(
        f"  NMI        = "
        f"{best_purity['NMI']:.4f}"
    )

    print()
    print("=" * 70)
    print(
        "AGGLOMERATIVE TUNING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()