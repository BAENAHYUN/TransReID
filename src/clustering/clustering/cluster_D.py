import time
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

from cluster_results import save_cluster_result


# =========================================================
# Configuration
# =========================================================

N_CLUSTERS = 1503

LINKAGE = "average"

METRIC = "cosine"


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
# Main
# =========================================================

def main():

    print("=" * 70)
    print(
        "CLUSTER D - AGGLOMERATIVE"
    )
    print("=" * 70)

    print()
    print(
        f"n_clusters : "
        f"{N_CLUSTERS}"
    )

    print(
        f"linkage    : "
        f"{LINKAGE}"
    )

    print(
        f"metric     : "
        f"{METRIC}"
    )

    # -----------------------------------------------------
    # Qdrant
    # -----------------------------------------------------

    print()
    print(
        "[1/4] Connecting Qdrant..."
    )

    client = get_client()

    # -----------------------------------------------------
    # Load vectors
    # -----------------------------------------------------

    print(
        "[2/4] Loading Market1501 Re-ID vectors..."
    )

    X, point_ids, payloads = (
        load_reid_vectors(
            client,
            source="market1501",
        )
    )

    print(
        f"Vector shape: {X.shape}"
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    print()
    print(
        "[3/4] Normalizing vectors..."
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
    # Agglomerative
    # -----------------------------------------------------

    print()
    print(
        "[4/4] Running Agglomerative..."
    )

    print(
        "This may take some time."
    )

    start = time.time()

    model = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric=METRIC,
        linkage=LINKAGE,
    )

    labels = model.fit_predict(
        X
    )

    elapsed = (
        time.time()
        - start
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    stats = calculate_statistics(
        labels,
        true_labels,
    )

    print()
    print(
        f"Elapsed: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("=" * 70)
    print(
        "AGGLOMERATIVE COMPLETE"
    )
    print("=" * 70)

    print(
        f"Samples       : "
        f"{len(labels)}"
    )

    print(
        f"Clusters      : "
        f"{stats['clusters']}"
    )

    print(
        f"Noise         : 0"
    )

    print()
    print(
        "Cluster size range:"
    )

    print(
        f"  Min : "
        f"{stats['min_cluster']}"
    )

    print(
        f"  Max : "
        f"{stats['max_cluster']}"
    )

    print(
        f"  Avg : "
        f"{stats['avg_cluster']:.2f}"
    )

    print()
    print(
        "Evaluation:"
    )

    print(
        f"  Purity : "
        f"{stats['purity']:.4f}"
    )

    print(
        f"  ARI    : "
        f"{stats['ARI']:.4f}"
    )

    print(
        f"  NMI    : "
        f"{stats['NMI']:.4f}"
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print()
    print(
        "Saving Agglomerative result..."
    )

    save_cluster_result(
        "cluster_D",
        labels,
        point_ids,
        payloads,
    )

    print()
    print("=" * 70)
    print(
        "CLUSTER D RESULT SAVED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()