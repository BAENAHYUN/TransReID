import time
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

from cluster_results import save_cluster_result


# =========================================================
# Configuration
# =========================================================

MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 5

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

    clustered_mask = (
        cluster_labels != -1
    )

    if not np.any(
        clustered_mask
    ):
        return 0.0

    clustered_true = (
        true_labels[
            clustered_mask
        ]
    )

    clustered_pred = (
        cluster_labels[
            clustered_mask
        ]
    )

    total_correct = 0

    for cluster_id in np.unique(
        clustered_pred
    ):

        mask = (
            clustered_pred
            == cluster_id
        )

        cluster_truth = (
            clustered_true[mask]
        )

        if len(cluster_truth) == 0:
            continue

        _, counts = np.unique(
            cluster_truth,
            return_counts=True,
        )

        total_correct += (
            counts.max()
        )

    return (
        total_correct
        / len(clustered_true)
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

    cluster_ids = [
        int(label)
        for label in unique
        if label >= 0
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
        "purity": purity,
        "ARI": ari,
        "NMI": nmi,
    }


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("CLUSTER C - HDBSCAN")
    print("=" * 70)

    print()
    print(
        f"min_cluster_size : "
        f"{MIN_CLUSTER_SIZE}"
    )

    print(
        f"min_samples      : "
        f"{MIN_SAMPLES}"
    )

    print(
        f"selection_method  : "
        f"{CLUSTER_SELECTION_METHOD}"
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
    # HDBSCAN
    # -----------------------------------------------------

    print()
    print(
        "[4/4] Running HDBSCAN..."
    )

    print(
        "This may take some time."
    )

    start = time.time()

    model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
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

    print()
    print(
        f"Elapsed: "
        f"{elapsed / 60:.2f} minutes"
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    stats = calculate_statistics(
        labels,
        true_labels,
    )

    print()
    print("=" * 70)
    print("HDBSCAN COMPLETE")
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
        f"Noise (-1)    : "
        f"{stats['noise']}"
    )

    print(
        f"Noise ratio   : "
        f"{stats['noise_ratio']:.4f}"
    )

    print()
    print("Cluster size range:")

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
    print("Evaluation:")

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
    # Largest clusters
    # -----------------------------------------------------

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    pairs = sorted(
        [
            (int(label), int(count))
            for label, count in zip(
                unique,
                counts,
            )
            if label >= 0
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    print()
    print("Largest 20 clusters:")

    for label, count in pairs[:20]:

        print(
            f"  Cluster "
            f"{label:5d} : "
            f"{count:5d}"
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print()
    print(
        "Saving HDBSCAN result..."
    )

    save_cluster_result(
        "cluster_C",
        labels,
        point_ids,
        payloads,
    )

    print()
    print("=" * 70)
    print(
        "CLUSTER C RESULT SAVED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()