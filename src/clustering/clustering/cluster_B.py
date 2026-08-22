import time
import numpy as np

from sklearn.cluster import DBSCAN

from qdrant_vectors import (
    get_client,
    load_reid_vectors,
    normalize_vectors,
)

from cluster_results import save_cluster_result


# =========================================================
# Configuration
# =========================================================

EPS = 0.08

MIN_SAMPLES = 5


# =========================================================
# DBSCAN
# =========================================================

def run_dbscan(
    X,
    eps=EPS,
    min_samples=MIN_SAMPLES,
):

    X = normalize_vectors(X)

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    labels = model.fit_predict(X)

    return {
        "algorithm": "DBSCAN",
        "labels": labels,
        "model": model,
    }


# =========================================================
# Statistics
# =========================================================

def print_statistics(labels):

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    cluster_labels = [
        label
        for label in unique
        if label >= 0
    ]

    noise_count = int(
        np.sum(labels == -1)
    )

    print()
    print("=" * 70)
    print("DBSCAN COMPLETE")
    print("=" * 70)

    print(
        f"Samples       : {len(labels)}"
    )

    print(
        f"Clusters      : {len(cluster_labels)}"
    )

    print(
        f"Noise (-1)    : {noise_count}"
    )

    if cluster_labels:

        cluster_counts = [
            count
            for label, count in zip(
                unique,
                counts,
            )
            if label >= 0
        ]

        print()
        print("Cluster size range:")

        print(
            f"  Min : {min(cluster_counts)}"
        )

        print(
            f"  Max : {max(cluster_counts)}"
        )

        print(
            f"  Avg : "
            f"{np.mean(cluster_counts):.2f}"
        )

        print()
        print("Largest 20 clusters:")

        pairs = sorted(
            zip(
                unique,
                counts,
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        shown = 0

        for label, count in pairs:

            if label < 0:
                continue

            print(
                f"  Cluster {label:5d} : "
                f"{count:5d}"
            )

            shown += 1

            if shown >= 20:
                break


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CLUSTER B - DBSCAN")
    print("=" * 70)

    print()
    print(
        f"eps         : {EPS}"
    )

    print(
        f"min_samples : {MIN_SAMPLES}"
    )

    # -----------------------------------------------------
    # Qdrant
    # -----------------------------------------------------

    print()
    print(
        "Connecting Qdrant..."
    )

    client = get_client()

    # -----------------------------------------------------
    # Load vectors
    # -----------------------------------------------------

    print(
        "Loading Market1501 Re-ID vectors..."
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
    # DBSCAN
    # -----------------------------------------------------

    print()
    print(
        "Running DBSCAN..."
    )

    print(
        "This may take longer than K-Means."
    )

    start = time.time()

    result = run_dbscan(
        X,
        eps=EPS,
        min_samples=MIN_SAMPLES,
    )

    elapsed = (
        time.time() - start
    )

    labels = result["labels"]

    print()
    print(
        f"Elapsed: "
        f"{elapsed / 60:.2f} minutes"
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    print_statistics(
        labels
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print()
    print(
        "Saving DBSCAN result..."
    )

    save_cluster_result(
        "cluster_B",
        labels,
        point_ids,
        payloads,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "CLUSTER B RESULT SAVED"
    )

    print(
        "=" * 70
    )