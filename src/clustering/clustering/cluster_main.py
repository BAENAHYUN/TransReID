import sys
from pathlib import Path
from cluster_results import save_cluster_result
import numpy as np


ROOT_DIR = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)


from qdrant_vectors import (
    get_client,
    load_reid_vectors,
)

from cluster_A import run_kmeans
from cluster_B import run_dbscan
from cluster_C import run_hdbscan
from cluster_D import run_agglomerative


# =========================================================
# Configuration
# =========================================================

SOURCE = "market1501"

KMEANS_CLUSTERS = 100

AGGLOMERATIVE_CLUSTERS = 100


# =========================================================
# Statistics
# =========================================================

def print_statistics(
    name,
    labels,
):

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Samples: {len(labels)}"
    )

    print(
        f"Clusters: "
        f"{len(unique[unique >= 0])}"
    )

    noise = np.sum(
        labels == -1
    )

    print(
        f"Noise: {noise}"
    )

    print()
    print(
        "Cluster sizes:"
    )

    for label, count in zip(
        unique,
        counts,
    ):

        print(
            f"  {label:>5} : {count}"
        )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print(
        "FORENSIC CLUSTERING PIPELINE"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Qdrant
    # -----------------------------------------------------

    client = get_client()

    print()
    print(
        "Loading Re-ID vectors from Qdrant..."
    )

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
    # A
    # -----------------------------------------------------

    result_A = run_kmeans(
        X,
        n_clusters=KMEANS_CLUSTERS,
    )

    print_statistics(
        "CLUSTER A - KMEANS",
        result_A["labels"],
    )

    # -----------------------------------------------------
    # B
    # -----------------------------------------------------

    result_B = run_dbscan(
        X,
        eps=0.35,
        min_samples=5,
    )

    print_statistics(
        "CLUSTER B - DBSCAN",
        result_B["labels"],
    )

    # -----------------------------------------------------
    # C
    # -----------------------------------------------------

    result_C = run_hdbscan(
        X,
        min_cluster_size=5,
        min_samples=5,
    )

    print_statistics(
        "CLUSTER C - HDBSCAN",
        result_C["labels"],
    )

    # -----------------------------------------------------
    # D
    # -----------------------------------------------------

    result_D = run_agglomerative(
        X,
        n_clusters=AGGLOMERATIVE_CLUSTERS,
    )

    print_statistics(
        "CLUSTER D - AGGLOMERATIVE",
        result_D["labels"],
    )

    print()
    print("=" * 70)
    print(
        "CLUSTERING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()