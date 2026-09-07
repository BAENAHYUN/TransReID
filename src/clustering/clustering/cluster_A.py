import numpy as np

from sklearn.cluster import KMeans

from qdrant_vectors import (
    get_client,
    load_reid_vectors,
    normalize_vectors,
)
from cluster_results import save_cluster_result

def run_kmeans(
    X,
    n_clusters=100,
    random_state=42,
):

    X = normalize_vectors(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )

    labels = model.fit_predict(X)

    return {
        "algorithm": "KMeans",
        "labels": labels,
        "model": model,
    }


if __name__ == "__main__":

    print("=" * 70)
    print("CLUSTER A - K-MEANS")
    print("=" * 70)

    print()
    print("Connecting Qdrant...")

    client = get_client()

    print("Loading Market1501 Re-ID vectors...")

    X, point_ids, payloads = load_reid_vectors(
        client,
        source="market1501",
    )

    print(
        f"Vector shape: {X.shape}"
    )

    print()
    print("Running K-Means...")
    print("n_clusters = 100")

    result = run_kmeans(
        X,
        n_clusters=100,
    )

    labels = result["labels"]

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    print()
    print("=" * 70)
    print("K-MEANS COMPLETE")
    print("=" * 70)

    print(
        f"Samples  : {len(labels)}"
    )

    print(
        f"Clusters : {len(unique)}"
    )

    print()
    print("Cluster size range:")
    print(
        f"  Min : {counts.min()}"
    )
    print(
        f"  Max : {counts.max()}"
    )
    print(
        f"  Avg : {counts.mean():.2f}"
    )

    print()
    print("First 20 clusters:")

    for label, count in zip(
        unique[:20],
        counts[:20],
    ):

        print(
            f"  Cluster {label:3d} : "
            f"{count:5d}"
        )
    save_cluster_result(
        "cluster_A",
        labels,
        point_ids,
        payloads,
    )

    print()
    print("Cluster A result saved.")
