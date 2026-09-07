import json
from pathlib import Path

import numpy as np


ROOT_DIR = Path(
    __file__
).resolve().parents[2]

RESULTS_DIR = (
    ROOT_DIR
    / "data"
    / "cluster_results"
)


# =========================================================
# Directory
# =========================================================

def ensure_result_directory(
    algorithm
):

    output_dir = (
        RESULTS_DIR
        / algorithm
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# =========================================================
# Save labels
# =========================================================

def save_labels(
    algorithm,
    labels,
):

    output_dir = (
        ensure_result_directory(
            algorithm
        )
    )

    path = (
        output_dir
        / "labels.npy"
    )

    np.save(
        path,
        labels,
    )

    return path


# =========================================================
# Save cluster JSON
# =========================================================

def save_clusters_json(
    algorithm,
    labels,
    point_ids,
    payloads,
):

    output_dir = (
        ensure_result_directory(
            algorithm
        )
    )

    clusters = {}

    for label, point_id, payload in zip(
        labels,
        point_ids,
        payloads,
    ):

        label = int(label)

        key = str(label)

        if key not in clusters:

            clusters[key] = []

        clusters[key].append(
            {
                "point_id": str(
                    point_id
                ),

                "source": payload.get(
                    "source"
                ),

                "dataset": payload.get(
                    "dataset"
                ),

                "split": payload.get(
                    "split"
                ),

                "market_id": payload.get(
                    "market_id"
                ),

                "camera_id": payload.get(
                    "camera_id"
                ),

                "original_image": payload.get(
                    "original_image"
                ),

                "filename": payload.get(
                    "filename"
                ),
            }
        )

    path = (
        output_dir
        / "clusters.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            clusters,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return path


# =========================================================
# Save summary
# =========================================================

def save_summary(
    algorithm,
    labels,
    point_ids,
):

    output_dir = (
        ensure_result_directory(
            algorithm
        )
    )

    labels = np.asarray(
        labels
    )

    unique_labels = np.unique(
        labels
    )

    cluster_sizes = {}

    for label in unique_labels:

        label = int(label)

        cluster_sizes[
            str(label)
        ] = int(
            np.sum(
                labels == label
            )
        )

    cluster_labels = [
        int(label)
        for label in unique_labels
        if label >= 0
    ]

    noise_count = int(
        np.sum(
            labels == -1
        )
    )

    summary = {

        "algorithm": algorithm,

        "total_points": len(
            point_ids
        ),

        "cluster_count": len(
            cluster_labels
        ),

        "noise_count": noise_count,

        "cluster_sizes": cluster_sizes,
    }

    path = (
        output_dir
        / "summary.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return path


# =========================================================
# Save complete result
# =========================================================

def save_cluster_result(
    algorithm,
    labels,
    point_ids,
    payloads,
):

    labels_path = save_labels(
        algorithm,
        labels,
    )

    clusters_path = (
        save_clusters_json(
            algorithm,
            labels,
            point_ids,
            payloads,
        )
    )

    summary_path = (
        save_summary(
            algorithm,
            labels,
            point_ids,
        )
    )

    print()
    print(
        f"[RESULT] {algorithm}"
    )

    print(
        f"  labels  : {labels_path}"
    )

    print(
        f"  clusters: {clusters_path}"
    )

    print(
        f"  summary : {summary_path}"
    )

    return {
        "labels": labels_path,
        "clusters": clusters_path,
        "summary": summary_path,
    }