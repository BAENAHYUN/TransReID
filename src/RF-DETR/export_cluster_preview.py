from pathlib import Path
import json
import shutil


ROOT_DIR = Path(__file__).resolve().parents[2]

CLUSTER_FILES = {
    "A_KMeans": (
        ROOT_DIR
        / "data"
        / "cluster_results"
        / "cluster_A"
        / "clusters.json"
    ),

    "B_DBSCAN": (
        ROOT_DIR
        / "data"
        / "cluster_results"
        / "cluster_B"
        / "clusters.json"
    ),

    "C_HDBSCAN": (
        ROOT_DIR
        / "data"
        / "cluster_results"
        / "cluster_C"
        / "clusters.json"
    ),

    "D_Agglomerative": (
        ROOT_DIR
        / "data"
        / "cluster_results"
        / "cluster_D"
        / "clusters.json"
    ),
}


OUTPUT_ROOT = (
    ROOT_DIR
    / "data"
    / "cluster_preview"
)


def export_cluster(
    algorithm_name,
    cluster_id,
):
    json_path = CLUSTER_FILES[
        algorithm_name
    ]

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    cluster_key = str(
        cluster_id
    )

    if cluster_key not in data:
        print(
            f"[ERROR] "
            f"{algorithm_name} "
            f"Cluster {cluster_id} not found"
        )
        return

    items = data[
        cluster_key
    ]

    output_dir = (
        OUTPUT_ROOT
        / algorithm_name
        / f"cluster_{cluster_id}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = 0
    missing = 0

    for item in items:

        image_path = item.get(
            "original_image"
        )

        if not image_path:
            continue

        source = (
            ROOT_DIR
            / image_path
        )

        if not source.exists():

            print(
                f"[MISSING] {source}"
            )

            missing += 1
            continue

        destination = (
            output_dir
            / source.name
        )

        shutil.copy2(
            source,
            destination,
        )

        copied += 1

    print()
    print("=" * 70)
    print(
        f"{algorithm_name} "
        f"Cluster {cluster_id}"
    )
    print("=" * 70)

    print(
        f"Total   : {len(items)}"
    )

    print(
        f"Copied  : {copied}"
    )

    print(
        f"Missing : {missing}"
    )

    print(
        f"Folder  : {output_dir}"
    )


if __name__ == "__main__":

    # 吏�湲� Query 寃곌낵�먯꽌 �섏삩 cluster��

    export_cluster(
        "A_KMeans",
        89,
    )

    export_cluster(
        "B_DBSCAN",
        274,
    )

    export_cluster(
        "C_HDBSCAN",
        1383,
    )

    export_cluster(
        "D_Agglomerative",
        526,
    )