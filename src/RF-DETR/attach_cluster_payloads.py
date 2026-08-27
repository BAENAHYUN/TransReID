from pathlib import Path
import json

from qdrant_client import QdrantClient


ROOT_DIR = Path(__file__).resolve().parents[2]

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "forensic_persons"


CLUSTER_FILES = {
    "cluster_kmeans":
        ROOT_DIR
        / "data"
        / "cluster_results"
        / "cluster_A"
        / "clusters.json",

    "cluster_dbscan":
        ROOT_DIR
        / "data"
        / "cluster_results"
        / "cluster_B"
        / "clusters.json",

    "cluster_hdbscan":
        ROOT_DIR
        / "data"
        / "cluster_results"
        / "cluster_C"
        / "clusters.json",

    "cluster_agglomerative":
        ROOT_DIR
        / "data"
        / "cluster_results"
        / "cluster_D"
        / "clusters.json",
}


def normalize_filename(value):
    return Path(str(value)).name


def load_cluster_json(path):
    """
    �꾩옱 clusters.json �뺤떇:

    {
        "62": [
            {
                "point_id": "...",
                "filename": "0727_c4s4_001235_04.jpg",
                ...
            },
            ...
        ],

        "63": [
            ...
        ]
    }

    諛섑솚:

    {
        "0727_c4s4_001235_04.jpg": 62,
        ...
    }
    """

    if not path.exists():
        print(f"[WARNING] File not found: {path}")
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    mapping = {}

    for cluster_id_str, items in data.items():

        try:
            cluster_id = int(
                cluster_id_str
            )
        except ValueError:
            print(
                f"[WARNING] Invalid cluster id: "
                f"{cluster_id_str}"
            )
            continue

        if not isinstance(items, list):
            continue

        for item in items:

            filename = item.get(
                "filename"
            )

            if not filename:

                filename = item.get(
                    "original_image"
                )

            if not filename:
                continue

            filename = normalize_filename(
                filename
            )

            mapping[
                filename
            ] = cluster_id

    return mapping


def load_all_clusters():

    all_clusters = {}

    print()
    print("=" * 70)
    print("LOADING CLUSTER RESULTS")
    print("=" * 70)

    for payload_name, path in (
        CLUSTER_FILES.items()
    ):

        print()
        print(
            f"[{payload_name}]"
        )

        print(
            f"File: {path}"
        )

        mapping = load_cluster_json(
            path
        )

        all_clusters[
            payload_name
        ] = mapping

        print(
            f"Images loaded: "
            f"{len(mapping)}"
        )

        if mapping:

            sample = next(
                iter(mapping.items())
            )

            print(
                f"Sample: "
                f"{sample[0]} "
                f"-> Cluster {sample[1]}"
            )

    return all_clusters


def main():

    print("=" * 70)
    print(
        "ATTACH A/B/C/D CLUSTER RESULTS "
        "TO QDRANT"
    )
    print("=" * 70)

    client = QdrantClient(
        url=QDRANT_URL
    )

    cluster_maps = (
        load_all_clusters()
    )

    offset = None

    processed = 0
    updated = 0
    no_cluster = 0

    # �뚭퀬由ъ쬁蹂� 留ㅼ묶 �듦퀎
    matched_counts = {
        name: 0
        for name
        in CLUSTER_FILES
    }

    while True:

        points, next_offset = (
            client.scroll(
                collection_name=
                    COLLECTION_NAME,

                limit=256,
                offset=offset,

                with_payload=True,
                with_vectors=False,
            )
        )

        if not points:
            break

        for point in points:

            processed += 1

            payload = (
                point.payload
                or {}
            )

            filename = (
                payload.get(
                    "filename"
                )
                or payload.get(
                    "crop_path"
                )
                or payload.get(
                    "original_image"
                )
            )

            if not filename:

                no_cluster += 1
                continue

            filename = (
                normalize_filename(
                    filename
                )
            )

            new_payload = {}

            for (
                payload_name,
                cluster_map,
            ) in cluster_maps.items():

                if (
                    filename
                    in cluster_map
                ):

                    cluster_id = (
                        cluster_map[
                            filename
                        ]
                    )

                    new_payload[
                        payload_name
                    ] = cluster_id

                    matched_counts[
                        payload_name
                    ] += 1

            if not new_payload:

                no_cluster += 1
                continue

            client.set_payload(
                collection_name=
                    COLLECTION_NAME,

                payload=
                    new_payload,

                points=[
                    point.id
                ],
            )

            updated += 1

            if (
                processed % 1000
                == 0
            ):

                print(
                    f"[{processed}] "
                    f"updated={updated} "
                    f"no_cluster={no_cluster}"
                )

        if next_offset is None:
            break

        offset = next_offset

    print()
    print("=" * 70)
    print("ATTACH COMPLETE")
    print("=" * 70)

    print(
        f"Processed        : "
        f"{processed}"
    )

    print(
        f"Updated          : "
        f"{updated}"
    )

    print(
        f"No cluster       : "
        f"{no_cluster}"
    )

    print()
    print(
        "Algorithm matches"
    )

    print(
        f"A KMeans         : "
        f"{matched_counts['cluster_kmeans']}"
    )

    print(
        f"B DBSCAN         : "
        f"{matched_counts['cluster_dbscan']}"
    )

    print(
        f"C HDBSCAN        : "
        f"{matched_counts['cluster_hdbscan']}"
    )

    print(
        f"D Agglomerative  : "
        f"{matched_counts['cluster_agglomerative']}"
    )


if __name__ == "__main__":
    main()