import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# =========================================================
# Paths
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    ROOT_DIR
    / "data"
    / "cluster_results"
)

OUTPUT_FILE = (
    RESULT_DIR
    / "cluster_comparison.json"
)


CLUSTERS = {
    "A": {
        "name": "K-Means",
        "dir": "cluster_A",
    },

    "B": {
        "name": "DBSCAN",
        "dir": "cluster_B",
    },

    "C": {
        "name": "HDBSCAN",
        "dir": "cluster_C",
    },

    "D": {
        "name": "Agglomerative",
        "dir": "cluster_D",
    },
}


# =========================================================
# JSON loader
# =========================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# =========================================================
# Recursive market_id finder
# =========================================================

def find_market_id(obj):

    if isinstance(obj, dict):

        if "market_id" in obj:

            try:
                return int(
                    obj["market_id"]
                )

            except Exception:
                pass

        # payload 안에 있을 수도 있음
        if "payload" in obj:

            result = find_market_id(
                obj["payload"]
            )

            if result is not None:
                return result

        for value in obj.values():

            result = find_market_id(
                value
            )

            if result is not None:
                return result

    return None


# =========================================================
# Convert clusters.json -> y_true / y_pred
# =========================================================

def load_cluster_labels(
    clusters_path,
):

    data = load_json(
        clusters_path
    )

    y_true = []
    y_pred = []

    missing_market_id = 0

    # -----------------------------------------------------
    # Expected structure:
    #
    # {
    #   "0": [ ... ],
    #   "1": [ ... ],
    #   "-1": [ ... ]
    # }
    # -----------------------------------------------------

    if isinstance(data, dict):

        for cluster_key, members in data.items():

            try:

                cluster_id = int(
                    cluster_key
                )

            except ValueError:

                # 혹시 상위 metadata key가 있으면 skip
                continue

            if not isinstance(
                members,
                list,
            ):

                continue

            for member in members:

                market_id = (
                    find_market_id(
                        member
                    )
                )

                if market_id is None:

                    missing_market_id += 1

                    continue

                y_true.append(
                    market_id
                )

                y_pred.append(
                    cluster_id
                )

    return (
        np.asarray(
            y_true,
            dtype=np.int32,
        ),

        np.asarray(
            y_pred,
            dtype=np.int32,
        ),

        missing_market_id,
    )


# =========================================================
# Purity
# =========================================================

def calculate_purity(
    true_labels,
    cluster_labels,
):

    # DBSCAN/HDBSCAN의 -1 noise는
    # purity 계산에서 제외
    valid_mask = (
        cluster_labels != -1
    )

    true_labels = (
        true_labels[
            valid_mask
        ]
    )

    cluster_labels = (
        cluster_labels[
            valid_mask
        ]
    )

    if len(
        true_labels
    ) == 0:

        return 0.0

    correct = 0

    for cluster_id in np.unique(
        cluster_labels
    ):

        mask = (
            cluster_labels
            == cluster_id
        )

        ids = (
            true_labels[
                mask
            ]
        )

        if len(ids) == 0:
            continue

        _, counts = np.unique(
            ids,
            return_counts=True,
        )

        correct += int(
            counts.max()
        )

    return (
        correct
        / len(true_labels)
    )


# =========================================================
# Load one algorithm
# =========================================================

def evaluate_cluster(
    cluster_id,
):

    info = CLUSTERS[
        cluster_id
    ]

    cluster_dir = (
        RESULT_DIR
        / info["dir"]
    )

    summary_path = (
        cluster_dir
        / "summary.json"
    )

    clusters_path = (
        cluster_dir
        / "clusters.json"
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = load_json(
        summary_path
    )

    cluster_count = int(
        summary.get(
            "cluster_count",
            0,
        )
    )

    noise_count = int(
        summary.get(
            "noise_count",
            0,
        )
    )

    total_points = int(
        summary.get(
            "total_points",
            0,
        )
    )

    # -----------------------------------------------------
    # Real cluster assignments
    # -----------------------------------------------------

    (
        y_true,
        y_pred,
        missing_market_id,
    ) = load_cluster_labels(
        clusters_path
    )
    # Market1501 junk / distractor ID 제외
    valid_gt = y_true > 0

    y_true = y_true[valid_gt]
    y_pred = y_pred[valid_gt]
    
    if len(y_true) == 0:

        raise RuntimeError(
            f"{cluster_id}: "
            "No market_id found "
            "in clusters.json"
        )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    purity = calculate_purity(
        y_true,
        y_pred,
    )

    ari = adjusted_rand_score(
        y_true,
        y_pred,
    )

    nmi = (
        normalized_mutual_info_score(
            y_true,
            y_pred,
        )
    )

    return {

        "cluster": cluster_id,

        "algorithm": (
            info["name"]
        ),

        "total_points": (
            total_points
        ),

        "evaluated_points": int(
            len(y_true)
        ),

        "missing_market_id": (
            missing_market_id
        ),

        "clusters": (
            cluster_count
        ),

        "noise": (
            noise_count
        ),

        "purity": float(
            purity
        ),

        "ari": float(
            ari
        ),

        "nmi": float(
            nmi
        ),
    }


# =========================================================
# Main
# =========================================================

def main():

    print(
        "=" * 85
    )

    print(
        "CLUSTERING RESULT COMPARISON"
    )

    print(
        "=" * 85
    )

    results = []

    for cluster_id in CLUSTERS:

        print()

        print(
            f"[{cluster_id}] "
            f"Evaluating "
            f"{CLUSTERS[cluster_id]['name']}..."
        )

        result = (
            evaluate_cluster(
                cluster_id
            )
        )

        results.append(
            result
        )

        print(
            f"  evaluated : "
            f"{result['evaluated_points']}"
        )

        print(
            f"  missing   : "
            f"{result['missing_market_id']}"
        )

    # =====================================================
    # Table
    # =====================================================

    print()

    print(
        "=" * 85
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 85
    )

    print(
        f"{'ID':<5}"
        f"{'Algorithm':<20}"
        f"{'Clusters':>10}"
        f"{'Noise':>10}"
        f"{'Purity':>12}"
        f"{'ARI':>12}"
        f"{'NMI':>12}"
    )

    print(
        "-" * 85
    )

    for r in results:

        print(
            f"{r['cluster']:<5}"
            f"{r['algorithm']:<20}"
            f"{r['clusters']:>10}"
            f"{r['noise']:>10}"
            f"{r['purity']:>12.4f}"
            f"{r['ari']:>12.4f}"
            f"{r['nmi']:>12.4f}"
        )

    # =====================================================
    # Best
    # =====================================================

    best_purity = max(
        results,
        key=lambda x: x["purity"],
    )

    best_ari = max(
        results,
        key=lambda x: x["ari"],
    )

    best_nmi = max(
        results,
        key=lambda x: x["nmi"],
    )

    print()

    print(
        "=" * 85
    )

    print(
        "BEST RESULTS"
    )

    print(
        "=" * 85
    )

    print()

    print(
        "Best Purity"
    )

    print(
        f"  {best_purity['cluster']} "
        f"{best_purity['algorithm']} "
        f"= {best_purity['purity']:.4f}"
    )

    print()

    print(
        "Best ARI"
    )

    print(
        f"  {best_ari['cluster']} "
        f"{best_ari['algorithm']} "
        f"= {best_ari['ari']:.4f}"
    )

    print()

    print(
        "Best NMI"
    )

    print(
        f"  {best_nmi['cluster']} "
        f"{best_nmi['algorithm']} "
        f"= {best_nmi['nmi']:.4f}"
    )

    # =====================================================
    # Save
    # =====================================================

    output = {

        "dataset": (
            "Market-1501"
        ),

        "embedding": (
            "CLIP-ReID 1280D"
        ),

        "results": (
            results
        ),

        "best": {

            "purity": (
                best_purity
            ),

            "ari": (
                best_ari
            ),

            "nmi": (
                best_nmi
            ),
        },
    }

    with open(
        OUTPUT_FILE,
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
        "=" * 85
    )

    print(
        "COMPARISON SAVED"
    )

    print(
        "=" * 85
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()