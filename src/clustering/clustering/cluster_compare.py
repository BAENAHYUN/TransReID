import json
from pathlib import Path


# =========================================================
# Paths
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT_DIR / "data" / "cluster_results"


# =========================================================
# Cluster definitions
# =========================================================

CLUSTERS = {
    "A": RESULT_DIR / "cluster_A",
    "B": RESULT_DIR / "cluster_B",
    "C": RESULT_DIR / "cluster_C",
    "D": RESULT_DIR / "cluster_D",
}


# =========================================================
# JSON loader
# =========================================================

def load_json(path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Summary loader
# =========================================================

def load_summary(cluster_name, cluster_dir):

    summary_path = cluster_dir / "summary.json"

    summary = load_json(summary_path)

    if summary is None:
        print(f"[WARNING] {cluster_name}: summary.json not found")
        return None

    return summary


# =========================================================
# Tuning loader
# =========================================================

def load_tuning(cluster_name, cluster_dir):

    tuning_files = [
        "kmeans_tuning.json",
        "dbscan_tuning.json",
        "hdbscan_tuning.json",
        "agglomerative_tuning.json",
    ]

    for filename in tuning_files:

        path = cluster_dir / filename

        if path.exists():
            data = load_json(path)

            if data is not None:
                return filename, data

    print(f"[WARNING] {cluster_name}: tuning result not found")

    return None, None


# =========================================================
# Metric extraction
# =========================================================

def get_metric(summary, key, default=None):

    if key in summary:
        return summary[key]

    evaluation = summary.get("evaluation", {})

    if key in evaluation:
        return evaluation[key]

    return default


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 80)
    print("CLUSTERING RESULT COMPARISON")
    print("=" * 80)

    results = []

    for cluster_name, cluster_dir in CLUSTERS.items():

        print()
        print(f"[{cluster_name}] Loading results...")

        summary = load_summary(
            cluster_name,
            cluster_dir,
        )

        if summary is None:
            continue

        tuning_file, tuning = load_tuning(
            cluster_name,
            cluster_dir,
        )

        algorithm = summary.get(
            "algorithm",
            f"Cluster {cluster_name}",
        )

        clusters = summary.get(
            "clusters",
            summary.get("n_clusters", 0),
        )

        noise = summary.get(
            "noise",
            0,
        )

        purity = get_metric(
            summary,
            "purity",
            0.0,
        )

        ari = get_metric(
            summary,
            "ari",
            0.0,
        )

        nmi = get_metric(
            summary,
            "nmi",
            0.0,
        )

        results.append({
            "cluster": cluster_name,
            "algorithm": algorithm,
            "clusters": clusters,
            "noise": noise,
            "purity": purity,
            "ari": ari,
            "nmi": nmi,
            "tuning_file": tuning_file,
        })

    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f"{'Cluster':<10}"
        f"{'Algorithm':<22}"
        f"{'Clusters':>10}"
        f"{'Noise':>10}"
        f"{'Purity':>10}"
        f"{'ARI':>10}"
        f"{'NMI':>10}"
    )

    print("-" * 80)

    for r in results:

        print(
            f"{r['cluster']:<10}"
            f"{r['algorithm']:<22}"
            f"{r['clusters']:>10}"
            f"{r['noise']:>10}"
            f"{r['purity']:>10.4f}"
            f"{r['ari']:>10.4f}"
            f"{r['nmi']:>10.4f}"
        )

    # =====================================================
    # Best results
    # =====================================================

    if results:

        best_ari = max(
            results,
            key=lambda x: x["ari"],
        )

        best_nmi = max(
            results,
            key=lambda x: x["nmi"],
        )

        best_purity = max(
            results,
            key=lambda x: x["purity"],
        )

        print()
        print("=" * 80)
        print("BEST RESULTS")
        print("=" * 80)

        print()
        print("Best ARI")
        print(
            f"  Cluster : {best_ari['cluster']}"
        )
        print(
            f"  Method  : {best_ari['algorithm']}"
        )
        print(
            f"  ARI     : {best_ari['ari']:.4f}"
        )

        print()
        print("Best NMI")
        print(
            f"  Cluster : {best_nmi['cluster']}"
        )
        print(
            f"  Method  : {best_nmi['algorithm']}"
        )
        print(
            f"  NMI     : {best_nmi['nmi']:.4f}"
        )

        print()
        print("Best Purity")
        print(
            f"  Cluster : {best_purity['cluster']}"
        )
        print(
            f"  Method  : {best_purity['algorithm']}"
        )
        print(
            f"  Purity  : {best_purity['purity']:.4f}"
        )

    # =====================================================
    # Save comparison
    # =====================================================

    output_path = RESULT_DIR / "cluster_comparison.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("COMPARISON SAVED")
    print("=" * 80)

    print(output_path)


if __name__ == "__main__":
    main()