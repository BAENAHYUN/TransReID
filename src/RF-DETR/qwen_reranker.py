"""
Qwen3-VL Reranker

1李� 寃���:
    CLIP-ReID + InsightFace + SigLIP2
    -> 3-way Fusion Top-N

2李� 寃���:
    Qwen3-VL-Embedding
    -> Query�� 媛� �꾨낫 �대�吏��� semantic similarity 怨꾩궛
    -> 理쒖쥌 �쒖쐞 �ъ젙��

Qwen�� 3-way Fusion �먯닔�� 吏곸젒 �욎� �딄퀬
�꾨낫援곗쓣 �ъ젙�ы븯�� reranker ��븷濡� �ъ슜�쒕떎.
"""

import sys
from pathlib import Path

import numpy as np

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)
# ============================================================
# PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RFDETR_DIR = ROOT_DIR / "src" / "RF-DETR"

if str(RFDETR_DIR) not in sys.path:
    sys.path.append(str(RFDETR_DIR))


# ============================================================
# Existing modules
# ============================================================

from reid_clip import load_reid_model
from face_insight import load_face_model
from siglip_semantic import load_semantic_model, get_image_embedding

from search_qdrant import (
    load_qdrant_client,
    get_reid_embedding,
    get_face_embedding,
)

from fusion_search_3way import (
    fusion_search_3way,
)

from qwen_vlm import (
    load_embedding_model,
    get_qwen_image_embedding,
    get_qwen_text_embedding,
)


# ============================================================
# Config
# ============================================================

FIRST_STAGE_TOP_K = 50
FINAL_TOP_K = 5


# ============================================================
# Utils
# ============================================================

def cosine_similarity(a, b):
    """
    �� embedding �ъ씠 cosine similarity 怨꾩궛.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / (a_norm * b_norm)
    )


def resolve_image_path(result):
    """
    Fusion 寃곌낵�먯꽌 �ㅼ젣 �꾨낫 �대�吏� 寃쎈줈瑜� �삳뒗��.

    �곗꽑�쒖쐞:
        1. crop_path
        2. original_image
    """

    candidates = [
        result.get("crop_path"),
        result.get("original_image"),
    ]

    for value in candidates:

        if not value:
            continue

        path = Path(value)

        # �덈�寃쎈줈
        if path.exists():
            return path

        # �꾨줈�앺듃 猷⑦듃 湲곗� �곷�寃쎈줈
        candidate = ROOT_DIR / path

        if candidate.exists():
            return candidate

    return None


# ============================================================
# Clustering Intermediate Result
# ============================================================

CLUSTER_KEYS = {
    "A_KMeans": "cluster_kmeans",
    "B_DBSCAN": "cluster_dbscan",
    "C_HDBSCAN": "cluster_hdbscan",
    "D_Agglomerative": "cluster_agglomerative",
}


def get_cluster_info_from_payload(payload):
    """Qdrant payload�먯꽌 A/B/C/D �대윭�ㅽ꽣 ID瑜� 異붿텧�쒕떎."""
    if payload is None:
        payload = {}
    return {
        name: payload.get(payload_key)
        for name, payload_key in CLUSTER_KEYS.items()
    }


def find_query_point_by_filename(client, query_image_path, collection_name="forensic_persons"):
    """Query �대�吏� filename�� 湲곗��쇰줈 �꾩옱 Qdrant point瑜� 李얜뒗��."""
    query_filename = Path(query_image_path).name
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            if payload.get("filename") == query_filename:
                return point

        if next_offset is None:
            break
        offset = next_offset

    return None
def get_qdrant_payload_by_filename(
    client,
    filename,
    collection_name="forensic_persons",
):
    """
    filename�쇰줈 Qdrant point 1媛쒕� 李얠븘
    理쒖떊 payload瑜� 諛섑솚�쒕떎.
    """

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(
                        value=filename
                    ),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return None

    return points[0].payload or {}

def compare_cluster_info(query_clusters, candidate_clusters):
    """Query�� �꾨낫媛� A/B/C/D�먯꽌 媛숈� cluster�몄� 鍮꾧탳�쒕떎."""
    comparison = {}
    same_count = 0
    valid_count = 0

    for name in CLUSTER_KEYS:
        q = query_clusters.get(name)
        c = candidate_clusters.get(name)

        if q is None or c is None:
            comparison[name] = "N/A"
            continue

        if q == -1 or c == -1:
            comparison[name] = "NOISE"
            continue

        valid_count += 1
        if q == c:
            comparison[name] = "SAME"
            same_count += 1
        else:
            comparison[name] = "DIFFERENT"

    ratio = same_count / valid_count if valid_count > 0 else None
    return {
        "comparison": comparison,
        "same_count": same_count,
        "valid_count": valid_count,
        "agreement_ratio": ratio,
    }


def print_query_cluster_info(query_clusters):
    print()
    print("=" * 70)
    print("QUERY CLUSTER INFORMATION")
    print("=" * 70)
    print(f"A K-Means       : {query_clusters.get('A_KMeans')}")
    print(f"B DBSCAN        : {query_clusters.get('B_DBSCAN')}")
    print(f"C HDBSCAN       : {query_clusters.get('C_HDBSCAN')}")
    print(f"D Agglomerative : {query_clusters.get('D_Agglomerative')}")


def print_candidate_cluster_result(candidate_payload, query_clusters=None):
    candidate_clusters = get_cluster_info_from_payload(candidate_payload)

    print()
    print("--- Clustering Intermediate Result ---")
    print(f"A K-Means       : Cluster {candidate_clusters.get('A_KMeans')}")
    print(f"B DBSCAN        : Cluster {candidate_clusters.get('B_DBSCAN')}")
    print(f"C HDBSCAN       : Cluster {candidate_clusters.get('C_HDBSCAN')}")
    print(f"D Agglomerative : Cluster {candidate_clusters.get('D_Agglomerative')}")

    if query_clusters is None:
        return candidate_clusters

    result = compare_cluster_info(query_clusters, candidate_clusters)
    comparison = result["comparison"]

    print()
    print("--- Cluster Agreement ---")
    print(f"A K-Means       : {comparison['A_KMeans']}")
    print(f"B DBSCAN        : {comparison['B_DBSCAN']}")
    print(f"C HDBSCAN       : {comparison['C_HDBSCAN']}")
    print(f"D Agglomerative : {comparison['D_Agglomerative']}")
    print()

    if result["valid_count"] > 0:
        print(
            f"Agreement       : {result['same_count']} / {result['valid_count']} "
            f"({result['agreement_ratio'] * 100:.1f}%)"
        )
    else:
        print("Agreement       : N/A")

    return candidate_clusters


# ============================================================
# Qwen reranking - Image Query
# ============================================================

def rerank_with_qwen_image(
    embedding_model,
    query_image_path,
    fusion_results,
    top_k=FINAL_TOP_K,
):
    """
    �대�吏� Query 湲곕컲 Qwen reranking.

    Query image
       ��
    Qwen3-VL Image Embedding
       ��
    �꾨낫 �대�吏� 媛곴컖 Qwen Embedding
       ��
    cosine similarity
       ��
    �ъ젙��
    """

    print()
    print("=" * 70)
    print("Qwen3-VL IMAGE RERANKING")
    print("=" * 70)

    print(
        f"Query image : {query_image_path}"
    )

    print(
        f"Candidates  : {len(fusion_results)}"
    )

    query_embedding = get_qwen_image_embedding(
        embedding_model,
        str(query_image_path),
    )

    reranked = []

    total = len(fusion_results)

    for index, result in enumerate(
        fusion_results,
        start=1,
    ):

        image_path = resolve_image_path(
            result
        )

        if image_path is None:

            print(
                f"[{index}/{total}] "
                "Image path not found"
            )

            continue

        try:

            candidate_embedding = (
                get_qwen_image_embedding(
                    embedding_model,
                    str(image_path),
                )
            )

            qwen_similarity = (
                cosine_similarity(
                    query_embedding,
                    candidate_embedding,
                )
            )

            new_result = dict(
                result
            )

            new_result[
                "qwen_similarity"
            ] = qwen_similarity

            new_result[
                "qwen_image_path"
            ] = str(image_path)

            reranked.append(
                new_result
            )

            print(
                f"[{index:02d}/{total}] "
                f"{qwen_similarity:.4f} "
                f"{image_path.name}"
            )

        except Exception as e:

            print(
                f"[WARNING] "
                f"{image_path}: {e}"
            )

    reranked.sort(
        key=lambda x: x[
            "qwen_similarity"
        ],
        reverse=True,
    )

    return reranked[:top_k]


# ============================================================
# Qwen reranking - Text Query
# ============================================================

def rerank_with_qwen_text(
    embedding_model,
    text_query,
    candidates,
    top_k=FINAL_TOP_K,
):
    """
    �띿뒪�� Query 湲곕컲 Qwen reranking.

    Text
       ��
    Qwen3-VL Text Embedding
       ��
    �꾨낫 �대�吏� Qwen Embedding
       ��
    cosine similarity
       ��
    �ъ젙��
    """

    print()
    print("=" * 70)
    print("Qwen3-VL TEXT RERANKING")
    print("=" * 70)

    print(
        f'Text query : "{text_query}"'
    )

    print(
        f"Candidates : {len(candidates)}"
    )

    query_embedding = (
        get_qwen_text_embedding(
            embedding_model,
            text_query,
        )
    )

    reranked = []

    total = len(candidates)

    for index, result in enumerate(
        candidates,
        start=1,
    ):

        image_path = resolve_image_path(
            result
        )

        if image_path is None:

            print(
                f"[{index}/{total}] "
                "Image path not found"
            )

            continue

        try:

            image_embedding = (
                get_qwen_image_embedding(
                    embedding_model,
                    str(image_path),
                )
            )

            qwen_similarity = (
                cosine_similarity(
                    query_embedding,
                    image_embedding,
                )
            )

            new_result = dict(
                result
            )

            new_result[
                "qwen_similarity"
            ] = qwen_similarity

            new_result[
                "qwen_image_path"
            ] = str(image_path)

            reranked.append(
                new_result
            )

            print(
                f"[{index:02d}/{total}] "
                f"{qwen_similarity:.4f} "
                f"{image_path.name}"
            )

        except Exception as e:

            print(
                f"[WARNING] "
                f"{image_path}: {e}"
            )

    reranked.sort(
        key=lambda x: x[
            "qwen_similarity"
        ],
        reverse=True,
    )

    return reranked[:top_k]


# ============================================================
# Print
# ============================================================

def print_reranked_results(
    results,
    client,
    query_clusters=None,
    title="Qwen3-VL Final Results",
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()

        print(
            f"[Rank {rank}]"
        )

        print(
            f"Qwen Similarity : "
            f"{result.get('qwen_similarity', 0):.4f}"
        )

        print(
            f"3-way Score     : "
            f"{result.get('fusion_score', 'N/A')}"
        )

        print(
            f"Re-ID           : "
            f"{result.get('reid_similarity', 'N/A')}"
        )

        print(
            f"Face            : "
            f"{result.get('face_similarity', 'N/A')}"
        )

        print(
            f"SigLIP2         : "
            f"{result.get('semantic_similarity', 'N/A')}"
        )

        print(
            f"Original Image  : "
            f"{result.get('original_image')}"
        )

        print(
            f"Crop            : "
            f"{result.get('crop_path')}"
        )

        # --------------------------------------------
        # 理쒖떊 Qdrant payload �ㅼ떆 議고쉶
        # --------------------------------------------

        image_path = (
            result.get("crop_path")
            or result.get("original_image")
        )

        if image_path:

            filename = Path(
                image_path
            ).name

            candidate_payload = (
                get_qdrant_payload_by_filename(
                    client,
                    filename,
                )
            )

        else:

            candidate_payload = None

        # --------------------------------------------
        # Cluster intermediate result
        # --------------------------------------------

        if candidate_payload:

            print_candidate_cluster_result(
                candidate_payload,
                query_clusters,
            )

        else:

            print()
            print(
                "--- Clustering Intermediate Result ---"
            )
            print(
                "Qdrant payload not found."
            )
# ============================================================
# Full Image Pipeline
# ============================================================

def image_query_pipeline(
    query_image_path,
):

    query_image_path = Path(
        query_image_path
    )

    if not query_image_path.exists():

        raise FileNotFoundError(
            f"Query image not found: "
            f"{query_image_path}"
        )

    print()
    print("=" * 80)
    print("FORENSIC IMAGE SEARCH")
    print("3-WAY FUSION -> QWEN RERANK")
    print("=" * 80)

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()
    print("[1/5] Loading CLIP-ReID...")

    reid_model, reid_device = (
        load_reid_model()
    )

    print()
    print("[2/5] Loading InsightFace...")

    face_model = (
        load_face_model()
    )

    print()
    print("[3/5] Loading SigLIP2...")

    (
        semantic_model,
        semantic_processor,
        semantic_device,
    ) = load_semantic_model()

    print()
    print("[4/5] Connecting Qdrant...")

    client = (
        load_qdrant_client()
    )

    # --------------------------------------------------------
    # Query cluster information
    # --------------------------------------------------------

    query_point = find_query_point_by_filename(
        client,
        query_image_path,
    )

    if query_point is not None:
        query_clusters = get_cluster_info_from_payload(
            query_point.payload
        )
        print_query_cluster_info(query_clusters)
    else:
        query_clusters = None
        print()
        print("[WARNING] Query image was not found inside Qdrant.")

    # --------------------------------------------------------
    # Query embeddings
    # --------------------------------------------------------

    print()
    print(
        "[5/5] Extracting query embeddings..."
    )

    reid_embedding = (
        get_reid_embedding(
            reid_model,
            reid_device,
            query_image_path,
        )
    )

    face_embedding = (
        get_face_embedding(
            face_model,
            query_image_path,
        )
    )

    semantic_embedding = (
        get_image_embedding(
            semantic_model,
            semantic_processor,
            semantic_device,
            str(query_image_path),
        )
    )

    # --------------------------------------------------------
    # First stage
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 1 - 3-WAY FUSION")
    print("=" * 80)

    fusion_results = (
        fusion_search_3way(
            client,
            reid_embedding,
            face_embedding,
            semantic_embedding,
            top_k=FIRST_STAGE_TOP_K,
        )
    )

    print(
        f"3-way candidates: "
        f"{len(fusion_results)}"
    )

    # --------------------------------------------------------
    # 硫붾え由� �뺣━ �� Qwen load
    # --------------------------------------------------------

    print()
    print(
        "Loading Qwen3-VL Embedding model..."
    )

    qwen_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # Second stage
    # --------------------------------------------------------

    final_results = (
        rerank_with_qwen_image(
            qwen_model,
            query_image_path,
            fusion_results,
            top_k=FINAL_TOP_K,
        )
    )

    print_reranked_results(
        final_results,
        client=client,
        query_clusters=query_clusters,
        title=(
            "FINAL IMAGE SEARCH RESULTS "
            "(3-Way Fusion + Qwen Reranking)"
        ),
    )

    return final_results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    TEST_QUERY_IMAGE = (
        ROOT_DIR
        / "data"
        / "Market-1501-v15.09.15"
        / "bounding_box_test"
        / "0001_c1s1_001051_03.jpg"
    )

    image_query_pipeline(
        TEST_QUERY_IMAGE
    )