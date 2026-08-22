from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


ROOT_DIR = Path(__file__).resolve().parents[2]

QDRANT_URL = "http://127.0.0.1:6333"

COLLECTION_NAME = "forensic_persons"


def get_client():
    return QdrantClient(QDRANT_URL)


def load_reid_vectors(
    client,
    source=None,
):
    """
    Qdrant에서 Re-ID 1280D 벡터 전체를 가져온다.

    Returns
    -------
    vectors : np.ndarray
        shape = (N, 1280)

    point_ids : list
        Qdrant point ID

    payloads : list
        각 point의 payload
    """

    vectors = []
    point_ids = []
    payloads = []

    offset = None

    while True:

        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=True,
        )

        if not records:
            break

        for record in records:

            # source filter
            if source is not None:

                payload = record.payload or {}

                if payload.get(
                    "source"
                ) != source:

                    continue

            vector = record.vector

            # named vector
            if isinstance(
                vector,
                dict,
            ):

                vector = vector.get(
                    "reid"
                )

            if vector is None:
                continue

            vector = np.asarray(
                vector,
                dtype=np.float32,
            )

            if vector.shape != (
                1280,
            ):
                continue

            vectors.append(
                vector
            )

            point_ids.append(
                record.id
            )

            payloads.append(
                record.payload or {}
            )

        if next_offset is None:
            break

        offset = next_offset

    if not vectors:

        raise RuntimeError(
            "No Re-ID vectors found."
        )

    X = np.vstack(
        vectors
    )

    return (
        X,
        point_ids,
        payloads,
    )


def load_face_vectors(
    client,
    source=None,
):
    """
    Qdrant에서 Face 512D 벡터 전체를 가져온다.

    Face vector가 없는 point는 제외한다.
    """

    vectors = []
    point_ids = []
    payloads = []

    offset = None

    while True:

        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=True,
        )

        if not records:
            break

        for record in records:

            payload = (
                record.payload or {}
            )

            if source is not None:

                if payload.get(
                    "source"
                ) != source:

                    continue

            vector = record.vector

            if isinstance(
                vector,
                dict,
            ):

                vector = vector.get(
                    "face"
                )

            if vector is None:
                continue

            vector = np.asarray(
                vector,
                dtype=np.float32,
            )

            if vector.shape != (
                512,
            ):
                continue

            vectors.append(
                vector
            )

            point_ids.append(
                record.id
            )

            payloads.append(
                payload
            )

        if next_offset is None:
            break

        offset = next_offset

    if not vectors:

        raise RuntimeError(
            "No Face vectors found."
        )

    X = np.vstack(
        vectors
    )

    return (
        X,
        point_ids,
        payloads,
    )


def normalize_vectors(X):
    """
    L2 normalization.
    """

    norms = np.linalg.norm(
        X,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return X / norms