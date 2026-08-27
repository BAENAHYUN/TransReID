from pathlib import Path
import uuid
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from reid_clip import load_reid_model, get_embedding
from face_insight import load_face_model, detect_faces
from siglip_semantic import (
    load_semantic_model,
    get_image_embedding,
)


ROOT_DIR = Path(__file__).resolve().parents[2]

MARKET_ROOT = (
    ROOT_DIR
    / "data"
    / "Market-1501-v15.09.15"
)

MARKET_DIRS = [
    MARKET_ROOT / "bounding_box_train",
    MARKET_ROOT / "bounding_box_test",
]

COLLECTION_NAME = "forensic_persons"

QDRANT_URL = "http://127.0.0.1:6333"

BATCH_SIZE = 64


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)

    if norm == 0:
        return x

    return x / norm


def parse_market_id(filename):
    """
    Market1501 filename:
    0002_c1s1_000451_03.jpg

    �� �レ옄瑜� identity濡� �ъ슜.
    """

    try:
        return int(filename.split("_")[0])
    except Exception:
        return -999


def get_face_embedding(face_model, image_path):

    faces = detect_faces(
        face_model,
        image_path,
    )

    if not faces:
        return None

    face = faces[0]

    if hasattr(face, "normed_embedding"):
        emb = face.normed_embedding
    elif hasattr(face, "embedding"):
        emb = face.embedding
    else:
        return None

    return normalize(emb)


def main():

    client = QdrantClient(
        url=QDRANT_URL
    )

    print("=" * 70)
    print("MARKET1501 -> QDRANT BUILD")
    print("Person + Face + Semantic")
    print("=" * 70)

    print("\nLoading CLIP-ReID...")
    reid_model, reid_device = load_reid_model()

    print("\nLoading InsightFace...")
    face_model = load_face_model()

    print("\nLoading SigLIP2...")
    (
        semantic_model,
        semantic_processor,
        semantic_device,
    ) = load_semantic_model()

    image_paths = []

    for folder in MARKET_DIRS:

        if not folder.exists():
            print(
                f"[WARNING] Missing: {folder}"
            )
            continue

        image_paths.extend(
            sorted(folder.glob("*.jpg"))
        )

    print()
    print(
        f"Found images: {len(image_paths)}"
    )

    points = []

    success = 0
    failed = 0
    face_count = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        try:

            # -----------------------------------------
            # CLIP-ReID
            # -----------------------------------------

            reid_embedding = get_embedding(
                reid_model,
                reid_device,
                image_path,
            )

            reid_embedding = normalize(
                reid_embedding
            )

            # -----------------------------------------
            # Face
            # -----------------------------------------

            face_embedding = get_face_embedding(
                face_model,
                image_path,
            )

            if face_embedding is not None:
                face_count += 1

            # -----------------------------------------
            # SigLIP2
            # -----------------------------------------

            semantic_embedding = (
                get_image_embedding(
                    semantic_model,
                    semantic_processor,
                    semantic_device,
                    str(image_path),
                )
            )

            semantic_embedding = normalize(
                semantic_embedding
            )

            # -----------------------------------------
            # Named vectors
            # -----------------------------------------

            vectors = {
                "reid": reid_embedding.tolist(),
                "semantic": semantic_embedding.tolist(),
            }

            if face_embedding is not None:
                vectors["face"] = (
                    face_embedding.tolist()
                )

            # -----------------------------------------
            # Payload
            # -----------------------------------------

            market_id = parse_market_id(
                image_path.name
            )

            split = (
                "train"
                if "bounding_box_train"
                in str(image_path)
                else "test"
            )

            payload = {
                "source": "market1501",
                "market_id": market_id,
                "split": split,
                "original_image": str(
                    image_path.relative_to(
                        ROOT_DIR
                    )
                ),
                "crop_path": str(
                    image_path.relative_to(
                        ROOT_DIR
                    )
                ),
                "filename": image_path.name,
            }

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload=payload,
            )

            points.append(point)

            success += 1

            if len(points) >= BATCH_SIZE:

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True,
                )

                points.clear()

            if (
                index % 100 == 0
                or index == len(image_paths)
            ):

                print(
                    f"[{index}/{len(image_paths)}] "
                    f"success={success} "
                    f"failed={failed} "
                    f"faces={face_count}"
                )

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] "
                f"{image_path.name}: {e}"
            )

    if points:

        print(
            f"\n[Qdrant] "
            f"Uploading last "
            f"{len(points)} points..."
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

    print()
    print("=" * 70)
    print("MARKET1501 BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Total images : "
        f"{len(image_paths)}"
    )

    print(
        f"Success      : "
        f"{success}"
    )

    print(
        f"Failed       : "
        f"{failed}"
    )

    print(
        f"Faces        : "
        f"{face_count}"
    )

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )

    print(
        f"Qdrant count : "
        f"{count.count}"
    )


if __name__ == "__main__":
    main()