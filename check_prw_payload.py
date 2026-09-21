from qdrant_client import QdrantClient
from collections import Counter

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "forensic_person"

SAMPLE_LIMIT = 10


def main():
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT
    )

    print("=" * 80)
    print("PRW PAYLOAD CHECK")
    print("=" * 80)

    # 컬렉션 존재 여부 확인
    try:
        info = client.get_collection(COLLECTION_NAME)
        print(f"[OK] collection: {COLLECTION_NAME}")
        print(f"[INFO] points_count: {info.points_count}")
    except Exception as e:
        print(f"[ERROR] collection access failed: {e}")
        return

    print()

    # 일부 point 조회
    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SAMPLE_LIMIT,
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        print(f"[ERROR] scroll failed: {e}")
        return

    if not points:
        print("[WARN] No points found.")
        return

    print(f"[INFO] showing {len(points)} sample points")
    print()

    key_counter = Counter()

    for idx, point in enumerate(points, 1):
        payload = point.payload or {}

        print("-" * 80)
        print(f"[{idx}] POINT ID: {point.id}")

        if not payload:
            print("payload: EMPTY")
            continue

        for key, value in payload.items():
            key_counter[key] += 1
            print(f"{key}: {value}")

    print()
    print("=" * 80)
    print("PAYLOAD KEY SUMMARY")
    print("=" * 80)

    for key, count in sorted(key_counter.items()):
        print(f"{key:<30} : {count}/{len(points)}")

    print()
    print("=" * 80)
    print("PRW RE-ID REQUIRED FIELD CHECK")
    print("=" * 80)

    # 흔히 사용할 수 있는 후보 key들
    candidate_groups = {
        "person_id": [
            "pid",
            "person_id",
            "identity",
            "identity_id",
            "label"
        ],
        "camera_id": [
            "camid",
            "camera_id",
            "camera",
            "cam"
        ],
        "image/crop path": [
            "crop_path",
            "image_path",
            "path",
            "file_path",
            "image_name",
            "filename"
        ],
        "source": [
            "source",
            "dataset",
            "source_type"
        ]
    }

    all_keys = set(key_counter.keys())

    for field_name, candidates in candidate_groups.items():
        found = [k for k in candidates if k in all_keys]

        if found:
            print(f"[OK] {field_name:<20} -> {found}")
        else:
            print(f"[MISSING?] {field_name:<20} -> candidate key not found")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
