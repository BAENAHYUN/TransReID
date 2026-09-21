from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333", timeout=120)

collections = [
    ("forensic_person", {
        "siglip2": 768,
        "irra": 512,
        "solider": 1024,
    }),
    ("forensic_object", {
        "siglip2": 768,
        "dinov2": 1536,
    }),
]

for collection, expected_dims in collections:
    print("=" * 88)
    print(collection)
    print("=" * 88)

    offset = None
    total = 0
    bad_payload = 0
    bad_vectors = 0
    dims_seen = {}

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=32,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        for p in points:
            payload = p.payload or {}

            if payload.get("source") != "final_db_candidates":
                continue

            total += 1

            required = [
                "video_stem",
                "detection_id",
                "crop_path",
                "frame_idx",
                "track_id",
                "selected_track_id",
                "selected_rank",
                "candidate_scope",
            ]

            missing = [
                k for k in required
                if k not in payload
            ]

            if missing:
                bad_payload += 1
                print(
                    "[BAD PAYLOAD]",
                    p.id,
                    "missing=",
                    missing,
                )

            vectors = p.vector or {}

            actual_names = set(vectors.keys())
            expected_names = set(expected_dims.keys())

            if actual_names != expected_names:
                bad_vectors += 1
                print(
                    "[BAD VECTOR NAMES]",
                    p.id,
                    "actual=",
                    sorted(actual_names),
                    "expected=",
                    sorted(expected_names),
                )

            for name, expected_dim in expected_dims.items():
                if name not in vectors:
                    continue

                dim = len(vectors[name])
                dims_seen.setdefault(name, set()).add(dim)

                if dim != expected_dim:
                    bad_vectors += 1
                    print(
                        "[BAD DIM]",
                        p.id,
                        name,
                        dim,
                        "expected",
                        expected_dim,
                    )

        if offset is None:
            break

    print()
    print("final_db_candidates points :", total)
    print("payload errors             :", bad_payload)
    print("vector errors              :", bad_vectors)
    print("dims seen                  :", dims_seen)
    print()



