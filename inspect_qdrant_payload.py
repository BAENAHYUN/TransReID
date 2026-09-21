from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

for collection in ["forensic_person", "forensic_object"]:
    print("=" * 80)
    print(collection)
    print("=" * 80)

    points, _ = client.scroll(
        collection_name=collection,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    found = False

    for p in points:
        payload = p.payload or {}

        if payload.get("source") == "final_db_candidates":
            print("POINT ID:", p.id)
            print("PAYLOAD:")
            for k, v in sorted(payload.items()):
                print(f"  {k}: {v}")
            found = True
            break

    if not found:
        print("No final_db_candidates point found in first 100 points")
