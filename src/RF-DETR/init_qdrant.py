from qdrant_client import QdrantClient, models

COLLECTION_NAME = "forensic_persons"

client = QdrantClient("http://127.0.0.1:6333")

if client.collection_exists(COLLECTION_NAME):
    print(f"Existing collection found: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "reid": models.VectorParams(
            size=1280,
            distance=models.Distance.COSINE,
        ),
        "face": models.VectorParams(
            size=512,
            distance=models.Distance.COSINE,
        ),
    },
)

print(f"Created collection: {COLLECTION_NAME}")

info = client.get_collection(COLLECTION_NAME)
print(info)