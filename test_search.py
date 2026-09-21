from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

c = QdrantClient(host="localhost", port=6333)

pts = c.scroll("forensic_person",
    scroll_filter={"must": [{"key": "source", "match": {"value": "forensic_image"}}]},
    limit=1, with_payload=True, with_vectors=True)[0]

if not pts:
    raise SystemExit("No forensic_image points found.")

qpt = pts[0]
qvec = qpt.vector["siglip2"]
print("Query:", qpt.payload["detection_id"])
print()

r_img = c.query_points("forensic_person",
    query=qvec, using="siglip2", limit=3,
    query_filter=Filter(must=[FieldCondition(key="media_type", match=MatchValue(value="image"))]),
    with_payload=True).points
print("=== image->image (top 3) ===")
for r in r_img:
    print("  score=" + str(round(r.score,4)) + "  " + r.payload["detection_id"])

print()

r_vid = c.query_points("forensic_person",
    query=qvec, using="siglip2", limit=3,
    query_filter=Filter(must=[FieldCondition(key="media_type", match=MatchValue(value="video"))]),
    with_payload=True).points
print("=== image->video (top 3) ===")
for r in r_vid:
    print("  score=" + str(round(r.score,4)) + "  track=" + str(r.payload.get("track_id","?")) + "  source=" + str(r.payload.get("source","?")))
