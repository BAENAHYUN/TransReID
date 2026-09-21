import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path("outputs/final_db_candidates")

def stable_id(video_stem, row):
    scope = str(row["candidate_scope"])
    track_id = int(
        row.get(
            "selected_track_id",
            row.get("long_track_id", -1)
        )
    )
    frame_idx = int(row["frame_idx"])
    rank = int(row.get("selected_rank", 0))

    key = (
        f"final-ingest|{video_stem}|{scope}|"
        f"{track_id}|{frame_idx}|{rank}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            key
        )
    )


expected = set()

for path in ROOT.glob("*/final_db_candidates.json"):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    video_stem = path.parent.name

    for row in data.get("candidates", []):
        if str(row.get("candidate_scope", "")).lower() == "person":
            expected.add(stable_id(video_stem, row))


client = QdrantClient(
    "http://localhost:6333",
    timeout=120,
)

actual = {}
offset = None

while True:
    points, offset = client.scroll(
        collection_name="forensic_person",
        limit=64,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    for p in points:
        payload = p.payload or {}

        if payload.get("source") == "final_db_candidates":
            actual[str(p.id)] = payload

    if offset is None:
        break


actual_ids = set(actual)

extra = sorted(actual_ids - expected)
missing = sorted(expected - actual_ids)

print("=" * 80)
print("PERSON ID AUDIT")
print("=" * 80)
print("expected :", len(expected))
print("actual   :", len(actual_ids))
print("extra    :", len(extra))
print("missing  :", len(missing))

print()
print("EXTRA POINTS")
for point_id in extra:
    p = actual[point_id]

    print(
        point_id,
        "| video=", p.get("video_stem"),
        "| track=", p.get("selected_track_id"),
        "| frame=", p.get("frame_idx"),
        "| rank=", p.get("selected_rank"),
        "| crop=", p.get("crop_path"),
    )

print()
print("MISSING IDS")
for point_id in missing:
    print(point_id)
