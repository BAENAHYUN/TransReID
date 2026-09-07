from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"

def normalize(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v if n <= 0 else v / n

def make_filter(media):
    if media == "all":
        return None
    return Filter(must=[FieldCondition(key="media_type", match=MatchValue(value=media))])

def video_time(frame_idx, video_path):
    if not video_path:
        return ""
    p = Path(video_path)
    if not p.exists():
        return ""
    cap = cv2.VideoCapture(str(p))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        return ""
    sec = frame_idx / fps
    return f"{int(sec//60):02d}:{sec%60:05.2f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--collection", required=True, choices=["forensic_person","forensic_object"])
    ap.add_argument("--vector", required=True, choices=["siglip2","irra","solider","dinov2"])
    ap.add_argument("--media", choices=["all","image","video"], default="all")
    ap.add_argument("--top-k", type=int, default=20)
    a = ap.parse_args()

    image_path = Path(a.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    cfg = PipelineConfig.load(CONFIG_PATH)
    reg = EmbedderRegistry(cfg)
    router = Router(cfg, reg, input_format="rgb")

    if a.collection == "forensic_person":
        scope = "person"
        allowed = {"siglip2","irra","solider"}
    else:
        scope = "object"
        allowed = {"siglip2","dinov2"}

    if a.vector not in allowed:
        raise ValueError(f"{a.collection} supports {sorted(allowed)}")

    vectors = router.embed_query_image(str(image_path), scope=scope)
    qv = normalize(vectors[a.vector])

    client = QdrantClient(url=cfg.qdrant.url)
    r = client.query_points(
        collection_name=a.collection,
        using=a.vector,
        query=qv.tolist(),
        query_filter=make_filter(a.media),
        limit=a.top_k,
        with_payload=True,
        with_vectors=False,
    )

    print("\n" + "="*88)
    print(f"RESULTS - {a.vector.upper()}")
    print("="*88)

    for rank, hit in enumerate(r.points, 1):
        pld = hit.payload or {}
        media_type = pld.get("media_type","")
        frame_idx = int(pld.get("frame_idx",0) or 0)
        print(f"[{rank:02d}] score={float(hit.score):.6f} | {media_type} | {pld.get('label','')}")
        if media_type == "video":
            vp = pld.get("video_path","") or ""
            print(f"     video={pld.get('video','')} | frame={frame_idx} | time={video_time(frame_idx, vp) or 'N/A'} | track={pld.get('track_id')}")
            print(f"     video_path={vp}")
        else:
            print(f"     image_id={pld.get('image_id','')}")
        print(f"     crop={pld.get('crop_path','')}")

if __name__ == "__main__":
    main()
