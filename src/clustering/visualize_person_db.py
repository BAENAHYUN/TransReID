from __future__ import annotations

import argparse
import csv
import hashlib
import html
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
from qdrant_client import QdrantClient

COLLECTION = "forensic_person"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120

DEFAULT_SAMPLES_PER_CLUSTER = 16
THUMB_SIZE = (220, 300)
THUMB_QUALITY = 78
SCROLL_BATCH = 1024


def project_root() -> Path:
    # intended location:
    # TransReID/src/clustering/visualize_person_db.py
    return Path(__file__).resolve().parents[2]


def resolve_crop_path(root: Path, payload: dict) -> Path | None:
    candidates = [
        payload.get("crop_path"),
        payload.get("image_path"),
        payload.get("path"),
    ]

    for raw in candidates:
        if not raw:
            continue

        p = Path(str(raw))
        if p.is_file():
            return p

        p2 = root / p
        if p2.is_file():
            return p2

    return None


def stable_name(point_id: str) -> str:
    h = hashlib.sha1(point_id.encode("utf-8")).hexdigest()[:20]
    return h + ".jpg"


def make_thumb(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            fitted = ImageOps.contain(im, THUMB_SIZE)

            canvas = Image.new("RGB", THUMB_SIZE, "white")
            x = (THUMB_SIZE[0] - fitted.width) // 2
            y = (THUMB_SIZE[1] - fitted.height) // 2
            canvas.paste(fitted, (x, y))

            canvas.save(
                dst,
                format="JPEG",
                quality=THUMB_QUALITY,
                optimize=True,
            )
    except Exception as exc:
        canvas = Image.new("RGB", THUMB_SIZE, "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"THUMB ERROR\n{src.name}\n{type(exc).__name__}", fill="black")
        canvas.save(dst, format="JPEG", quality=70)


def escape(v) -> str:
    return html.escape("" if v is None else str(v))


def mmss(payload: dict) -> str:
    val = (
        payload.get("time_mmss")
        or payload.get("timestamp")
        or payload.get("timestamp_sec")
        or payload.get("time_sec")
    )
    if val is None:
        return ""
    return str(val)


def item_meta_text(item: dict) -> str:
    p = item["payload"]
    media = p.get("media_type", "")
    if media == "video":
        parts = [
            f"video={p.get('video') or p.get('video_name') or ''}",
            f"track={p.get('track_id') or p.get('person_id') or ''}",
            f"frame={p.get('frame_idx') or p.get('frame_number') or ''}",
            f"time={mmss(p)}",
        ]
    else:
        parts = [
            f"image={p.get('image_id') or ''}",
            f"source={p.get('source') or ''}",
        ]
    return " | ".join(x for x in parts if not x.endswith("="))


def item_card(item: dict, rel_asset: str, show_cluster: bool) -> str:
    p = item["payload"]
    cid = p.get("cluster_id")
    media = p.get("media_type", "")

    top = (
        f'<div class="badge">{escape(media)}</div>'
        + (f'<div class="badge">cluster {escape(cid)}</div>' if show_cluster else "")
    )

    return f"""
    <div class="item" data-media="{escape(media)}" data-cluster="{escape(cid)}">
      <div class="thumb-wrap">
        <img loading="lazy" src="{escape(rel_asset)}" alt="">
      </div>
      <div class="badges">{top}</div>
      <div class="meta">{escape(item_meta_text(item))}</div>
      <div class="pid">point={escape(item["id"])}</div>
    </div>
    """


def html_head(title: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
  :root {{
    --bg: #f5f5f5;
    --card: #ffffff;
    --line: #dddddd;
    --text: #111111;
    --muted: #666666;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: Arial, "Malgun Gothic", sans-serif;
    background: var(--bg);
    color: var(--text);
  }}
  .top {{
    position: sticky;
    top: 0;
    z-index: 20;
    background: rgba(255,255,255,.96);
    border-bottom: 1px solid var(--line);
    padding: 16px 20px;
  }}
  h1 {{ margin: 0 0 8px; font-size: 24px; }}
  h2 {{ margin: 28px 0 10px; }}
  .summary {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
  }}
  .stat {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px 12px;
    min-width: 120px;
  }}
  .controls {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
  }}
  button, input {{
    padding: 8px 10px;
    border: 1px solid #bbb;
    border-radius: 7px;
    background: white;
  }}
  .content {{ padding: 18px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
  }}
  .item {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 8px;
    overflow: hidden;
  }}
  .thumb-wrap {{
    width: 100%;
    display: flex;
    justify-content: center;
    background: #fafafa;
    border-radius: 6px;
    overflow: hidden;
  }}
  .thumb-wrap img {{
    width: 100%;
    max-width: 220px;
    height: 300px;
    object-fit: contain;
    display: block;
  }}
  .badges {{ margin-top: 7px; display:flex; gap:6px; flex-wrap:wrap; }}
  .badge {{
    display:inline-block;
    border:1px solid #bbb;
    border-radius:999px;
    padding:2px 7px;
    font-size:12px;
  }}
  .meta {{
    margin-top: 6px;
    font-size: 12px;
    line-height: 1.4;
    word-break: break-all;
  }}
  .pid {{
    margin-top: 5px;
    color: var(--muted);
    font-size: 10px;
    word-break: break-all;
  }}
  .cluster {{
    background: white;
    border:1px solid var(--line);
    border-radius: 10px;
    margin: 16px 0;
    padding: 14px;
  }}
  .cluster-head {{
    display:flex;
    justify-content:space-between;
    gap:12px;
    flex-wrap:wrap;
    align-items:end;
    margin-bottom: 10px;
  }}
  .bar-bg {{
    height: 10px;
    background: #eeeeee;
    border-radius: 999px;
    overflow:hidden;
    min-width: 180px;
  }}
  .bar {{
    height:100%;
    background:#444444;
  }}
  .hidden {{ display:none !important; }}
  .note {{
    margin:8px 0 0;
    color:var(--muted);
    font-size:13px;
  }}
</style>
<script>
function setMedia(media) {{
  document.querySelectorAll('.item').forEach(el => {{
    const ok = media === 'all' || el.dataset.media === media;
    el.classList.toggle('hidden', !ok);
  }});
}}
function filterCluster() {{
  const q = document.getElementById('clusterSearch');
  if (!q) return;
  const term = q.value.trim();
  document.querySelectorAll('.cluster').forEach(el => {{
    const cid = el.dataset.cluster || '';
    el.classList.toggle('hidden', term && cid !== term);
  }});
}}
</script>
</head>
<body>
"""


def reservoir_add(reservoirs, seen, cid, item, limit, rng):
    seen[cid] += 1
    bucket = reservoirs[cid]

    if len(bucket) < limit:
        bucket.append(item)
        return

    j = rng.randint(1, seen[cid])
    if j <= limit:
        bucket[j - 1] = item


def scan_qdrant(client: QdrantClient, sample_per_cluster: int):
    reservoirs = defaultdict(list)
    seen = defaultdict(int)

    cluster_counts = defaultdict(int)
    cluster_image = defaultdict(int)
    cluster_video = defaultdict(int)

    total = 0
    missing_cluster = 0
    missing_media = 0
    rng = random.Random(42)

    offset = None

    print("----- Qdrant payload scan -----")

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for p in points:
            payload = p.payload or {}
            total += 1

            media = payload.get("media_type")
            if media not in ("image", "video"):
                missing_media += 1

            cid = payload.get("cluster_id")
            if cid is None:
                missing_cluster += 1
                continue

            cid = int(cid)
            cluster_counts[cid] += 1

            if media == "image":
                cluster_image[cid] += 1
            elif media == "video":
                cluster_video[cid] += 1

            item = {
                "id": str(p.id),
                "payload": payload,
            }

            reservoir_add(
                reservoirs,
                seen,
                cid,
                item,
                sample_per_cluster,
                rng,
            )

        if total % 20_000 < SCROLL_BATCH:
            print(
                f"  scanned={total:,} "
                f"clusters={len(cluster_counts):,} "
                f"missing_cluster={missing_cluster:,}"
            )

        if offset is None:
            break

    return {
        "total": total,
        "missing_cluster": missing_cluster,
        "missing_media": missing_media,
        "reservoirs": reservoirs,
        "cluster_counts": cluster_counts,
        "cluster_image": cluster_image,
        "cluster_video": cluster_video,
    }


def prepare_assets(root: Path, out_dir: Path, reservoirs: dict):
    asset_dir = out_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    all_items = []
    missing_paths = 0

    for cid in sorted(reservoirs):
        for item in reservoirs[cid]:
            payload = item["payload"]
            src = resolve_crop_path(root, payload)

            filename = stable_name(item["id"])
            dst = asset_dir / filename

            if src is None:
                missing_paths += 1
                placeholder = Image.new("RGB", THUMB_SIZE, "white")
                draw = ImageDraw.Draw(placeholder)
                draw.text((10, 10), "CROP PATH NOT FOUND", fill="black")
                placeholder.save(dst, "JPEG", quality=70)
            elif not dst.exists():
                make_thumb(src, dst)

            item["asset"] = f"assets/{filename}"
            all_items.append(item)

    return all_items, missing_paths


def write_summary_csv(out_dir: Path, stats: dict):
    path = out_dir / "cluster_summary.csv"

    counts = stats["cluster_counts"]
    images = stats["cluster_image"]
    videos = stats["cluster_video"]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "total", "image", "video"])
        for cid in sorted(counts):
            w.writerow([cid, counts[cid], images[cid], videos[cid]])

    return path


def write_before_html(out_dir: Path, stats: dict, all_items: list):
    path = out_dir / "person_db_before.html"

    items = list(all_items)
    rng = random.Random(20260907)
    rng.shuffle(items)

    total_image = sum(stats["cluster_image"].values())
    total_video = sum(stats["cluster_video"].values())

    parts = [
        html_head("Person DB - Before Clustering View"),
        """
<div class="top">
  <h1>Person DB — Before Clustering View</h1>
  <div class="note">
    동일한 샘플들을 cluster_id를 무시하고 섞어서 표시합니다.
    After HTML과 비교하면 클러스터링 전/후 배열 변화를 눈으로 확인할 수 있습니다.
  </div>
""",
        '<div class="summary">',
        f'<div class="stat"><b>Total DB</b><br>{stats["total"]:,}</div>',
        f'<div class="stat"><b>Image</b><br>{total_image:,}</div>',
        f'<div class="stat"><b>Video</b><br>{total_video:,}</div>',
        f'<div class="stat"><b>Displayed sample</b><br>{len(items):,}</div>',
        '</div>',
        """
  <div class="controls">
    <button onclick="setMedia('all')">All</button>
    <button onclick="setMedia('image')">Image</button>
    <button onclick="setMedia('video')">Video</button>
  </div>
</div>
<div class="content">
  <div class="grid">
""",
    ]

    for item in items:
        parts.append(item_card(item, item["asset"], show_cluster=False))

    parts.append("</div></div></body></html>")
    path.write_text("".join(parts), encoding="utf-8")
    return path


def write_after_html(out_dir: Path, stats: dict):
    path = out_dir / "person_clusters_after.html"

    counts = stats["cluster_counts"]
    images = stats["cluster_image"]
    videos = stats["cluster_video"]
    reservoirs = stats["reservoirs"]

    max_count = max(counts.values()) if counts else 1
    total_image = sum(images.values())
    total_video = sum(videos.values())

    parts = [
        html_head("Person DB - After Clustering"),
        """
<div class="top">
  <h1>Person DB — After MiniBatch K-Means</h1>
  <div class="note">
    같은 DB point를 cluster_id 기준으로 묶어 표시합니다.
    각 클러스터는 전체 point 수와 Image/Video 구성, 대표 샘플을 함께 보여줍니다.
  </div>
""",
        '<div class="summary">',
        f'<div class="stat"><b>Total DB</b><br>{stats["total"]:,}</div>',
        f'<div class="stat"><b>Image</b><br>{total_image:,}</div>',
        f'<div class="stat"><b>Video</b><br>{total_video:,}</div>',
        f'<div class="stat"><b>Clusters</b><br>{len(counts):,}</div>',
        '</div>',
        """
  <div class="controls">
    <button onclick="setMedia('all')">All samples</button>
    <button onclick="setMedia('image')">Image samples</button>
    <button onclick="setMedia('video')">Video samples</button>
    <input id="clusterSearch" oninput="filterCluster()" placeholder="cluster ID">
  </div>
</div>
<div class="content">
""",
    ]

    for cid in sorted(counts, key=lambda c: (-counts[c], c)):
        pct = 100.0 * counts[cid] / max_count

        parts.append(
            f"""
<section class="cluster" data-cluster="{cid}">
  <div class="cluster-head">
    <div>
      <h2 style="margin:0">Cluster {cid}</h2>
      <div class="note">
        total={counts[cid]:,} |
        image={images[cid]:,} |
        video={videos[cid]:,} |
        displayed={len(reservoirs[cid]):,}
      </div>
    </div>
    <div>
      <div class="bar-bg">
        <div class="bar" style="width:{pct:.2f}%"></div>
      </div>
    </div>
  </div>
  <div class="grid">
"""
        )

        for item in reservoirs[cid]:
            parts.append(item_card(item, item["asset"], show_cluster=True))

        parts.append("</div></section>")

    parts.append("</div></body></html>")
    path.write_text("".join(parts), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Visualize forensic_person DB before/after clustering "
            "using the same sampled points."
        )
    )
    ap.add_argument(
        "--samples-per-cluster",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLUSTER,
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    if args.samples_per_cluster <= 0:
        raise ValueError("--samples-per-cluster must be >= 1")

    root = project_root()

    out_dir = (
        args.output.resolve()
        if args.output is not None
        else root / "src" / "clustering" / "output" / "person_visualization"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        timeout=QDRANT_TIMEOUT,
    )

    print("=" * 78)
    print("PERSON DB / CLUSTER VISUALIZATION")
    print("=" * 78)
    print("collection          :", COLLECTION)
    print("samples per cluster :", args.samples_per_cluster)
    print("output              :", out_dir)

    stats = scan_qdrant(client, args.samples_per_cluster)

    print("\n----- Scan result -----")
    print("total          :", f"{stats['total']:,}")
    print("clusters       :", f"{len(stats['cluster_counts']):,}")
    print("missing cluster:", f"{stats['missing_cluster']:,}")
    print("missing media  :", f"{stats['missing_media']:,}")

    if stats["missing_media"] != 0:
        raise RuntimeError(
            "media_type이 없는 point가 있습니다. DB 무결성을 먼저 확인하세요."
        )

    if stats["missing_cluster"] != 0:
        raise RuntimeError(
            "cluster_id가 없는 point가 있습니다. "
            "MiniBatch clustering payload 적용 완료 후 다시 실행하세요."
        )

    all_items, missing_paths = prepare_assets(
        root,
        out_dir,
        stats["reservoirs"],
    )

    print("\n----- Thumbnail result -----")
    print("displayed samples:", f"{len(all_items):,}")
    print("missing crop path:", f"{missing_paths:,}")

    before_path = write_before_html(out_dir, stats, all_items)
    after_path = write_after_html(out_dir, stats)
    csv_path = write_summary_csv(out_dir, stats)

    print("\n" + "=" * 78)
    print("VISUALIZATION COMPLETE")
    print("Before HTML :", before_path)
    print("After HTML  :", after_path)
    print("Summary CSV :", csv_path)
    print("=" * 78)


if __name__ == "__main__":
    main()
