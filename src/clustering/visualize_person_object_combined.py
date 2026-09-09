from __future__ import annotations

import argparse
import hashlib
import html
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from qdrant_client import QdrantClient


COLLECTIONS = {
    "person": "forensic_person",
    "object": "forensic_object",
}

SCROLL_BATCH = 1024
DEFAULT_SAMPLES_PER_CLUSTER = 12
DEFAULT_BEFORE_SAMPLES = 300
THUMB_SIZE = (180, 240)
THUMB_QUALITY = 78

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "src" / "clustering" / "output" / "combined_visualization"


def stable_name(scope: str, point_id: str) -> str:
    h = hashlib.sha1(f"{scope}:{point_id}".encode("utf-8")).hexdigest()[:20]
    return f"{scope}_{h}.jpg"


def resolve_crop_path(payload: dict) -> Path | None:
    candidates = [
        payload.get("crop_path"),
        payload.get("image_path"),
        payload.get("file"),
        payload.get("path"),
    ]

    for raw in candidates:
        if not raw:
            continue

        p = Path(str(raw))
        if p.is_file():
            return p

        p2 = ROOT / p
        if p2.is_file():
            return p2

    return None


def make_thumb(src: Path | None, dst: Path, label: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src is None:
        im = Image.new("RGB", THUMB_SIZE, "white")
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), f"CROP NOT FOUND\n{label}", fill="black")
        im.save(dst, "JPEG", quality=70)
        return

    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            fitted = ImageOps.contain(im, THUMB_SIZE)
            canvas = Image.new("RGB", THUMB_SIZE, "white")
            x = (THUMB_SIZE[0] - fitted.width) // 2
            y = (THUMB_SIZE[1] - fitted.height) // 2
            canvas.paste(fitted, (x, y))
            canvas.save(dst, "JPEG", quality=THUMB_QUALITY, optimize=True)
    except Exception:
        im = Image.new("RGB", THUMB_SIZE, "white")
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), f"THUMB ERROR\n{label}", fill="black")
        im.save(dst, "JPEG", quality=70)


def reservoir_add(bucket: list, seen: int, item: dict, limit: int, rng: random.Random) -> None:
    if len(bucket) < limit:
        bucket.append(item)
        return

    j = rng.randint(1, seen)
    if j <= limit:
        bucket[j - 1] = item


def scan_collection(
    client: QdrantClient,
    scope: str,
    collection: str,
    samples_per_cluster: int,
    before_samples: int,
):
    rng = random.Random(42 if scope == "person" else 84)

    cluster_samples = defaultdict(list)
    cluster_seen = defaultdict(int)
    cluster_counts = defaultdict(int)
    cluster_image = defaultdict(int)
    cluster_video = defaultdict(int)

    before_bucket = []
    before_seen = 0

    total = 0
    image_count = 0
    video_count = 0
    missing_media = 0
    missing_cluster = 0

    offset = None

    print(f"\n[{scope.upper()}] scanning {collection} ...")

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
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
            if media == "image":
                image_count += 1
            elif media == "video":
                video_count += 1
            else:
                missing_media += 1

            item = {
                "id": str(p.id),
                "scope": scope,
                "payload": payload,
            }

            before_seen += 1
            reservoir_add(before_bucket, before_seen, item, before_samples, rng)

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

            cluster_seen[cid] += 1
            reservoir_add(
                cluster_samples[cid],
                cluster_seen[cid],
                item,
                samples_per_cluster,
                rng,
            )

        if total % 50000 < SCROLL_BATCH:
            print(
                f"  scanned={total:,} "
                f"clusters={len(cluster_counts)} "
                f"missing_cluster={missing_cluster:,}"
            )

        if next_offset is None:
            break
        offset = next_offset

    return {
        "scope": scope,
        "collection": collection,
        "total": total,
        "image": image_count,
        "video": video_count,
        "missing_media": missing_media,
        "missing_cluster": missing_cluster,
        "cluster_counts": cluster_counts,
        "cluster_image": cluster_image,
        "cluster_video": cluster_video,
        "cluster_samples": cluster_samples,
        "before_samples": before_bucket,
    }


def prepare_assets(scope: str, stats: dict, out_dir: Path) -> None:
    asset_dir = out_dir / "assets" / scope
    asset_dir.mkdir(parents=True, exist_ok=True)

    items = list(stats["before_samples"])
    for cid in stats["cluster_samples"]:
        items.extend(stats["cluster_samples"][cid])

    unique = {}
    for item in items:
        unique[item["id"]] = item

    for item in unique.values():
        src = resolve_crop_path(item["payload"])
        filename = stable_name(scope, item["id"])
        dst = asset_dir / filename
        if not dst.exists():
            make_thumb(src, dst, item["id"])
        item["asset"] = f"assets/{scope}/{filename}"

    # Push asset path back to every sampled item instance.
    for item in stats["before_samples"]:
        item["asset"] = f"assets/{scope}/{stable_name(scope, item['id'])}"
    for cid in stats["cluster_samples"]:
        for item in stats["cluster_samples"][cid]:
            item["asset"] = f"assets/{scope}/{stable_name(scope, item['id'])}"


def esc(v) -> str:
    return html.escape("" if v is None else str(v))


def meta_text(item: dict) -> str:
    p = item["payload"]
    media = p.get("media_type", "")
    label = p.get("label") or p.get("class") or ""

    if media == "video":
        video = p.get("video") or p.get("video_name") or ""
        track = (
            p.get("person_id")
            or p.get("object_id")
            or p.get("track_id")
            or p.get("stitched_id")
            or ""
        )
        frame = p.get("frame_idx", "")
        timev = p.get("timestamp_sec", "")
        return f"{label} | {video} | id={track} | frame={frame} | t={timev}"

    src = p.get("source") or p.get("filename") or p.get("image_id") or ""
    return f"{label} | {src}"


def card(item: dict, show_cluster: bool = False) -> str:
    p = item["payload"]
    media = p.get("media_type", "")
    cid = p.get("cluster_id", "")

    cluster_badge = (
        f'<span class="badge">cluster {esc(cid)}</span>'
        if show_cluster
        else ""
    )

    return f"""
    <div class="card" data-media="{esc(media)}">
      <img loading="lazy" src="{esc(item['asset'])}">
      <div class="badges">
        <span class="badge">{esc(media)}</span>
        {cluster_badge}
      </div>
      <div class="meta">{esc(meta_text(item))}</div>
      <div class="pid">point={esc(item['id'])}</div>
    </div>
    """


def html_doc(person: dict, obj: dict) -> str:
    def stats_bar(s: dict) -> str:
        return f"""
        <div class="stats">
          <div class="stat"><b>Total</b><br>{s['total']:,}</div>
          <div class="stat"><b>Image</b><br>{s['image']:,}</div>
          <div class="stat"><b>Video</b><br>{s['video']:,}</div>
          <div class="stat"><b>Clusters</b><br>{len(s['cluster_counts']):,}</div>
        </div>
        """

    def before_section(s: dict) -> str:
        cards = "".join(card(x, False) for x in s["before_samples"])
        return f"""
        <section class="subview" data-view="before">
          <h2>기존 DB 샘플</h2>
          <p class="note">
            cluster_id를 무시하고 원래 DB point를 섞어서 보여주는 비교용 샘플입니다.
          </p>
          <div class="grid">{cards}</div>
        </section>
        """

    def clusters_section(s: dict) -> str:
        parts = []
        counts = s["cluster_counts"]
        images = s["cluster_image"]
        videos = s["cluster_video"]
        samples = s["cluster_samples"]

        for cid in sorted(counts, key=lambda c: (-counts[c], c)):
            cards = "".join(card(x, True) for x in samples[cid])
            parts.append(
                f"""
                <section class="cluster" data-cluster="{cid}">
                  <div class="cluster-head">
                    <h2>Cluster {cid}</h2>
                    <div>
                      total={counts[cid]:,} |
                      image={images[cid]:,} |
                      video={videos[cid]:,}
                    </div>
                  </div>
                  <div class="grid">{cards}</div>
                </section>
                """
            )
        return f"""
        <section class="subview" data-view="clusters">
          <div class="cluster-tools">
            <input class="cluster-search" placeholder="cluster ID 입력"
                   oninput="filterCluster(this)">
          </div>
          {''.join(parts)}
        </section>
        """

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Person + Object Cluster Visualization</title>
<style>
* {{ box-sizing:border-box; }}
body {{
  margin:0;
  font-family:Arial,"Malgun Gothic",sans-serif;
  background:#f4f6f8;
  color:#111;
}}
header {{
  position:sticky;
  top:0;
  z-index:20;
  background:white;
  border-bottom:1px solid #ddd;
  padding:14px 20px;
}}
h1 {{ margin:0 0 8px; font-size:24px; }}
.tabs, .view-tabs {{
  display:flex; gap:8px; flex-wrap:wrap; margin-top:8px;
}}
button, input {{
  border:1px solid #bbb;
  background:white;
  border-radius:7px;
  padding:8px 12px;
  cursor:pointer;
}}
button.active {{
  background:#222;
  color:white;
}}
.panel {{ display:none; padding:18px; }}
.panel.active {{ display:block; }}
.subview {{ display:none; }}
.subview.active {{ display:block; }}
.stats {{ display:flex; gap:10px; flex-wrap:wrap; margin:12px 0; }}
.stat {{
  background:white; border:1px solid #ddd; border-radius:9px;
  padding:9px 13px; min-width:120px;
}}
.note {{ color:#666; font-size:13px; }}
.grid {{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(185px,1fr));
  gap:10px;
}}
.card {{
  background:white; border:1px solid #ddd; border-radius:9px;
  padding:7px; overflow:hidden;
}}
.card img {{
  width:100%; height:240px; object-fit:contain;
  background:#fafafa; border-radius:6px;
}}
.badges {{ display:flex; gap:5px; flex-wrap:wrap; margin-top:5px; }}
.badge {{
  border:1px solid #bbb; border-radius:999px;
  padding:2px 7px; font-size:11px;
}}
.meta {{
  margin-top:5px; font-size:11px; line-height:1.4;
  word-break:break-all;
}}
.pid {{ margin-top:4px; color:#777; font-size:9px; word-break:break-all; }}
.cluster {{
  background:white; border:1px solid #ddd; border-radius:10px;
  padding:12px; margin:14px 0;
}}
.cluster-head {{
  display:flex; justify-content:space-between; gap:10px;
  align-items:end; flex-wrap:wrap; margin-bottom:9px;
}}
.cluster-head h2 {{ margin:0; }}
.cluster-tools {{ margin-bottom:12px; }}
.hidden {{ display:none !important; }}
</style>
<script>
function switchScope(scope) {{
  document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
  document.querySelector('#panel-' + scope).classList.add('active');
  document.querySelectorAll('.scope-btn').forEach(x => x.classList.remove('active'));
  document.querySelector('#btn-' + scope).classList.add('active');
}}
function switchView(scope, view) {{
  const panel = document.querySelector('#panel-' + scope);
  panel.querySelectorAll('.subview').forEach(x => x.classList.remove('active'));
  panel.querySelector('[data-view="' + view + '"]').classList.add('active');
  panel.querySelectorAll('.view-btn').forEach(x => x.classList.remove('active'));
  panel.querySelector('[data-btn="' + view + '"]').classList.add('active');
}}
function filterCluster(input) {{
  const panel = input.closest('.panel');
  const term = input.value.trim();
  panel.querySelectorAll('.cluster').forEach(el => {{
    el.classList.toggle('hidden', term && el.dataset.cluster !== term);
  }});
}}
</script>
</head>
<body>
<header>
  <h1>Person + Object DB / Cluster Visualization</h1>
  <div class="tabs">
    <button id="btn-person" class="scope-btn active"
            onclick="switchScope('person')">Person</button>
    <button id="btn-object" class="scope-btn"
            onclick="switchScope('object')">Object</button>
  </div>
</header>

<div id="panel-person" class="panel active">
  <h2>forensic_person</h2>
  {stats_bar(person)}
  <div class="view-tabs">
    <button class="view-btn active" data-btn="before"
            onclick="switchView('person','before')">기존 DB</button>
    <button class="view-btn" data-btn="clusters"
            onclick="switchView('person','clusters')">클러스터 결과</button>
  </div>
  {before_section(person)}
  {clusters_section(person)}
</div>

<div id="panel-object" class="panel">
  <h2>forensic_object</h2>
  {stats_bar(obj)}
  <div class="view-tabs">
    <button class="view-btn active" data-btn="before"
            onclick="switchView('object','before')">기존 DB</button>
    <button class="view-btn" data-btn="clusters"
            onclick="switchView('object','clusters')">클러스터 결과</button>
  </div>
  {before_section(obj)}
  {clusters_section(obj)}
</div>

<script>
switchView('person','before');
switchView('object','before');
switchScope('person');
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="Person + Object DB/cluster visualization in one HTML"
    )
    ap.add_argument(
        "--samples-per-cluster",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLUSTER,
    )
    ap.add_argument(
        "--before-samples",
        type=int,
        default=DEFAULT_BEFORE_SAMPLES,
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = ap.parse_args()

    if args.samples_per_cluster <= 0:
        raise ValueError("--samples-per-cluster must be >= 1")
    if args.before_samples <= 0:
        raise ValueError("--before-samples must be >= 1")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    client = QdrantClient("localhost", port=6333, timeout=120)

    person = scan_collection(
        client,
        "person",
        COLLECTIONS["person"],
        args.samples_per_cluster,
        args.before_samples,
    )
    obj = scan_collection(
        client,
        "object",
        COLLECTIONS["object"],
        args.samples_per_cluster,
        args.before_samples,
    )

    for stats in (person, obj):
        print(
            f"\n[{stats['scope'].upper()}] "
            f"total={stats['total']:,} "
            f"image={stats['image']:,} "
            f"video={stats['video']:,} "
            f"clusters={len(stats['cluster_counts'])} "
            f"missing_cluster={stats['missing_cluster']:,} "
            f"missing_media={stats['missing_media']:,}"
        )

        if stats["missing_media"] != 0:
            raise RuntimeError(
                f"{stats['collection']}: media_type missing point exists."
            )

        if stats["missing_cluster"] != 0:
            raise RuntimeError(
                f"{stats['collection']}: cluster_id missing point exists. "
                "클러스터링 payload 적용을 먼저 완료하세요."
            )

        prepare_assets(
            stats["scope"],
            stats,
            out_dir,
        )

    out_html = out_dir / "person_object_db_clusters.html"
    out_html.write_text(
        html_doc(person, obj),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("COMBINED VISUALIZATION COMPLETE")
    print("=" * 78)
    print("HTML :", out_html)
    print("Person:", f"{person['total']:,}", "points")
    print("Object:", f"{obj['total']:,}", "points")
    print("Total :", f"{person['total'] + obj['total']:,}", "points")
    print("=" * 78)


if __name__ == "__main__":
    main()
