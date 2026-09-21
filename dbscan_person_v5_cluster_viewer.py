#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import mimetypes
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from qdrant_client import QdrantClient
from config import PipelineConfig

DEFAULT_COLLECTION = "forensic_person"
DEFAULT_PAYLOAD_KEY = "dbscan_person_v5_cl"


def esc(v):
    return html.escape("" if v is None else str(v))


def q(v):
    return quote(str(v), safe="")


def parse_args():
    p = argparse.ArgumentParser(description="Qdrant DBSCAN person cluster HTML viewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--payload-key", default=DEFAULT_PAYLOAD_KEY)
    p.add_argument("--page-size", type=int, default=60)
    return p.parse_args()


class ClusterIndex:
    def __init__(self, client, collection, payload_key):
        self.client = client
        self.collection = collection
        self.payload_key = payload_key
        self.points = {}
        self.clusters = defaultdict(list)
        self.counts = Counter()
        self.reload()

    def reload(self):
        self.points.clear()
        self.clusters.clear()
        self.counts.clear()

        offset = None
        while True:
            pts, offset = self.client.scroll(
                collection_name=self.collection,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not pts:
                break

            for p in pts:
                payload = p.payload or {}
                if self.payload_key not in payload:
                    continue

                try:
                    cid = int(payload[self.payload_key])
                except Exception:
                    continue

                pid = str(p.id)
                self.points[pid] = payload
                self.clusters[cid].append(pid)
                self.counts[cid] += 1

            if offset is None:
                break

        for cid, ids in self.clusters.items():
            ids.sort(
                key=lambda pid: (
                    str(self.points[pid].get("video_stem", "")),
                    int(self.points[pid].get("frame_idx", 0) or 0),
                    int(self.points[pid].get("selected_rank", 0) or 0),
                )
            )

    def cluster_ids(self):
        ids = [cid for cid in self.counts if cid != -1]
        ids.sort(key=lambda cid: (-self.counts[cid], cid))
        if -1 in self.counts:
            ids.append(-1)
        return ids


def make_handler(index, default_page_size):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print("[cluster-viewer]", fmt % args)

        def send_html(self, text, status=200):
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/image":
                return self.serve_image(parsed)
            if parsed.path == "/reload":
                index.reload()
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
                return
            if parsed.path == "/":
                return self.serve_page(parsed)
            self.send_error(404)

        def serve_image(self, parsed):
            params = parse_qs(parsed.query)
            raw = params.get("path", [""])[0]
            path = Path(unquote(raw))

            try:
                path = path.resolve()
            except Exception:
                self.send_error(400, "Bad path")
                return

            if not path.is_file():
                self.send_error(404, "Image not found")
                return

            mime, _ = mimetypes.guess_type(str(path))
            if not mime:
                mime = "application/octet-stream"

            try:
                data = path.read_bytes()
            except Exception as exc:
                self.send_error(500, str(exc))
                return

            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def serve_page(self, parsed):
            params = parse_qs(parsed.query)

            cluster_raw = params.get("cluster", [""])[0]
            page_raw = params.get("page", ["1"])[0]
            size_raw = params.get("size", [str(default_page_size)])[0]

            try:
                selected_cluster = int(cluster_raw) if cluster_raw != "" else None
            except Exception:
                selected_cluster = None

            try:
                page = max(1, int(page_raw))
            except Exception:
                page = 1

            try:
                page_size = max(10, min(200, int(size_raw)))
            except Exception:
                page_size = default_page_size

            cluster_ids = index.cluster_ids()
            total_points = sum(index.counts.values())
            noise = index.counts.get(-1, 0)
            assigned = total_points - noise

            options = ['<option value="/">All clusters</option>']
            for cid in cluster_ids:
                title = "Noise" if cid == -1 else f"Cluster {cid}"
                selected = " selected" if cid == selected_cluster else ""
                options.append(
                    f'<option value="/?cluster={cid}"{selected}>{esc(title)} · {index.counts[cid]} points</option>'
                )

            if selected_cluster is None:
                parts = []
                for cid in cluster_ids:
                    ids = index.clusters[cid]
                    thumbs = []

                    for pid in ids[:8]:
                        payload = index.points[pid]
                        crop = payload.get("crop_path", "")
                        if crop:
                            src = f"/image?path={q(crop)}"
                            thumbs.append(f'<img loading="lazy" src="{src}" alt="">')

                    title = "Noise" if cid == -1 else f"Cluster {cid}"

                    parts.append(
                        '<a class="cluster-card" href="/?cluster={cid}">'
                        '<div class="cluster-head"><b>{title}</b><span>{count} points</span></div>'
                        '<div class="thumbs">{thumbs}</div>'
                        '</a>'.format(
                            cid=cid,
                            title=esc(title),
                            count=index.counts[cid],
                            thumbs="".join(thumbs),
                        )
                    )

                content = '<section class="cluster-grid">' + "".join(parts) + "</section>"
                page_info = f"{len(cluster_ids)} clusters"

            else:
                ids = index.clusters.get(selected_cluster, [])
                total = len(ids)
                start = (page - 1) * page_size
                end = min(start + page_size, total)

                cards = []

                for pid in ids[start:end]:
                    payload = index.points[pid]
                    crop = payload.get("crop_path", "")
                    src = f"/image?path={q(crop)}" if crop else ""

                    if src:
                        image_html = f'<a href="{src}" target="_blank"><img loading="lazy" src="{src}" alt=""></a>'
                    else:
                        image_html = '<div class="missing">no crop_path</div>'

                    keys = [
                        "video_stem",
                        "frame_idx",
                        "timestamp_sec",
                        "selected_track_id",
                        "canonical_person_id",
                        "selected_rank",
                        "label",
                        "confidence",
                        "quality_score",
                        "final_person_score",
                        "candidate_scope",
                        "crop_path",
                    ]

                    rows = []
                    for key in keys:
                        if key in payload:
                            rows.append(
                                f"<tr><th>{esc(key)}</th><td>{esc(payload.get(key))}</td></tr>"
                            )

                    cards.append(
                        '<article class="point-card">'
                        f'<div class="image-wrap">{image_html}</div>'
                        '<div class="body">'
                        f'<div class="title">{esc(payload.get("video_stem",""))}</div>'
                        '<div class="meta">'
                        f'frame <b>{esc(payload.get("frame_idx",""))}</b> · '
                        f'track <b>{esc(payload.get("selected_track_id", payload.get("track_id","")))}</b> · '
                        f'rank <b>{esc(payload.get("selected_rank",""))}</b>'
                        '</div>'
                        '<details><summary>Payload</summary>'
                        f'<table>{"".join(rows)}</table>'
                        f'<code>{esc(pid)}</code>'
                        '</details>'
                        '</div></article>'
                    )

                prev_link = ""
                next_link = ""

                if page > 1:
                    prev_link = (
                        f'<a class="btn" href="/?cluster={selected_cluster}&page={page-1}&size={page_size}">← Prev</a>'
                    )
                if end < total:
                    next_link = (
                        f'<a class="btn" href="/?cluster={selected_cluster}&page={page+1}&size={page_size}">Next →</a>'
                    )

                pager = f'<div class="pager">{prev_link}{next_link}</div>'
                content = pager + '<section class="point-grid">' + "".join(cards) + "</section>" + pager

                title = "Noise" if selected_cluster == -1 else f"Cluster {selected_cluster}"
                page_info = f"{title} · {total} points · showing {start + 1 if total else 0}-{end}"

            doc = f'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DBSCAN Person Cluster Viewer</title>
<style>
:root {{
  --bg:#09101c; --panel:#121a2a; --panel2:#0e1522; --line:#27334a;
  --text:#eef5ff; --muted:#9caac2; --blue:#77aaff;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:Segoe UI,Arial,sans-serif; }}
header {{ position:sticky; top:0; z-index:10; background:rgba(9,16,28,.96); border-bottom:1px solid var(--line); }}
.wrap {{ width:min(1700px,96vw); margin:auto; }}
header .wrap {{ padding:16px 0; }}
h1 {{ margin:0 0 10px; font-size:22px; }}
.toolbar {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
select,.btn {{ background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:9px; padding:9px 11px; text-decoration:none; }}
.info {{ color:var(--muted); margin-left:auto; }}
main.wrap {{ padding:20px 0 50px; }}
.stats {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }}
.stat {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:10px 14px; }}
.stat b {{ font-size:20px; display:block; }}
.cluster-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; }}
.cluster-card {{ display:block; text-decoration:none; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
.cluster-card:hover {{ border-color:var(--blue); }}
.cluster-head {{ display:flex; justify-content:space-between; gap:10px; padding:12px 14px; }}
.cluster-head span {{ color:var(--muted); }}
.thumbs {{ display:grid; grid-template-columns:repeat(4,1fr); height:190px; background:#070c15; }}
.thumbs img {{ width:100%; height:95px; object-fit:cover; border:1px solid #111928; }}
.point-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; }}
.point-card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
.image-wrap {{ height:260px; background:#070c15; display:flex; align-items:center; justify-content:center; }}
.image-wrap img {{ width:100%; height:100%; object-fit:contain; }}
.missing {{ color:var(--muted); }}
.body {{ padding:12px; }}
.title {{ font-weight:800; word-break:break-all; }}
.meta {{ margin-top:5px; color:var(--muted); font-size:13px; }}
details {{ margin-top:10px; border-top:1px solid var(--line); padding-top:8px; }}
summary {{ cursor:pointer; color:var(--blue); font-weight:700; }}
table {{ width:100%; border-collapse:collapse; margin-top:7px; font-size:12px; }}
th,td {{ padding:5px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; word-break:break-all; }}
th {{ color:var(--muted); width:38%; }}
code {{ display:block; margin-top:7px; background:var(--panel2); padding:7px; border-radius:7px; overflow:auto; }}
.pager {{ display:flex; justify-content:flex-end; gap:8px; margin:0 0 12px; }}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>DBSCAN Person Cluster Viewer</h1>
    <div class="toolbar">
      <select onchange="location.href=this.value">
        {''.join(options)}
      </select>
      <a class="btn" href="/">Summary</a>
      <a class="btn" href="/reload">Reload Qdrant</a>
      <span class="info">{esc(page_info)}</span>
    </div>
  </div>
</header>
<main class="wrap">
  <div class="stats">
    <div class="stat"><b>{len([x for x in index.counts if x != -1])}</b>clusters</div>
    <div class="stat"><b>{assigned}</b>assigned</div>
    <div class="stat"><b>{noise}</b>noise</div>
    <div class="stat"><b>{total_points}</b>total payload points</div>
  </div>
  {content}
</main>
</body>
</html>'''

            self.send_html(doc)

    return Handler


def main():
    args = parse_args()

    cfg = PipelineConfig.load(str(Path(__file__).resolve().parent / "pipeline.yaml"))
    client = QdrantClient(
        url=cfg.qdrant.url,
        timeout=120,
    )

    print("[LOAD] Qdrant cluster payloads...")
    index = ClusterIndex(client, args.collection, args.payload_key)

    print("=" * 80)
    print("DBSCAN PERSON CLUSTER HTML VIEWER")
    print("=" * 80)
    print("collection :", args.collection)
    print("payload key:", args.payload_key)
    print("clusters   :", len([x for x in index.counts if x != -1]))
    print("noise      :", index.counts.get(-1, 0))
    print("points     :", sum(index.counts.values()))
    print(f"viewer     : http://{args.host}:{args.port}")
    print("stop       : Ctrl+C")
    print("=" * 80)

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(index, args.page_size),
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
