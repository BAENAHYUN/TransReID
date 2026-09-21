#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
video_search_html.py — 비디오 DB + 클러스터 기반 인물 검색 → HTML 결과 뷰어

사용법
------
  # 이미지로 검색
  python video_search_html.py --image query.jpg --scope person -k 20 --open

  # 텍스트로 검색
  python video_search_html.py --text "빨간 상의를 입은 남성" --scope person -k 20 --open

  # 오브젝트 검색
  python video_search_html.py --image bag.jpg --scope object -k 15 --open

결과
----
  search_results_html/video_search_{timestamp}.html
  → 클러스터별로 그룹화된 다크 테마 갤러리
"""

from __future__ import annotations

import argparse
import base64
import html as htmllib
import json
import sys
import time
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unified_search_4mode as us

OUT_DIR = ROOT / "search_results_html"
MAX_EMBED = 4 * 1024 * 1024   # 4 MB 이하만 base64


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def img_uri(path_str: Optional[str]) -> str:
    if not path_str:
        return _placeholder()
    p = Path(path_str)
    if not p.is_file():
        return _placeholder()
    try:
        if p.stat().st_size > MAX_EMBED:
            return _placeholder()
        data = p.read_bytes()
        ext = p.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
        return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return _placeholder()


def _placeholder() -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="160">'
        '<rect width="100%" height="100%" fill="#1e1e1e"/>'
        '<text x="50%" y="50%" font-size="9" fill="#555" '
        'text-anchor="middle" dominant-baseline="middle">no img</text>'
        '</svg>'
    )
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"


def score_color(s: float) -> str:
    if s >= 0.75:  return "#22c55e"
    if s >= 0.60:  return "#f59e0b"
    if s >= 0.45:  return "#f97316"
    return "#ef4444"


def fmt_time(sec: Optional[float]) -> str:
    if sec is None:
        return "--:--"
    try:
        s = float(sec)
        return f"{int(s // 60):02d}:{s % 60:05.2f}"
    except Exception:
        return "--:--"


def short_id(cid: Optional[str], n: int = 8) -> str:
    if not cid:
        return "no-cluster"
    return str(cid)[-n:]


# ─────────────────────────────────────────────────────────────────────────────
# 검색 실행
# ─────────────────────────────────────────────────────────────────────────────
def run_search(args: argparse.Namespace) -> Dict[str, Any]:
    """unified_search_4mode 의 image-video 또는 text-video 모드 실행."""
    ns = argparse.Namespace(
        scope=args.scope,
        top_k=args.top_k,
        group_size=args.group_size,
        candidate_k=args.candidate_k,
        vector=args.vector,
        config=str(ROOT / "pipeline.yaml"),
        no_translate=args.no_translate,
        translate_backend=getattr(args, "translate_backend", "none"),
        translate_model_id=None,
        expand=False,
    )

    if args.image:
        ns.image = args.image
        print(f"[검색] image-video  scope={args.scope}  image={args.image}  k={args.top_k}")
        return us.run_image_video(ns)
    else:
        ns.text = args.text
        print(f"[검색] text-video  scope={args.scope}  text={args.text!r}  k={args.top_k}")
        return us.run_text_video(ns)


# ─────────────────────────────────────────────────────────────────────────────
# 클러스터 그룹핑
# ─────────────────────────────────────────────────────────────────────────────
def group_by_cluster(rows: List[Dict]) -> Dict[str, List[Dict]]:
    clusters: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        cid = str(row.get("cluster_id") or "unclustered")
        clusters[cid].append(row)
    # 각 클러스터를 best score 순으로 정렬
    for cid in clusters:
        clusters[cid].sort(key=lambda r: r.get("score", 0), reverse=True)
    # 클러스터 자체를 best score 기준으로 정렬
    return dict(sorted(clusters.items(), key=lambda kv: kv[1][0].get("score", 0), reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# HTML 생성
# ─────────────────────────────────────────────────────────────────────────────
def build_html(
    payload: Dict[str, Any],
    args: argparse.Namespace,
    clusters: Dict[str, List[Dict]],
) -> str:
    query_label = args.image or args.text or "unknown"
    query_uri   = img_uri(args.image) if args.image else None
    scope       = payload.get("scope", args.scope)
    vectors     = payload.get("vectors", [])
    timing      = payload.get("timing", {})
    ts          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_tracks = sum(len(v) for v in clusters.values())

    # ── cluster cards HTML ──────────────────────────────────────────
    cluster_cards = []
    for ci, (cid, rows) in enumerate(clusters.items()):
        best = rows[0]
        best_score = best.get("score", 0.0)
        sc = score_color(best_score)
        short = short_id(cid)

        # 각 트랙 카드
        track_cards = []
        for ri, row in enumerate(rows):
            score  = row.get("score", 0.0)
            video  = row.get("video") or "—"
            t_sec  = row.get("timestamp_sec")
            t_str  = fmt_time(t_sec)
            crop   = row.get("crop_path", "")
            tkey   = row.get("track_key", "")
            gs     = row.get("group_summary", {})
            n_pts  = gs.get("count", "?")
            orig   = gs.get("original_tracks", [])
            n_orig = len(orig)
            crop_uri = img_uri(crop)
            sc2 = score_color(score)

            # 원본 track id 들 (줄임)
            orig_str = ", ".join(str(x) for x in orig[:6])
            if len(orig) > 6:
                orig_str += f" …+{len(orig)-6}"

            track_cards.append(f"""
            <div class="track-card" title="track_key: {htmllib.escape(tkey)}">
              <img class="crop-img" src="{crop_uri}" alt="crop" loading="lazy">
              <div class="track-info">
                <span class="score-badge" style="background:{sc2}22;color:{sc2};border:1px solid {sc2}55">
                  {score:.4f}
                </span>
                <span class="track-rank">#{ri+1}</span>
              </div>
              <div class="track-meta">
                <div class="meta-row">🎬 {htmllib.escape(Path(video).stem if video!="—" else "—")}</div>
                <div class="meta-row">⏱ {t_str}</div>
                <div class="meta-row">📍 {n_pts} pts · {n_orig} tracks</div>
              </div>
            </div>""")

        tracks_html = "\n".join(track_cards)

        # cluster 요약
        all_videos = sorted({row.get("video") or "" for row in rows if row.get("video")})
        vid_list = ", ".join(Path(v).stem for v in all_videos[:4])
        if len(all_videos) > 4:
            vid_list += f" +{len(all_videos)-4}"

        cluster_cards.append(f"""
      <div class="cluster-card" id="cluster-{ci}">
        <div class="cluster-header" onclick="toggleCluster({ci})">
          <div class="cluster-left">
            <span class="cluster-rank">#{ci+1}</span>
            <span class="cluster-id" title="{htmllib.escape(cid)}">
              Cluster <code>{short}</code>
            </span>
            <span class="cluster-badge">{len(rows)} tracks</span>
            {f'<span class="vid-badge">{htmllib.escape(vid_list)}</span>' if vid_list else ''}
          </div>
          <div class="cluster-right">
            <span class="best-score" style="color:{sc}">
              best: {best_score:.4f}
            </span>
            <span class="toggle-icon" id="icon-{ci}">▼</span>
          </div>
        </div>
        <div class="cluster-body" id="body-{ci}">
          <div class="track-strip">
            {tracks_html}
          </div>
        </div>
      </div>""")

    clusters_html = "\n".join(cluster_cards)

    # query panel
    if query_uri:
        query_panel = f"""
        <div class="query-panel">
          <img class="query-img" src="{query_uri}" alt="query">
          <div class="query-info">
            <div class="qi-mode">Image Query</div>
            <div class="qi-path">{htmllib.escape(str(args.image))}</div>
          </div>
        </div>"""
    else:
        query_panel = f"""
        <div class="query-panel text-query">
          <div class="query-text-icon">💬</div>
          <div class="query-info">
            <div class="qi-mode">Text Query</div>
            <div class="qi-text">"{htmllib.escape(args.text or "")}"</div>
          </div>
        </div>"""

    # timing
    t_embed  = timing.get("embed", 0) or 0
    t_search = timing.get("search", 0) or 0
    t_total  = t_embed + t_search

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Video Cluster Search — {htmllib.escape(str(Path(query_label).stem if args.image else query_label[:30]))}</title>
<style>
:root {{
  --bg: #0a0a0f;
  --bg2: #12121a;
  --bg3: #1a1a26;
  --border: #2a2a3a;
  --accent: #6366f1;
  --accent2: #8b5cf6;
  --text: #e2e8f0;
  --text2: #94a3b8;
  --text3: #64748b;
  --radius: 10px;
  --shadow: 0 4px 20px rgba(0,0,0,0.5);
}}
* {{ box-sizing: border-box; margin:0; padding:0; }}
body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height:100vh; }}

/* ── header ── */
.page-header {{
  background: linear-gradient(135deg, #0a0a1f 0%, #12121a 100%);
  border-bottom: 1px solid var(--border);
  padding: 20px 28px;
  display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
}}
.logo {{ font-size:22px; font-weight:700; color:var(--accent); letter-spacing:-0.5px; }}
.logo span {{ color:var(--accent2); }}
.header-meta {{ display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--text2); }}
.meta-chip {{
  background: var(--bg3); border:1px solid var(--border);
  padding: 4px 10px; border-radius: 20px;
}}
.meta-chip.accent {{ border-color:var(--accent); color:var(--accent); }}

/* ── query panel ── */
.query-panel {{
  display:flex; align-items:center; gap:16px;
  background: var(--bg2); border:1px solid var(--border);
  border-radius: var(--radius); padding:16px 20px; margin:20px 28px;
  box-shadow: var(--shadow);
}}
.query-img {{ height:100px; width:auto; border-radius:6px; border:2px solid var(--border); object-fit:cover; }}
.text-query {{ border-color: var(--accent); }}
.query-text-icon {{ font-size:40px; }}
.qi-mode {{ font-size:11px; color:var(--text3); text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }}
.qi-path, .qi-text {{ font-size:14px; color:var(--text); word-break:break-all; }}
.qi-text {{ font-size:18px; font-style:italic; color:var(--accent2); }}

/* ── stats bar ── */
.stats-bar {{
  display:flex; gap:12px; flex-wrap:wrap;
  padding: 0 28px 16px;
}}
.stat-box {{
  background:var(--bg2); border:1px solid var(--border);
  border-radius:8px; padding:10px 18px;
  min-width:110px; text-align:center;
}}
.stat-val {{ font-size:22px; font-weight:700; color:var(--accent); }}
.stat-lbl {{ font-size:11px; color:var(--text3); margin-top:2px; }}

/* ── cluster cards ── */
.main-content {{ padding: 0 28px 40px; }}
.section-title {{ font-size:13px; color:var(--text3); text-transform:uppercase; letter-spacing:1px; margin-bottom:14px; }}

.cluster-card {{
  background: var(--bg2); border:1px solid var(--border);
  border-radius: var(--radius); margin-bottom:14px;
  box-shadow: var(--shadow); overflow:hidden;
  transition: border-color .2s;
}}
.cluster-card:hover {{ border-color: var(--accent); }}

.cluster-header {{
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 18px; cursor:pointer; user-select:none;
  background: var(--bg3);
  transition: background .15s;
}}
.cluster-header:hover {{ background: #1e1e30; }}
.cluster-left  {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
.cluster-right {{ display:flex; align-items:center; gap:14px; }}

.cluster-rank  {{ font-size:18px; font-weight:700; color:var(--accent); min-width:30px; }}
.cluster-id    {{ font-size:14px; color:var(--text); }}
.cluster-id code {{ background:#0a0a1f; padding:2px 6px; border-radius:4px; font-family:monospace; font-size:12px; color:var(--accent2); }}
.cluster-badge {{ background:#22c55e22; color:#22c55e; border:1px solid #22c55e44; border-radius:12px; font-size:12px; padding:2px 9px; }}
.vid-badge     {{ color:var(--text3); font-size:12px; }}
.best-score    {{ font-size:13px; font-weight:600; }}
.toggle-icon   {{ font-size:12px; color:var(--text3); transition:transform .2s; }}
.toggle-icon.open {{ transform:rotate(180deg); }}

.cluster-body  {{ padding:14px 16px; }}

/* ── track strip ── */
.track-strip {{
  display:flex; flex-wrap:wrap; gap:10px;
}}
.track-card {{
  background: var(--bg3); border:1px solid var(--border);
  border-radius:8px; width:110px; overflow:hidden;
  cursor:default; transition:border-color .15s, transform .15s;
  flex-shrink:0;
}}
.track-card:hover {{ border-color:var(--accent); transform:scale(1.03); }}
.crop-img {{ width:100%; height:160px; object-fit:cover; display:block; background:#111; }}
.track-info {{ display:flex; justify-content:space-between; align-items:center; padding:4px 6px; gap:4px; }}
.score-badge {{ font-size:10px; font-weight:600; padding:2px 5px; border-radius:4px; }}
.track-rank  {{ font-size:10px; color:var(--text3); }}
.track-meta  {{ padding:4px 6px 8px; font-size:10px; color:var(--text3); }}
.meta-row    {{ margin-bottom:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

/* ── toolbar ── */
.toolbar {{
  display:flex; gap:8px; align-items:center; margin-bottom:16px;
}}
.btn {{
  background: var(--bg3); border:1px solid var(--border); color:var(--text);
  padding:6px 14px; border-radius:6px; font-size:12px; cursor:pointer;
  transition: border-color .15s, background .15s;
}}
.btn:hover {{ border-color:var(--accent); background:#1e1e30; }}
.btn.accent {{ background: var(--accent); border-color: var(--accent); color:#fff; }}
</style>
</head>
<body>

<!-- header -->
<div class="page-header">
  <div class="logo">Forensic<span>Search</span></div>
  <div class="header-meta">
    <span class="meta-chip accent">{htmllib.escape(scope.upper())}</span>
    <span class="meta-chip">{'image-video' if args.image else 'text-video'}</span>
    <span class="meta-chip">vectors: {htmllib.escape(' + '.join(vectors))}</span>
    <span class="meta-chip">embed {t_embed:.2f}s · search {t_search:.2f}s</span>
    <span class="meta-chip">{ts}</span>
  </div>
</div>

<!-- query -->
{query_panel}

<!-- stats -->
<div class="stats-bar">
  <div class="stat-box"><div class="stat-val">{len(clusters)}</div><div class="stat-lbl">Clusters</div></div>
  <div class="stat-box"><div class="stat-val">{total_tracks}</div><div class="stat-lbl">Tracks</div></div>
  <div class="stat-box"><div class="stat-val">{args.top_k}</div><div class="stat-lbl">Requested</div></div>
  <div class="stat-box"><div class="stat-val">{t_total:.1f}s</div><div class="stat-lbl">Total Time</div></div>
</div>

<!-- content -->
<div class="main-content">
  <div class="toolbar">
    <button class="btn accent" onclick="expandAll()">모두 펼치기</button>
    <button class="btn" onclick="collapseAll()">모두 접기</button>
    <span style="font-size:12px;color:var(--text3);margin-left:8px;">
      클러스터 = Leiden 동일 인물 그룹 · 클릭하여 트랙 보기
    </span>
  </div>
  <div class="section-title">결과 — 클러스터 정렬</div>
  {clusters_html}
</div>

<script>
var states = {{}};
function toggleCluster(i) {{
  var body = document.getElementById('body-' + i);
  var icon = document.getElementById('icon-' + i);
  if (!states[i]) {{ states[i] = 'open'; }}
  if (states[i] === 'open') {{
    body.style.display = 'none'; icon.classList.remove('open'); states[i] = 'closed';
  }} else {{
    body.style.display = ''; icon.classList.add('open'); states[i] = 'open';
  }}
}}
function expandAll() {{
  document.querySelectorAll('.cluster-body').forEach(function(b,i) {{ b.style.display=''; }});
  document.querySelectorAll('.toggle-icon').forEach(function(ic) {{ ic.classList.add('open'); }});
  for(var k in states) states[k]='open';
}}
function collapseAll() {{
  document.querySelectorAll('.cluster-body').forEach(function(b) {{ b.style.display='none'; }});
  document.querySelectorAll('.toggle-icon').forEach(function(ic) {{ ic.classList.remove('open'); }});
  for(var k in states) states[k]='closed';
}}
// 기본: 상위 3개 펼침
[0,1,2].forEach(function(i) {{
  var b = document.getElementById('body-'+i);
  var ic = document.getElementById('icon-'+i);
  if(b) {{ b.style.display=''; if(ic) ic.classList.add('open'); states[i]='open'; }}
}});
// 나머지 접기
for(var j=3; j<{len(clusters)}; j++) {{
  var b2 = document.getElementById('body-'+j);
  if(b2) b2.style.display='none';
}}
</script>

</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="비디오 DB 클러스터 검색 → HTML")
    q = p.add_mutually_exclusive_group(required=True)
    q.add_argument("--image", "-i",  help="query crop 이미지 경로")
    q.add_argument("--text",  "-t",  help="검색 텍스트 (한글/영어)")
    p.add_argument("--scope", default="person", choices=["person", "object"])
    p.add_argument("--top-k", "-k", type=int, default=20)
    p.add_argument("--group-size", type=int, default=3)
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--vector", default=None, help="solider / irra / siglip2 / dinov2 (기본: scope별 자동)")
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("--translate-backend", default="none", choices=["opus","nllb","none"])
    p.add_argument("--out", default=None, help="출력 HTML 경로 (기본: search_results_html/)")
    p.add_argument("--open", action="store_true", help="생성 후 브라우저 자동 오픈")
    return p.parse_args()


def main():
    args = parse_args()

    t0 = time.time()
    result = run_search(args)
    rows   = result.get("results", [])
    t_total = time.time() - t0

    print(f"[결과] {len(rows)} tracks 반환  ({t_total:.1f}s)")

    if not rows:
        print("결과가 없습니다. Qdrant 서버가 실행 중인지 확인하세요.")
        sys.exit(1)

    # 클러스터 그룹핑
    clusters = group_by_cluster(rows)
    print(f"[클러스터] {len(clusters)} 개 클러스터")

    # HTML 생성
    html_str = build_html(result, args, clusters)

    # 저장
    if args.out:
        out_path = Path(args.out)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(args.image).stem if args.image else "text"
        out_path = OUT_DIR / f"video_search_{stem}_{ts_str}.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str, encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"\n✅  HTML 저장: {out_path}  ({size_kb:.0f} KB)")

    if args.open:
        webbrowser.open(out_path.as_uri())
        print("브라우저 열기 완료.")


if __name__ == "__main__":
    main()
