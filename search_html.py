"""
search_html.py — crop 한 장 검색 → HTML 갤러리 즉시 오픈

    python search_html.py --crop data/query/person.jpg -k 30 --open
    python search_html.py --crop q.jpg -k 50 --solider --out results/my.html
    python search_html.py --crop q.jpg --object -k 20 --open
"""

from __future__ import annotations

import argparse
import base64
import html as htmllib
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_person_crop import CropSearcher, to_json, CONFIG_PATH

OUT_DIR = ROOT / "search_results_html"


# ─────────────────────────────────────────────────────────────────────────────
# 이미지 → base64 data URI
# ─────────────────────────────────────────────────────────────────────────────
MAX_EMBED = 3 * 1024 * 1024   # 3 MB 이하만 base64 embed


def img_uri(path_str: Optional[str]) -> str:
    """파일을 base64 data URI 로. 실패하면 placeholder SVG."""
    if not path_str:
        return _placeholder()
    p = Path(path_str)
    if not p.is_file():
        return _placeholder()
    if p.stat().st_size > MAX_EMBED:
        return p.as_uri()          # 너무 크면 file:// 링크
    try:
        data = p.read_bytes()
        ext  = p.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
        return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return _placeholder()


def _placeholder() -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="180">'
        '<rect width="100%" height="100%" fill="#2a2a2a"/>'
        '<text x="50%" y="50%" font-size="12" fill="#666" '
        'text-anchor="middle" dominant-baseline="middle">no image</text>'
        '</svg>'
    )
    enc = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{enc}"


# ─────────────────────────────────────────────────────────────────────────────
# score → 색상
# ─────────────────────────────────────────────────────────────────────────────
def score_color(s: float) -> str:
    if s >= 0.7:  return "#22c55e"   # green
    if s >= 0.5:  return "#f59e0b"   # amber
    if s >= 0.35: return "#f97316"   # orange
    return "#ef4444"                  # red


# ─────────────────────────────────────────────────────────────────────────────
# HTML 생성
# ─────────────────────────────────────────────────────────────────────────────
def build_html(payload: Dict[str, Any], query_path: str, scope: str) -> str:
    query_uri  = img_uri(query_path)
    query_name = htmllib.escape(Path(query_path).name)
    now        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hits = []
    for crop in payload.get("crops", []):
        for h in crop.get("results", []):
            hits.append(h)

    total = len(hits)

    # ── 결과 카드 HTML ──────────────────────────────────────────────────────
    cards_html = ""
    for h in hits:
        rank   = h.get("rank", "?")
        score  = h.get("pre_qwen_score", h.get("qdrant_score", 0.0))
        qs     = h.get("qdrant_score", 0.0)
        sol    = h.get("solider_score")
        cp     = h.get("crop_path", "")
        vid    = h.get("image_id", "")
        track  = h.get("track_id", "")
        fi     = h.get("frame_idx", "")
        uri    = img_uri(cp)
        col    = score_color(float(score))
        fname  = htmllib.escape(Path(cp).name if cp else "")

        sol_badge = ""
        if sol is not None:
            sol_col   = score_color(float(sol))
            sol_badge = (
                f'<span class="badge" style="background:{sol_col}20;'
                f'color:{sol_col};border-color:{sol_col}40">'
                f'SOL {float(sol):.4f}</span>'
            )

        cards_html += f"""
        <div class="card">
          <div class="rank">#{rank}</div>
          <div class="thumb-wrap">
            <img src="{uri}" alt="rank {rank}" loading="lazy">
          </div>
          <div class="info">
            <div class="score-row">
              <span class="score" style="color:{col}">{float(score):.4f}</span>
              <span class="badge" style="background:{col}20;color:{col};border-color:{col}40">
                q {float(qs):.4f}
              </span>
              {sol_badge}
            </div>
            <div class="meta">{htmllib.escape(str(vid))}</div>
            <div class="meta">track {htmllib.escape(str(track))} &nbsp;·&nbsp; frame {fi}</div>
            <div class="fname">{fname}</div>
          </div>
        </div>"""

    # ── 전체 HTML ──────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReID Search — {query_name}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:    #0f1117;
    --card:  #1a1d27;
    --border:#2d3148;
    --text:  #e2e8f0;
    --sub:   #94a3b8;
    --acc:   #6366f1;
  }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}

  /* ── header ── */
  .header {{ display:flex; align-items:flex-start; gap:24px; margin-bottom:32px;
             background:var(--card); border:1px solid var(--border);
             border-radius:16px; padding:20px; }}
  .query-thumb {{ flex:0 0 140px; }}
  .query-thumb img {{ width:140px; height:200px; object-fit:contain;
                      border-radius:8px; background:#111; display:block; }}
  .query-info h1 {{ font-size:1.3rem; font-weight:700; margin-bottom:8px; }}
  .query-info .tag {{ display:inline-block; padding:2px 10px; border-radius:20px;
                      font-size:0.75rem; margin-right:6px; margin-bottom:6px;
                      background:var(--acc)22; color:var(--acc);
                      border:1px solid var(--acc)44; }}
  .query-info .sub {{ color:var(--sub); font-size:0.82rem; margin-top:4px; }}

  /* ── grid ── */
  .grid {{ display:grid;
           grid-template-columns:repeat(auto-fill,minmax(175px,1fr));
           gap:14px; }}
  .card {{ background:var(--card); border:1px solid var(--border);
           border-radius:12px; overflow:hidden; transition:border-color .15s; }}
  .card:hover {{ border-color:var(--acc); }}
  .rank {{ font-size:0.7rem; color:var(--sub); padding:6px 10px 0;
           font-weight:600; letter-spacing:.05em; }}
  .thumb-wrap {{ padding:8px; }}
  .thumb-wrap img {{ width:100%; height:220px; object-fit:contain;
                     border-radius:6px; background:#111; display:block; }}
  .info {{ padding:8px 10px 12px; }}
  .score-row {{ display:flex; flex-wrap:wrap; align-items:center; gap:4px;
                margin-bottom:6px; }}
  .score {{ font-size:1.05rem; font-weight:700; }}
  .badge {{ font-size:0.68rem; padding:1px 6px; border-radius:20px;
            border:1px solid transparent; font-weight:600; }}
  .meta  {{ font-size:0.72rem; color:var(--sub); line-height:1.5;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .fname {{ font-size:0.65rem; color:#475569; margin-top:2px;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

  /* ── divider ── */
  .section-title {{ font-size:0.85rem; color:var(--sub); font-weight:600;
                    letter-spacing:.06em; text-transform:uppercase;
                    margin-bottom:14px; }}
</style>
</head>
<body>

<div class="header">
  <div class="query-thumb">
    <img src="{query_uri}" alt="query">
  </div>
  <div class="query-info">
    <h1>ReID Search Results</h1>
    <div>
      <span class="tag">scope: {htmllib.escape(scope)}</span>
      <span class="tag">top-{total}</span>
      <span class="tag">{now}</span>
    </div>
    <p class="sub" style="margin-top:8px">Query: <strong>{query_name}</strong></p>
    <p class="sub">Score 범례:
      <span style="color:#22c55e">■</span> ≥0.7 &nbsp;
      <span style="color:#f59e0b">■</span> ≥0.5 &nbsp;
      <span style="color:#f97316">■</span> ≥0.35 &nbsp;
      <span style="color:#ef4444">■</span> &lt;0.35
    </p>
  </div>
</div>

<p class="section-title">검색 결과 — {total}건</p>
<div class="grid">
{cards_html}
</div>

</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="crop 검색 → HTML 갤러리")
    ap.add_argument("--crop",  "-c", required=True)
    ap.add_argument("--limit", "-k", type=int, default=20)
    ap.add_argument("--object", dest="is_object", action="store_true")
    ap.add_argument("--names",  nargs="*", default=None)
    ap.add_argument("--solider", dest="solider_rerank", action="store_true")
    ap.add_argument("--with-solider", action="store_true")
    ap.add_argument("--solider-pool", type=int, default=200)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--crop-root", default=None)
    ap.add_argument("--out",  default=None, help="출력 HTML 경로 (기본: search_results_html/)")
    ap.add_argument("--open", action="store_true", help="생성 후 브라우저 자동 오픈")
    args = ap.parse_args()

    scope = "object" if args.is_object else "person"
    crop  = Path(args.crop).resolve()

    if not crop.is_file():
        sys.exit(f"[!] crop 이미지가 없습니다: {crop}")

    print(f"[*] 검색 중: {crop.name}  scope={scope}  top-{args.limit}")

    searcher = CropSearcher(
        config_path=args.config,
        crop_root=args.crop_root,
    )
    try:
        res = searcher.search(
            str(crop),
            scope=scope,
            limit=args.limit,
            names=args.names,
            with_solider=args.with_solider,
            solider_rerank=args.solider_rerank,
            solider_pool=args.solider_pool,
        )
    finally:
        searcher.release()

    payload = to_json(res)

    # ── 출력 경로 결정 ─────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = crop.stem[:40]
        out_path = OUT_DIR / f"{stem}_{scope}_k{args.limit}_{ts}.html"

    # ── HTML 생성 ──────────────────────────────────────────────────────────
    html_str = build_html(payload, str(crop), scope)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str, encoding="utf-8")

    print(f"[+] HTML 저장: {out_path}")
    print(f"    hits: {sum(len(c.get('hits',[])) for c in payload.get('crops',[]))}")

    if args.open:
        webbrowser.open(out_path.as_uri())
        print("[+] 브라우저 오픈")


if __name__ == "__main__":
    main()
