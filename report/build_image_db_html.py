#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_image_db_html.py — 이미지 DB를 자기완결 HTML 한 장으로 만든다.

forensic_person / forensic_object 에서 media_type == "image" 인 point 만 모아
요약 통계와 crop 썸네일 브라우저를 담은 HTML 파일 하나를 생성한다.
이미지는 base64 로 박아 넣으므로 다른 PC 에서 열어도 그림이 보인다.

읽기 전용이다. Qdrant 에 쓰지 않고 파일도 건드리지 않는다.

사용
----
    python build_image_db_html.py
    python build_image_db_html.py --samples 3000 --out image_db.html
    python build_image_db_html.py --media-type any      # 영상 포함
    python build_image_db_html.py --source final_db_candidates

media_type 이 비어 있는 오래된 point 가 섞여 있으면 --media-type any 로 한 번
돌려서 상단 요약의 media_type 분포를 확인한 뒤 다시 좁히면 된다.
"""

from __future__ import annotations

import argparse
import base64
import html as htmllib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from report_common import (
    normalize_sources, common_summary, file_info, warning, write_json,
    result_markers, load_manifests, latest_build, manifest_html,
    DEFAULT_CONFIG_PATH, CONFIG_HELP, load_pipeline_settings, resolve, applied_config,
)

ROOT = Path(__file__).resolve().parents[1]

# dataviz 기준 팔레트 (단일 계열이므로 slot 1 만 사용)
PALETTE = {
    "light": {
        "surface": "#fcfcfb", "surface2": "#f2f2ef", "line": "#dededa",
        "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#78776f",
        "series": "#2a78d6", "person": "#2a78d6", "object": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "surface2": "#232322", "line": "#35352f",
        "primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#8f8e84",
        "series": "#3987e5", "person": "#3987e5", "object": "#d95926",
    },
}

CROP_KEYS = ("crop_path", "path", "image_path")


# ---------------------------------------------------------------------------
# Qdrant 수집
# ---------------------------------------------------------------------------

def build_filter(models, media_type: str, source: Optional[str]):
    must = []
    if media_type and media_type != "any":
        must.append(models.FieldCondition(
            key="media_type", match=models.MatchValue(value=media_type)
        ))
    sources = normalize_sources(source)
    if sources:
        must.append(models.FieldCondition(
            key="source", match=(models.MatchValue(value=sources[0]) if len(sources) == 1
                                 else models.MatchAny(any=sources))
        ))
    return models.Filter(must=must) if must else None


def scroll_collection(
    client, models, collection: str, media_type: str,
    source: Optional[str], max_scan: int, batch: int, scan_meta=None,
) -> Tuple[List[Dict[str, Any]], int]:
    """payload 만 훑는다. 벡터는 가져오지 않는다."""
    flt = build_filter(models, media_type, source)
    rows: List[Dict[str, Any]] = []
    offset = None
    scanned = 0

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=min(batch, max_scan - scanned) if max_scan else batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for p in points:
            pl = p.payload or {}
            crop = None
            for k in CROP_KEYS:
                if pl.get(k):
                    crop = str(pl[k])
                    break
            rows.append({
                "point_id": str(p.id),
                "collection": collection,
                "label": str(pl.get("label") or "unknown"),
                "is_person": pl.get("is_person"),
                "media_type": str(pl.get("media_type") or ""),
                "source": str(pl.get("source") or ""),
                "image_id": str(pl.get("image_id") or ""),
                "score": float(pl.get("score") or 0.0),
                "bbox": pl.get("bbox") or [],
                "crop_path": crop,
                "dup": pl.get("duplicate_group_id"),
                "amb": pl.get("ambiguous_group_id"),
                "embedding_build_id": pl.get("embedding_build_id"),
            })

        scanned += len(points)
        print(f"  {collection}: {scanned:,} scanned", end="\r", flush=True)

        if offset is None:
            break
        if max_scan and scanned >= max_scan:
            break

    print(f"  {collection}: {scanned:,} scanned            ")
    if scan_meta is not None:
        truncated = bool(max_scan and scanned >= max_scan and offset is not None)
        scan_meta.update(count=len(rows), scanned=scanned, truncated=truncated,
                         stop_reason="max_scan" if truncated else "exhausted")
    return rows, scanned


# ---------------------------------------------------------------------------
# 통계
# ---------------------------------------------------------------------------

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = Counter(r["label"] for r in rows)
    media = Counter(r["media_type"] or "(없음)" for r in rows)
    sources = Counter(r["source"] or "(없음)" for r in rows)
    colls = Counter(r["collection"] for r in rows)

    n_person = sum(1 for r in rows if r["is_person"] is True)
    n_object = sum(1 for r in rows if r["is_person"] is False)
    n_unknown = len(rows) - n_person - n_object

    dup_groups = Counter(r["dup"] for r in rows if r["dup"])
    amb_groups = Counter(r["amb"] for r in rows if r["amb"])

    widths, heights = [], []
    for r in rows:
        b = r["bbox"]
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            try:
                widths.append(abs(float(b[2]) - float(b[0])))
                heights.append(abs(float(b[3]) - float(b[1])))
            except (TypeError, ValueError):
                pass

    def med(xs):
        if not xs:
            return 0.0
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    with_crop = sum(1 for r in rows if r["crop_path"])

    return {
        "total": len(rows),
        "labels": labels,
        "media": media,
        "sources": sources,
        "collections": colls,
        "n_person": n_person,
        "n_object": n_object,
        "n_unknown": n_unknown,
        "n_labels": len(labels),
        "dup_group_count": len(dup_groups),
        "dup_member_count": sum(dup_groups.values()),
        "amb_group_count": len(amb_groups),
        "amb_member_count": sum(amb_groups.values()),
        "median_w": med(widths),
        "median_h": med(heights),
        "with_crop": with_crop,
        "images": len({r["image_id"] for r in rows if r["image_id"]}),
        "build_ids": Counter(str(r["embedding_build_id"]) for r in rows if r.get("embedding_build_id")),
        "legacy": sum(1 for r in rows if not r.get("embedding_build_id")),
    }


# ---------------------------------------------------------------------------
# 표본 추출 — 라벨별로 고르게
# ---------------------------------------------------------------------------

def stratified_sample(
    rows: List[Dict[str, Any]], n: int, seed: int
) -> List[Dict[str, Any]]:
    usable = [r for r in rows if r["crop_path"]]
    if not usable or n <= 0:
        return []
    if len(usable) <= n:
        return usable

    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in usable:
        by_label[r["label"]].append(r)
    for v in by_label.values():
        rng.shuffle(v)

    # 라벨을 돌아가며 한 장씩 뽑아 희귀 라벨이 통째로 빠지지 않게 한다.
    picked: List[Dict[str, Any]] = []
    order = sorted(by_label, key=lambda k: -len(by_label[k]))
    idx = {k: 0 for k in order}
    while len(picked) < n:
        progressed = False
        for k in order:
            i = idx[k]
            if i < len(by_label[k]):
                picked.append(by_label[k][i])
                idx[k] = i + 1
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    return picked


# ---------------------------------------------------------------------------
# 썸네일
# ---------------------------------------------------------------------------

def load_thumbs(
    rows: List[Dict[str, Any]], size: int, quality: int, crop_root: Optional[Path], thumb_meta=None
) -> Tuple[int, int]:
    if thumb_meta is not None:
        thumb_meta.update(fallback_used=0)
    try:
        from PIL import Image
    except ImportError:
        print("  [경고] Pillow 가 없어 썸네일을 건너뜁니다. pip install pillow")
        return 0, len(rows)

    ok = miss = 0
    for i, r in enumerate(rows, 1):
        raw = r["crop_path"]
        p = Path(str(raw).replace("\\", "/"))
        if not p.is_absolute() and not p.is_file():
            p = ROOT / p
        if not p.is_file() and crop_root:
            cand = crop_root / p.name
            if cand.is_file():
                p = cand
                if thumb_meta is not None:
                    thumb_meta["fallback_used"] += 1
        if not p.is_file():
            miss += 1
            continue
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((size, size * 2), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=quality, optimize=True)
            r["thumb"] = base64.b64encode(buf.getvalue()).decode("ascii")
            ok += 1
        except Exception:  # noqa: BLE001
            miss += 1
        if i % 200 == 0:
            print(f"  썸네일 {i:,}/{len(rows):,}", end="\r", flush=True)
    print(f"  썸네일 {ok:,}장 생성, {miss:,}장 실패            ")
    return ok, miss


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def css_vars(mode: str) -> str:
    c = PALETTE[mode]
    return "\n".join(f"      --{k}: {v};" for k, v in c.items())


def bar_rows(labels: Counter, top: int) -> str:
    if not labels or top < 1:
        return '<p class="empty">라벨이 없습니다.</p>'
    items = labels.most_common(top)
    mx = items[0][1] or 1
    out = []
    for name, cnt in items:
        pct = cnt / mx * 100
        share = cnt / (sum(labels.values()) or 1) * 100
        out.append(
            f'<div class="bar-row">'
            f'<div class="bar-name" title="{htmllib.escape(name)}">'
            f'{htmllib.escape(name)}</div>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{pct:.2f}%"></div></div>'
            f'<div class="bar-val">{cnt:,}<span class="bar-pct">'
            f'{share:.1f}%</span></div>'
            f'</div>'
        )
    return "\n".join(out)


def kv_rows(counter: Counter, top: int = 12) -> str:
    if not counter:
        return '<tr><td colspan="2" class="muted">없음</td></tr>'
    total = sum(counter.values())
    out = []
    for k, v in counter.most_common(top):
        out.append(
            f"<tr><td>{htmllib.escape(str(k))}</td>"
            f"<td class='num'>{v:,}<span class='bar-pct'>"
            f"{v/total*100:.1f}%</span></td></tr>"
        )
    return "\n".join(out)


def build_html(stats: Dict[str, Any], sample: List[Dict[str, Any]],
               meta: Dict[str, Any]) -> str:
    partial = bool(meta.get("부분 스캔"))
    manifests = stats.get("manifests", {})
    latest = latest_build(manifests)
    build_note = (latest or "시각 미확인") if stats.get("build_ids") else "legacy 만"
    cards = [
        ("전체 point", f"{stats['total']:,}", "부분 스캔 — 수집된 point 수" if partial else "이미지 DB 내 point 수"),
        ("임베딩 build", build_note, f"legacy {stats.get('legacy', 0):,}"),
        ("원본 이미지", f"{stats['images']:,}", "서로 다른 image_id"),
        ("라벨 종류", f"{stats['n_labels']:,}", "detection class"),
        ("person / object", f"{stats['n_person']:,} / {stats['n_object']:,}",
         f"미분류 {stats['n_unknown']:,}"),
        ("중복 그룹", f"{stats['dup_group_count']:,}",
         f"소속 point {stats['dup_member_count']:,}"),
        ("모호 그룹", f"{stats['amb_group_count']:,}",
         f"소속 point {stats['amb_member_count']:,}"),
        ("crop 경로 보유", f"{stats['with_crop']:,}",
         f"{stats['with_crop']/max(stats['total'],1)*100:.1f}%"),
        ("중간 crop 크기", f"{stats['median_w']:.0f}×{stats['median_h']:.0f}",
         "bbox 기준 픽셀"),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="card-label">{htmllib.escape(t)}</div>'
        f'<div class="card-value">{htmllib.escape(v)}</div>'
        f'<div class="card-note">{htmllib.escape(n)}</div></div>'
        for t, v, n in cards
    )

    label_options = "\n".join(
        f'<option value="{htmllib.escape(k)}">{htmllib.escape(k)} ({v:,})</option>'
        for k, v in stats["labels"].most_common()
    )

    tiles = []
    for r in sample:
        thumb = r.get("thumb")
        kind = ("person" if r["is_person"] is True
                else "object" if r["is_person"] is False else "unknown")
        name = Path(str(r["crop_path"] or "")).name
        img = (f'<img loading="lazy" src="data:image/jpeg;base64,{thumb}" '
               f'alt="{htmllib.escape(name)}">'
               if thumb else '<div class="no-img">이미지 없음</div>')
        tiles.append(
            f'<figure class="tile" data-label="{htmllib.escape(r["label"])}" '
            f'data-kind="{kind}" '
            f'data-text="{htmllib.escape((r["label"]+" "+name+" "+r["image_id"]).lower())}">'
            f'{img}'
            f'<figcaption>'
            f'<span class="tag tag-{kind}">{htmllib.escape(r["label"])}</span>'
            f'<span class="fname" title="{htmllib.escape(name)}">'
            f'{htmllib.escape(name)}</span>'
            f'<span class="meta">{htmllib.escape(r["image_id"][:44])}</span>'
            f'</figcaption></figure>'
        )
    tiles_html = "\n".join(tiles) or '<p class="empty">표본이 없습니다.</p>'

    meta_rows = "\n".join(
        f"<tr><td>{htmllib.escape(k)}</td><td>{htmllib.escape(str(v))}</td></tr>"
        for k, v in meta.items()
    )

    return f"""<!DOCTYPE html>
<html lang="ko" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>이미지 DB 리포트</title>
<style>
  :root {{
{css_vars("light")}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
{css_vars("dark")}
    }}
  }}
  :root[data-theme="dark"] {{
{css_vars("dark")}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0 16px 64px;
    background: var(--surface); color: var(--primary);
    font: 14px/1.6 -apple-system, "Segoe UI", "Malgun Gothic", system-ui, sans-serif;
  }}
  .wrap {{ max-width: 1240px; margin: 0 auto; }}
  header {{ padding: 28px 0 18px; border-bottom: 1px solid var(--line); }}
  h1 {{ margin: 0 0 6px; font-size: 22px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--secondary); font-size: 13px; }}
  h2 {{ font-size: 15px; margin: 30px 0 12px; color: var(--primary); }}
  .muted {{ color: var(--muted); }}

  .cards {{ display: grid; gap: 10px; margin-top: 18px;
           grid-template-columns: repeat(auto-fill, minmax(178px, 1fr)); }}
  .card {{ background: var(--surface2); border: 1px solid var(--line);
          border-radius: 8px; padding: 12px 14px; }}
  .card-label {{ font-size: 11px; color: var(--muted); letter-spacing: .02em; }}
  .card-value {{ font-size: 21px; font-weight: 700; margin: 3px 0 2px;
                font-variant-numeric: tabular-nums; }}
  .card-note {{ font-size: 11px; color: var(--secondary); }}

  .split {{ display: grid; gap: 22px; grid-template-columns: 1.55fr 1fr; }}
  @media (max-width: 860px) {{ .split {{ grid-template-columns: 1fr; }} }}

  .bar-row {{ display: grid; grid-template-columns: 116px 1fr 104px;
             align-items: center; gap: 10px; margin-bottom: 2px; }}
  .bar-name {{ font-size: 12px; color: var(--secondary);
              overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .bar-track {{ height: 15px; background: var(--surface2); border-radius: 4px; }}
  .bar-fill {{ height: 100%; background: var(--series);
              border-radius: 0 4px 4px 0; min-width: 2px; }}
  .bar-val {{ font-size: 12px; text-align: right; color: var(--secondary);
             font-variant-numeric: tabular-nums; }}
  .bar-pct {{ color: var(--muted); margin-left: 7px; font-size: 11px; }}

  table.kv {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  table.kv td {{ padding: 5px 8px; border-bottom: 1px solid var(--line);
                color: var(--secondary); }}
  table.kv td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

  .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
              margin: 14px 0 16px; }}
  input, select {{ background: var(--surface2); color: var(--primary);
                  border: 1px solid var(--line); border-radius: 6px;
                  padding: 7px 10px; font-size: 13px; font-family: inherit; }}
  input[type=search] {{ min-width: 230px; }}
  button {{ background: var(--surface2); color: var(--secondary);
           border: 1px solid var(--line); border-radius: 6px;
           padding: 7px 12px; font-size: 13px; cursor: pointer;
           font-family: inherit; }}
  button.on {{ background: var(--series); color: #fff; border-color: var(--series); }}
  #count {{ color: var(--muted); font-size: 12px; margin-left: auto; }}

  .grid {{ display: grid; gap: 9px;
          grid-template-columns: repeat(auto-fill, minmax(116px, 1fr)); }}
  .tile {{ margin: 0; background: var(--surface2); border: 1px solid var(--line);
          border-radius: 7px; overflow: hidden; }}
  .tile img {{ width: 100%; height: 132px; object-fit: cover; display: block;
              background: var(--line); }}
  .no-img {{ height: 132px; display: flex; align-items: center;
            justify-content: center; color: var(--muted); font-size: 11px; }}
  figcaption {{ padding: 6px 7px 7px; display: flex; flex-direction: column;
               gap: 2px; }}
  .tag {{ font-size: 10px; padding: 1px 6px; border-radius: 3px;
         align-self: flex-start; color: #fff; }}
  .tag-person {{ background: var(--person); }}
  .tag-object {{ background: var(--object); }}
  .tag-unknown {{ background: var(--muted); }}
  .fname, .meta {{ font-size: 10.5px; color: var(--secondary);
                  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .meta {{ color: var(--muted); }}
  .empty {{ color: var(--muted); padding: 20px 0; }}
  footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--line);
           color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>이미지 DB 리포트{' · 부분 스캔' if partial else ''}</h1>
  <div class="sub">{htmllib.escape(meta.get('생성 시각', ''))} ·
    {htmllib.escape(meta.get('컬렉션', ''))} · media_type={htmllib.escape(meta.get('media_type 필터', ''))}</div>
</header>

<div class="cards">{card_html}</div>

<div class="split">
  <section>
    <h2>라벨 분포 (상위 {meta.get('차트 라벨 수', 25)}종)</h2>
    {bar_rows(stats['labels'], int(meta.get('차트 라벨 수', 25)))}
  </section>
  <section>
    <h2>구성</h2>
    <table class="kv">
      <tr><td colspan="2" class="muted">컬렉션</td></tr>
      {kv_rows(stats['collections'])}
      <tr><td colspan="2" class="muted">media_type</td></tr>
      {kv_rows(stats['media'])}
      <tr><td colspan="2" class="muted">source</td></tr>
      {kv_rows(stats['sources'], 8)}
      <tr><td colspan="2" class="muted">build id 분포</td></tr>
      {kv_rows(stats.get('build_ids', Counter()), max(1, len(stats.get('build_ids', {}))))}
    </table>
  </section>
</div>

<h2>참조 build manifest (입력 stats 수량은 현재 필터의 DB point 수가 아님)</h2>
{manifest_html(manifests)}
<h2>crop 표본 ({len(sample):,}장)</h2>
<div class="controls">
  <input type="search" id="q" placeholder="라벨 · 파일명 · image_id 검색">
  <select id="lab"><option value="">라벨 전체</option>{label_options}</select>
  <button data-kind="" class="on">전체</button>
  <button data-kind="person">person</button>
  <button data-kind="object">object</button>
  <span id="count"></span>
</div>
<div class="grid" id="grid">{tiles_html}</div>

<footer>
  <table class="kv" style="max-width:640px">{meta_rows}</table>
  <p>표본은 라벨별로 돌아가며 뽑아 희귀 라벨이 빠지지 않게 했습니다.
     따라서 표본 내 라벨 비율은 DB 전체 비율과 다릅니다 — 비율은 위 분포를 보세요.</p>
</footer>

</div>
<script>
(function () {{
  var grid = document.getElementById('grid');
  var tiles = Array.prototype.slice.call(grid.querySelectorAll('.tile'));
  var q = document.getElementById('q');
  var lab = document.getElementById('lab');
  var countEl = document.getElementById('count');
  var kindBtns = Array.prototype.slice.call(
    document.querySelectorAll('button[data-kind]'));
  var kind = '';

  function apply() {{
    var text = (q.value || '').trim().toLowerCase();
    var label = lab.value;
    var shown = 0;
    for (var i = 0; i < tiles.length; i++) {{
      var t = tiles[i];
      var ok = (!label || t.dataset.label === label)
            && (!kind || t.dataset.kind === kind)
            && (!text || t.dataset.text.indexOf(text) !== -1);
      t.style.display = ok ? '' : 'none';
      if (ok) shown++;
    }}
    countEl.textContent = shown.toLocaleString() + ' / '
                        + tiles.length.toLocaleString() + ' 장';
  }}

  q.addEventListener('input', apply);
  lab.addEventListener('change', apply);
  kindBtns.forEach(function (b) {{
    b.addEventListener('click', function () {{
      kind = b.dataset.kind;
      kindBtns.forEach(function (x) {{ x.classList.remove('on'); }});
      b.classList.add('on');
      apply();
    }});
  }});
  apply();
}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="이미지 DB를 자기완결 HTML 한 장으로 만든다 (읽기 전용)"
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    ap.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--person-collection", default=None, help=CONFIG_HELP)
    ap.add_argument("--object-collection", default=None, help=CONFIG_HELP)
    ap.add_argument("--media-type", default="image",
                    help="image | video | any (기본 image)")
    ap.add_argument("--source", default=None, help="source payload 필터")
    ap.add_argument("--samples", type=int, default=2000,
                    help="썸네일 표본 수 (기본 2000)")
    ap.add_argument("--thumb-size", type=int, default=160)
    ap.add_argument("--thumb-quality", type=int, default=72)
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--crop-root", default=None,
                    help="payload 경로가 안 맞을 때 파일명으로 찾을 폴더")
    ap.add_argument("--max-scan", type=int, default=0,
                    help="컬렉션당 최대 스캔 수 (0=전체)")
    ap.add_argument("--scroll-batch", type=int, default=1024)
    ap.add_argument("--chart-labels", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="image_db_report.html")
    ap.add_argument("--manifest-dir", default="data/build_manifests")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    for name, minimum in (("chart_labels", 1), ("samples", 0), ("thumb_size", 16),
                          ("scroll_batch", 1), ("max_scan", 0)):
        if getattr(args, name) < minimum:
            ap.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    if not 1 <= args.thumb_quality <= 95:
        ap.error("--thumb-quality must be in 1..95")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    names = ('person_collection', 'object_collection', 'qdrant_url')
    try:
        settings = load_pipeline_settings(args.config, require=not all(
            getattr(args, name) not in (None, '') for name in names))
        for name in names:
            setattr(args, name, resolve(getattr(args, name), getattr(settings, name, None), name))
    except (OSError, ValueError) as exc:
        build_parser().error(str(exc))

    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        print("qdrant-client 가 필요합니다: pip install qdrant-client")
        return 2

    client = QdrantClient(
        url=args.qdrant_url, api_key=args.api_key, timeout=args.timeout
    )

    print("=" * 66)
    print("이미지 DB 수집")
    print("=" * 66)

    rows: List[Dict[str, Any]] = []
    used: List[str] = []
    per_collection, warnings, inputs = {}, [], []
    if settings is None:
        warnings.append(warning('CONFIG_UNAVAILABLE', f'config 없음: {args.config}; 명시 CLI 사용'))
    else:
        inputs.append(file_info('config', settings.config_path))
    for coll in (args.person_collection, args.object_collection):
        try:
            if not client.collection_exists(coll):
                print(f"  [건너뜀] 컬렉션 없음: {coll}")
                warnings.append(warning("COLLECTION_MISSING", f"컬렉션 없음: {coll}"))
                per_collection[coll] = dict(count=0, scanned=0, truncated=False, stop_reason="exhausted")
                continue
        except Exception as e:  # noqa: BLE001
            print(f"  [오류] {coll} 확인 실패: {e}")
            return 3
        scan_meta = {}
        got, _ = scroll_collection(
            client, models, coll, args.media_type,
            args.source, args.max_scan, args.scroll_batch, scan_meta,
        )
        per_collection[coll] = scan_meta
        if scan_meta["truncated"]:
            warnings.append(warning("PARTIAL_SCAN", f"{coll}: 부분 스캔 ({scan_meta['scanned']})"))
        rows.extend(got)
        used.append(coll)

    if not rows:
        print()
        print("조건에 맞는 point 가 0건입니다.")
        print("  --media-type any 로 한 번 돌려 media_type 분포를 확인해 보세요.")
        return 1

    stats = summarize(rows)
    print(f"\n수집 완료: {stats['total']:,} point / 라벨 {stats['n_labels']}종")

    sample: List[Dict[str, Any]] = []
    thumbnails = dict(enabled=not args.no_thumbs, sampled=0, ok=0, missing=0, fallback_used=0)
    if not args.no_thumbs:
        sample = stratified_sample(rows, args.samples, args.seed)
        print(f"표본 {len(sample):,}장 썸네일 생성 중...")
        crop_root = Path(args.crop_root).resolve() if args.crop_root else None
        ok, miss = load_thumbs(sample, args.thumb_size, args.thumb_quality, crop_root, thumbnails)
        thumbnails.update(sampled=len(sample), ok=ok, missing=miss)
        if miss:
            print(f"[경고] 선택 표본 썸네일 누락: {miss}")
            warnings.append(warning("THUMBNAILS_MISSING", f"선택 표본 썸네일 누락: {miss}"))

    stats["manifests"] = load_manifests(args.manifest_dir, stats["build_ids"], warnings, inputs)

    meta = {
        "yaml 파일": Path(settings.config_path).name if settings else '미기록 (CLI)',
        "생성 시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "컬렉션": " + ".join(used),
        "media_type 필터": args.media_type,
        "source 필터": args.source or "(없음)",
        "전체 point": f"{stats['total']:,}",
        "표본 수": f"{len(sample):,}",
        "썸네일 최대 변": f"{args.thumb_size}px",
        "차트 라벨 수": args.chart_labels,
        "Qdrant": args.qdrant_url,
        "부분 스캔": any(v["truncated"] for v in per_collection.values()),
        "썸네일 상태 (선택 표본 기준)": thumbnails,
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(stats, sample, meta), encoding="utf-8", newline="\n")
    inputs.append(file_info("out_html", out))
    summary = common_summary("build_image_db_html.py", out, inputs, warnings)
    summary['config'] = applied_config(settings, **{name: getattr(args, name) for name in names})
    summary.update({key: stats[key] for key in ("total", "images", "n_person", "n_object", "n_unknown", "n_labels", "build_ids", "legacy")})
    summary.update(qdrant_url=args.qdrant_url, collections=used, per_collection=per_collection,
                   filters=dict(source=normalize_sources(args.source), media_type=args.media_type),
                   labels_top=stats["labels"].most_common(25), thumbnails=thumbnails, sample_size=len(sample))
    sidecar = out.with_suffix(".summary.json")
    write_json(sidecar, summary)
    mb = out.stat().st_size / (1024 * 1024)

    print()
    print("=" * 66)
    print(f"생성 완료: {out}")
    print(f"크기: {mb:.1f} MB")
    print("=" * 66)
    result_markers(sidecar, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
