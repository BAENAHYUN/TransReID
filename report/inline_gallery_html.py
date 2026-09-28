#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inline_gallery_html.py — 기존 갤러리 HTML 의 assets/ 이미지를 base64 로 내장한 사본을 만든다.

build_leiden_gallery*.py 가 만든 HTML 은 crop 을 assets/ 에 복사하고 상대경로로
참조한다. 폴더째로 열면 잘 보이지만, 파일 하나만 옮기거나 앱 안에서 스냅샷으로
열면 이미지가 빈 칸이 된다. 이 스크립트는 **원본 결과를 다시 계산하지 않고**
HTML 텍스트만 변환한다 — 예전 클러스터링 결과를 그대로 보존한 채 자기완결
파일로 만든다.

    python inline_gallery_html.py --html outputs/clustering/.../leiden_track_gallery.html
    -> 같은 폴더에 <이름>_inline.html

이미지는 --thumb-size 로 줄여 넣는다 (기본 240px). 원본 assets/ 는 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from report_common import (
    local_reference, path_href, file_info, common_summary, warning, write_json, result_markers,
)

IMG_RE = re.compile(r'<img\b([^>]*?)\bsrc=(["\'])(?P<src>[^"\']+)\2([^>]*)>', re.IGNORECASE)
A_OPEN_RE = re.compile(r'<a\b[^>]*\bhref=(["\'])(?P<href>assets/[^"\']+)\1[^>]*>', re.IGNORECASE)


def to_data_uri(path: Path, max_side: int, quality: int) -> str | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def build_parser():
    ap = argparse.ArgumentParser(description="갤러리 HTML 의 assets/ 이미지를 base64 로 내장한 사본 생성")
    ap.add_argument("--html", required=True, help="원본 갤러리 HTML")
    ap.add_argument("--out", default=None, help="출력 경로 (기본: <원본>_inline.html)")
    ap.add_argument("--thumb-size", type=int, default=240)
    ap.add_argument("--thumb-quality", type=int, default=75)
    ap.add_argument("--drop-links", action="store_true", default=True,
                    help="assets/ 로 가는 <a href> 를 제거한다 (data URI 중복으로 크기 2배 방지). 기본 켬")
    ap.add_argument("--keep-links", action="store_false", dest="drop_links", help="assets 링크 유지")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.thumb_size < 16:
        ap.error("--thumb-size must be >= 16")
    if not 1 <= args.thumb_quality <= 95:
        ap.error("--thumb-quality must be in 1..95")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    src_path = Path(args.html).resolve()
    base_dir = src_path.parent
    out_path = Path(args.out).resolve() if args.out else src_path.with_name(src_path.stem + "_inline.html")

    html = src_path.read_text(encoding="utf-8")
    inputs = [file_info("source_html", src_path)]
    warnings = []
    embedded = 0
    missing = 0
    external_images = 0
    cache: dict[str, str | None] = {}

    def repl_img(m: re.Match) -> str:
        nonlocal embedded, missing, external_images
        src = m.group("src")
        import html as html_lib
        value = html_lib.unescape(src)
        if value.lower().startswith("data:"):
            return m.group(0)
        if value.lower().startswith(("http:", "https:", "//")):
            external_images += 1
            return m.group(0)
        p = local_reference(src, base_dir)
        rel = str(p)
        if rel not in cache:
            inputs.append(file_info("image", p))
            cache[rel] = to_data_uri(p, args.thumb_size, args.thumb_quality) if p.is_file() else None
        uri = cache[rel]
        if uri is None:
            missing += 1
            return m.group(0)
        embedded += 1
        q = m.group(2)
        return f"<img{m.group(1)}src={q}{uri}{q}{m.group(4)}>"

    new_html = IMG_RE.sub(repl_img, html)

    if args.drop_links:
        # <a href="assets/..."> ... </a> 의 여는 태그만 제거하고, 짝이 되는 </a> 는 놔둔다.
        # 갤러리 카드 구조가 <a ...><img ...></a> 라 여는 태그를 지우면 닫는 </a> 도 지워야 한다.
        new_html = re.sub(r'<a\b[^>]*\bhref=(["\'])assets/[^"\']+\1[^>]*>(\s*<img\b[^>]*>)\s*</a>',
                          r"\2", new_html, flags=re.IGNORECASE)
    elif out_path.parent != base_dir:
        def repl_link(match):
            path = local_reference(match.group('href'), base_dir)
            return match.group(0).replace(match.group('href'), path_href(path, out_path))
        new_html = A_OPEN_RE.sub(repl_link, new_html)

    # 제목에 표시
    new_html = new_html.replace("</h1>", " · inline</h1>", 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(new_html, encoding="utf-8", newline="\n")
    if missing:
        warnings.append(warning("MISSING_IMAGES", f"내장 실패 이미지: {missing}"))
    if external_images:
        warnings.append(warning("EXTERNAL_IMAGE_REMAINS", f"외부 이미지 잔존: {external_images}"))
    summary = common_summary("inline_gallery_html.py", out_path, inputs, warnings)
    summary.update(embedded=embedded, missing=missing, external_images=external_images,
                   self_contained=missing == 0 and external_images == 0)
    sidecar = out_path.with_suffix('.summary.json')
    write_json(sidecar, summary)
    size_mb = out_path.stat().st_size / 1048576
    print(f"source   : {src_path}")
    print(f"out      : {out_path}")
    print(f"embedded : {embedded:,}  missing: {missing:,}  size: {size_mb:.1f} MB")
    result_markers(sidecar, out_path)
    return 0 if missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
