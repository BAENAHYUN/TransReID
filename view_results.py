"""
view_results.py — 검색 결과를 HTML 갤러리로 보기
==============================================

    python search_person_crop.py --crop q.jpg -k 20 --solider --json-out r.json
    python view_results.py --in r.json --open

쿼리 crop 과 후보 crop 을 나란히 띄운 HTML 을 만든다. 숫자만 보고는 순위가
그럴듯한지 알 수 없으므로, 눈으로 확인하는 것이 사실상 유일한 검증 수단이다.

무엇을 봐야 하는가
----------------
1) **1위가 쿼리 자신인가** — DB 에 있는 crop 을 쿼리로 넣었다면 그래야 한다.
   아니면 색인과 질의의 전처리가 어긋난 것이다.

2) **점수 차이가 순위와 맞는가** — SOLIDER 점수가 0.9544~0.9227 처럼 좁은
   범위에 몰려 있으면, 그 순위는 노이즈일 수 있다. ReID 임베딩은 "둘 다
   사람"이라는 공통 특징 때문에 기저 코사인이 높게 깔린다. 0.92 가 "같은
   사람"을 뜻하는지 "그냥 사람"을 뜻하는지는 이미지를 봐야 안다.

3) **retriever 별 순위가 왜 다른가** — solider 와 qdrant 점수가 함께 표시된다.
   SOLIDER 2위인데 qdrant 0.0168 이면 두 모델이 완전히 다른 판단을 한 것이다.
   어느 쪽이 맞는지는 이미지로만 판정할 수 있다.

이미지를 어떻게 넣는가
-------------------
기본은 base64 로 HTML 에 직접 박는다(--embed). 파일을 옮겨도 깨지지 않고,
브라우저의 로컬 파일 접근 제한에도 걸리지 않는다. 후보가 많으면 파일이
커지므로 --no-embed 로 상대경로 링크를 쓸 수도 있다.

사용
----
    python view_results.py --in r.json                    # results.html 생성
    python view_results.py --in r.json --open             # 만들고 브라우저로 열기
    python view_results.py --in r.json --out compare.html
    python view_results.py --in a.json b.json --open      # 여러 결과 비교
    python view_results.py --in r.json --top 10 --no-embed
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent

# 이 값보다 큰 이미지는 base64 로 넣지 않는다 (HTML 이 너무 커진다)
MAX_EMBED_BYTES = 2 * 1024 * 1024


def data_uri(path: Path) -> Optional[str]:
    """이미지를 base64 data URI 로. 실패하면 None."""
    try:
        size = path.stat().st_size
        if size > MAX_EMBED_BYTES:
            return None
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def img_src(raw: Optional[str], embed: bool, out_dir: Path) -> Optional[str]:
    """HTML img src 를 만든다."""
    if not raw:
        return None
    p = Path(raw)
    if not p.is_file():
        return None

    if embed:
        uri = data_uri(p)
        if uri:
            return uri

    # base64 실패 또는 --no-embed: file:// 절대경로
    return p.resolve().as_uri()


def score_bar(value: float, lo: float, hi: float) -> str:
    """점수를 막대 너비(%)로. 좁은 범위를 눈에 보이게 늘려 준다."""
    if hi - lo < 1e-9:
        return "50"
    pct = (value - lo) / (hi - lo) * 100.0
    return f"{max(2.0, min(100.0, pct)):.1f}"


def collect_rows(payload: Dict[str, Any], top: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in payload.get("crops") or []:
        for row in (item.get("results") or [])[:top]:
            rows.append(row)
    return rows


def render_one(
    payload: Dict[str, Any],
    label: str,
    top: int,
    embed: bool,
    out_dir: Path,
) -> str:
    parts: List[str] = []

    crops = payload.get("crops") or []
    if not crops:
        return f"<section><h2>{html.escape(label)}</h2><p>결과가 없습니다.</p></section>"

    for item in crops:
        rows = (item.get("results") or [])[:top]

        query_crop = item.get("query_crop")
        query_text = item.get("query_text") or item.get("query_text_original")

        # ---- 헤더 ----
        stages = item.get("stages") or payload.get("stages") or []
        meta_bits = []
        if item.get("vectors_used"):
            meta_bits.append("벡터: " + " + ".join(item["vectors_used"]))
        if item.get("reranked_by_solider"):
            meta_bits.append("SOLIDER 재정렬 ON")
        if payload.get("qwen"):
            meta_bits.append(f"Qwen: {payload.get('qwen_mode')}")
        if item.get("timing"):
            t = item["timing"]
            meta_bits.append(
                "소요: " + " / ".join(f"{k} {v}s" for k, v in t.items())
            )

        parts.append('<section class="run">')
        parts.append(f"<h2>{html.escape(label)}</h2>")

        if meta_bits:
            parts.append(
                '<p class="meta">' + " · ".join(html.escape(m) for m in meta_bits)
                + "</p>"
            )
        for s in stages:
            parts.append(f'<p class="stage">{html.escape(str(s))}</p>')

        # ---- 쿼리 ----
        parts.append('<div class="query">')
        if query_text:
            parts.append(
                '<div class="qtext">쿼리 문장<br><strong>'
                + html.escape(str(query_text)) + "</strong></div>"
            )
        if query_crop:
            src = img_src(query_crop, embed, out_dir)
            name = html.escape(Path(str(query_crop)).name)
            if src:
                parts.append(
                    f'<figure class="qimg"><img src="{src}" alt="query">'
                    f"<figcaption>쿼리<br>{name}</figcaption></figure>"
                )
            else:
                parts.append(
                    f'<div class="missing">쿼리 이미지를 열 수 없습니다<br>{name}</div>'
                )
        parts.append("</div>")

        if not rows:
            parts.append("<p>결과가 없습니다.</p></section>")
            continue

        # ---- 점수 범위 (막대 스케일용) ----
        def values(key: str) -> List[float]:
            return [
                float(r[key]) for r in rows
                if r.get(key) is not None
            ]

        primary_key = (
            "solider_score" if any(r.get("solider_score") is not None for r in rows)
            else "qdrant_score"
        )
        vals = values(primary_key)
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)

        if vals and (hi - lo) < 0.05:
            parts.append(
                '<p class="warn">주의: '
                + html.escape(
                    f"{primary_key} 범위가 {lo:.4f}~{hi:.4f} 로 매우 좁습니다 "
                    f"(차이 {hi - lo:.4f}). 이 순위는 노이즈일 수 있습니다. "
                    f"이미지를 직접 비교해 판단하세요."
                )
                + "</p>"
            )

        # ---- 후보 ----
        parts.append('<div class="grid">')

        query_name = Path(str(query_crop)).name if query_crop else None

        for r in rows:
            path = r.get("crop_path")
            name = Path(str(path)).name if path else str(r.get("point_id", ""))[:12]
            is_self = bool(query_name and name == query_name)

            src = img_src(path, embed, out_dir)

            classes = ["card"]
            if is_self:
                classes.append("self")
            verified = r.get("verified")
            if verified is True:
                classes.append("ok")
            elif verified is False:
                classes.append("no")

            parts.append(f'<figure class="{" ".join(classes)}">')

            rank = r.get("rank", "?")
            badge = f'<span class="rank">{rank}</span>'
            if is_self:
                badge += '<span class="tag self-tag">쿼리 자신</span>'
            if verified is True:
                badge += '<span class="tag ok-tag">확인</span>'
            elif verified is False:
                badge += '<span class="tag no-tag">미달</span>'
            parts.append(f'<div class="badges">{badge}</div>')

            if src:
                parts.append(f'<img src="{src}" alt="{html.escape(name)}">')
            else:
                parts.append('<div class="missing">이미지 없음</div>')

            # 점수들
            lines: List[str] = []
            for key, caption in (
                ("solider_score", "SOLIDER"),
                ("qwen_score", "Qwen"),
                ("qwen_embed_score", "Qwen-emb"),
                ("qwen_judge_score", "Qwen-judge"),
                ("final_score", "final"),
                ("qdrant_score", "qdrant"),
            ):
                v = r.get(key)
                if v is None:
                    continue
                lines.append(f"{caption}={float(v):.4f}")

            primary = r.get(primary_key)
            if primary is not None:
                w = score_bar(float(primary), lo, hi)
                parts.append(f'<div class="bar"><i style="width:{w}%"></i></div>')

            parts.append('<figcaption>')
            parts.append(
                f'<span class="label">{html.escape(str(r.get("label", "")))}</span> '
            )
            parts.append("<br>".join(html.escape(x) for x in lines))
            parts.append(f'<div class="fname">{html.escape(name)}</div>')
            if r.get("image_id"):
                parts.append(
                    f'<div class="fname">{html.escape(str(r["image_id"]))}</div>'
                )
            if r.get("qwen_reason"):
                parts.append(
                    f'<div class="reason">{html.escape(str(r["qwen_reason"]))}</div>'
                )
            parts.append("</figcaption></figure>")

        parts.append("</div></section>")

    return "\n".join(parts)


CSS = """
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px;
  background: #14161a; color: #e8e8ea;
  font: 14px/1.5 "Segoe UI", "Malgun Gothic", system-ui, sans-serif;
}
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 8px; color: #9ecbff; }
.top { border-bottom: 1px solid #2a2e36; padding-bottom: 16px; }
.meta, .stage { color: #9aa0a8; font-size: 12px; margin: 2px 0; }
.warn {
  background: #3a2e12; border: 1px solid #6b5416; color: #f0d894;
  padding: 10px 12px; border-radius: 6px; font-size: 13px; margin: 12px 0;
}
.query { display: flex; gap: 16px; align-items: flex-start; margin: 16px 0 8px; }
.qtext {
  background: #1d2027; border: 1px solid #2a2e36; border-radius: 6px;
  padding: 12px 14px; font-size: 13px; color: #c8cdd4; max-width: 520px;
}
.qimg { margin: 0; }
.qimg img {
  max-height: 260px; border: 2px solid #4a90d9; border-radius: 6px;
  display: block; background: #000;
}
.qimg figcaption { font-size: 11px; color: #9aa0a8; margin-top: 4px; }
.grid {
  display: grid; gap: 14px;
  grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
}
.card {
  background: #1d2027; border: 1px solid #2a2e36; border-radius: 8px;
  padding: 8px; margin: 0; position: relative;
}
.card.self { border-color: #4a90d9; background: #1b2431; }
.card.ok { border-color: #3d8b40; }
.card.no { border-color: #7a3b3b; }
.card img {
  width: 100%; height: 190px; object-fit: contain;
  background: #000; border-radius: 4px; display: block;
}
.badges { display: flex; gap: 4px; align-items: center; margin-bottom: 6px;
          flex-wrap: wrap; }
.rank {
  background: #2a2e36; color: #e8e8ea; border-radius: 4px;
  padding: 1px 7px; font-weight: 600; font-size: 12px;
}
.tag { font-size: 10px; padding: 1px 6px; border-radius: 3px; }
.self-tag { background: #4a90d9; color: #08121d; font-weight: 600; }
.ok-tag { background: #3d8b40; color: #eafaea; }
.no-tag { background: #7a3b3b; color: #ffe8e8; }
.bar {
  height: 4px; background: #2a2e36; border-radius: 2px;
  margin: 6px 0 4px; overflow: hidden;
}
.bar i { display: block; height: 100%; background: #4a90d9; }
figcaption { font-size: 11px; color: #b6bcc4; margin-top: 4px; }
.label { color: #9ecbff; font-weight: 600; }
.fname {
  color: #767c85; font-size: 10px; margin-top: 3px;
  word-break: break-all; line-height: 1.3;
}
.reason { color: #c8cdd4; font-size: 11px; margin-top: 5px; font-style: italic; }
.missing {
  height: 190px; display: flex; align-items: center; justify-content: center;
  background: #101216; color: #767c85; border-radius: 4px; font-size: 11px;
  text-align: center; padding: 8px;
}
.hint {
  margin-top: 40px; padding-top: 16px; border-top: 1px solid #2a2e36;
  color: #9aa0a8; font-size: 12px;
}
.hint li { margin: 4px 0; }
"""


def build_html(
    payloads: List[Dict[str, Any]],
    labels: List[str],
    top: int,
    embed: bool,
    out_dir: Path,
) -> str:
    body = "\n".join(
        render_one(p, lab, top, embed, out_dir)
        for p, lab in zip(payloads, labels)
    )

    first = payloads[0] if payloads else {}
    title = first.get("query") or first.get("query_image") or "검색 결과"

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>{html.escape(str(title))}</title>
<style>{CSS}</style></head>
<body>
<div class="top">
  <h1>{html.escape(str(title))}</h1>
  <p class="meta">파란 테두리 = 쿼리 자신 · 막대는 상대 점수(범위 내 정규화)</p>
</div>
{body}
<div class="hint">
  <strong>무엇을 봐야 하는가</strong>
  <ul>
    <li>1위가 쿼리 자신인가 — DB 에 있는 crop 을 넣었다면 그래야 한다.
        아니면 색인/질의 전처리가 어긋난 것이다.</li>
    <li>점수 범위가 좁으면(예: 0.92~0.95) 그 순위는 노이즈일 수 있다.
        ReID 임베딩은 "둘 다 사람"이라는 공통 특징으로 기저 코사인이 높다.</li>
    <li>SOLIDER 순위와 qdrant 점수가 어긋나면 두 모델이 다른 판단을 한 것이다.
        어느 쪽이 맞는지는 이미지로만 판정할 수 있다.</li>
  </ul>
</div>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="검색 결과 JSON 을 HTML 갤러리로 변환"
    )
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="검색 결과 JSON (여러 개 주면 나란히 비교)")
    ap.add_argument("--out", default=None,
                    help="출력 HTML 경로 (기본 첫 입력과 같은 폴더의 results.html)")
    ap.add_argument("--top", type=int, default=20,
                    help="crop 당 표시할 후보 수")
    ap.add_argument("--no-embed", dest="embed", action="store_false", default=True,
                    help="이미지를 base64 로 넣지 않고 file:// 링크를 쓴다")
    ap.add_argument("--open", dest="do_open", action="store_true",
                    help="만든 뒤 브라우저로 열기")
    args = ap.parse_args()

    payloads: List[Dict[str, Any]] = []
    labels: List[str] = []

    for raw in args.inputs:
        p = Path(raw)
        if not p.is_file():
            print(f"입력 JSON 이 없습니다: {p}", file=sys.stderr)
            return 1
        try:
            payloads.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패 ({p}): {e}", file=sys.stderr)
            return 1
        labels.append(p.name)

    out = Path(args.out) if args.out else Path(args.inputs[0]).with_name("results.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = build_html(payloads, labels, args.top, args.embed, out.parent)
    out.write_text(doc, encoding="utf-8")

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"HTML 생성: {out.resolve()}  ({size_mb:.1f} MB)")

    if args.embed and size_mb > 50:
        print(
            "  파일이 큽니다. --no-embed 를 쓰면 이미지를 링크로 대체해 "
            "훨씬 작아집니다."
        )

    if args.do_open:
        webbrowser.open(out.resolve().as_uri())
        print("  브라우저로 열었습니다.")
    else:
        print(f"  브라우저로 열기: start {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())