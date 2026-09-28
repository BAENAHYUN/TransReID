from __future__ import annotations

import argparse
import html
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from report_common import (
    iter_assignments, is_noise, size_histogram, path_href, common_summary,
    file_info, warning, write_json, result_markers, load_json, cluster_details,
    table, histogram_html,
    DEFAULT_CONFIG_PATH, CONFIG_HELP, load_pipeline_settings, resolve, applied_config,
)

CROP_KEYS = ("crop_path", "selected_crop_path", "image_path", "path")
VIDEO_KEYS = ("video_path", "video", "source_path")


def chunks(seq: Sequence[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class QdrantHTTP:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 120):
        import requests
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.s.headers.update({"api-key": api_key})

    def post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        r = self.s.post(self.base_url + path, json=body, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"Qdrant POST failed: {r.status_code} {path}\n{r.text[:2000]}")
        return r.json()

    def retrieve_points(self, collection: str, ids: Sequence[Any], batch_size: int = 256):
        out: Dict[str, Dict[str, Any]] = {}
        for batch in chunks(list(ids), batch_size):
            data = self.post(
                f"/collections/{collection}/points",
                {"ids": list(batch), "with_payload": True, "with_vector": False},
            )
            for row in data.get("result") or []:
                out[str(row.get("id"))] = row.get("payload") or {}
        return out


def load_assignments(path: Path, errors=None):
    errors = errors if errors is not None else dict(count=0, first_line=None)
    return list(iter_assignments(path, errors))


def split_groups(rows):
    groups = defaultdict(list)
    noise = []
    for row in rows:
        cid = row.get("cluster_id")
        if is_noise(row):
            noise.append(row)
        else:
            groups[str(cid)].append(row)
    return dict(groups), noise


def choose_clusters(groups, top_n: int, medium_n: int, small_n: int, seed: int):
    rng = random.Random(seed)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    selected = []
    used = set()

    for cid, members in ordered[:top_n]:
        selected.append((cid, members, "largest"))
        used.add(cid)

    medium = [(c, m) for c, m in ordered if c not in used and 5 <= len(m) <= 100]
    if len(medium) > medium_n:
        medium = rng.sample(medium, medium_n)
    for cid, members in medium:
        selected.append((cid, members, "medium"))
        used.add(cid)

    small = [(c, m) for c, m in ordered if c not in used and 2 <= len(m) <= 4]
    if len(small) > small_n:
        small = rng.sample(small, small_n)
    for cid, members in small:
        selected.append((cid, members, "small"))

    return selected


def first_payload(payload, keys):
    for k in keys:
        v = payload.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def resolve_path(raw: Optional[str], project_root: Path) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p
    if not p.is_absolute():
        q = (project_root / p).resolve()
        if q.is_file():
            return q
    q = (project_root / raw.replace("\\", "/")).resolve()
    if q.is_file():
        return q
    return None


def inline_data_uri(src: Path, max_side: int, quality: int) -> Optional[str]:
    """crop 을 썸네일로 줄여 base64 data URI 로 만든다. 실패하면 None."""
    try:
        import base64
        import io
        from PIL import Image
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def safe_name(point_id: Any, suffix: str):
    clean = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(point_id))
    return clean[:140] + suffix


def esc(v: Any):
    return html.escape("" if v is None else str(v))


def build_parser():
    p = argparse.ArgumentParser(description="Build HTML image gallery from Leiden assignments + Qdrant payloads")
    p.add_argument("--assignments", default=r"outputs\clustering\leiden\person\person_leiden_assignments.jsonl")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--target", choices=['person', 'object'], default=None)
    p.add_argument("--collection", default=None, help=CONFIG_HELP)
    p.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    p.add_argument("--api-key", default=None)
    p.add_argument("--project-root", default=".")
    p.add_argument("--output-dir", default=r"outputs\clustering\leiden\person\gallery")
    p.add_argument("--top-clusters", type=int, default=20)
    p.add_argument("--medium-clusters", type=int, default=10)
    p.add_argument("--small-clusters", type=int, default=10)
    p.add_argument("--images-per-cluster", type=int, default=40)
    p.add_argument("--noise-samples", type=int, default=80)
    p.add_argument("--qdrant-batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--title", default="Leiden Person Cluster Gallery",
                   help="HTML 제목. object 갤러리는 따로 지정한다.")
    p.add_argument("--html-name", default="leiden_person_gallery.html",
                   help="출력 HTML 파일명. person/object 갤러리가 겹치지 않게 한다.")
    p.add_argument("--inline-images", action="store_true",
                   help="crop 을 assets/ 로 복사하지 않고 base64 썸네일로 HTML 에 내장한다 (자기완결 한 파일).")
    p.add_argument("--thumb-size", type=int, default=240,
                   help="--inline-images 썸네일 최대 변 길이(px)")
    p.add_argument("--thumb-quality", type=int, default=75)
    p.add_argument("--report", default=None, help="cluster report JSON; 기본 assignments 옆 자동 탐색")
    return p


def parse_args(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    for name, minimum in (("top_clusters", 0), ("medium_clusters", 0), ("small_clusters", 0),
                          ("noise_samples", 0), ("images_per_cluster", 1), ("thumb_size", 16),
                          ("qdrant_batch_size", 1)):
        if getattr(args, name) < minimum:
            p.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    if not 1 <= args.thumb_quality <= 95:
        p.error("--thumb-quality must be in 1..95")
    return args


def main(argv=None):
    """copied_images counts successfully shown images, including inline mode.

    External assets are refreshed by size/mtime only; their contents are not
    compared when both attributes match.
    """
    args = parse_args(argv)
    target = args.target or next((v for v in ('person', 'object')
                                 if Path(args.assignments).name.startswith(v + '_')), None)
    if args.collection in (None, '') and target is None:
        build_parser().error('--target 또는 --collection 지정')
    try:
        settings = load_pipeline_settings(args.config, require=not (
            args.collection not in (None, '') and args.qdrant_url not in (None, '')))
        args.collection = resolve(args.collection, settings.collection_for(target)
                                  if settings and target else None, 'collection')
        args.qdrant_url = resolve(args.qdrant_url, settings.qdrant_url if settings else None, 'qdrant_url')
    except (OSError, ValueError) as exc:
        build_parser().error(str(exc))

    assignments_path = Path(args.assignments).resolve()
    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    html_path = (out_dir / args.html_name).resolve()
    report_path = Path(args.report).resolve() if args.report else assignments_path.with_name(f"{target}_leiden_report.json") if target else None
    warnings = []
    if settings is None:
        warnings.append(warning('CONFIG_UNAVAILABLE', f'config 없음: {args.config}; 명시 CLI 사용'))
    cluster_report = load_json(report_path, warnings, target) if report_path else None
    if report_path is None:
        warnings.append(warning("REPORT_UNKNOWN", "cluster report 미기록", target))
    config = (cluster_report or {}).get("config") or {}
    cluster_stats = (cluster_report or {}).get("stats") or {}
    inputs = [file_info("assignments", assignments_path)]
    if settings:
        inputs.insert(0, file_info('config', settings.config_path))
    if report_path:
        inputs.append(file_info("report", report_path))
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.inline_images:
        assets_dir.mkdir(parents=True, exist_ok=True)

    parse_errors = dict(count=0, first_line=None)
    rows = load_assignments(assignments_path, parse_errors)
    if parse_errors["count"]:
        warnings.append(warning("PARSE_ERRORS", f"깨진 assignments 줄: {parse_errors}", target))
    if not rows:
        # 예전에는 예외로 멈췄다. 이제는 빈 갤러리를 만들되 sidecar/GUI 에 경고로 드러낸다.
        print(f"[경고] 유효한 assignments 가 없습니다: {assignments_path}")
        warnings.append(warning("NO_ASSIGNMENTS", f"유효한 assignments 없음: {assignments_path}", target))
    groups, noise = split_groups(rows)
    selected = choose_clusters(groups, args.top_clusters, args.medium_clusters, args.small_clusters, args.seed)

    rng = random.Random(args.seed)
    selected_rows = []
    render_plan = []

    for rank, (cid, members, kind) in enumerate(selected, 1):
        if len(members) > args.images_per_cluster:
            head = members[:min(4, args.images_per_cluster)]
            rest = members[len(head):]
            need = args.images_per_cluster - len(head)
            show = head + (rng.sample(rest, need) if need > 0 and len(rest) > need else rest[:need])
        else:
            show = list(members)
        render_plan.append((rank, cid, kind, members, show))
        selected_rows.extend(show)

    noise_show = list(noise)
    if len(noise_show) > args.noise_samples:
        noise_show = rng.sample(noise_show, args.noise_samples)
    selected_rows.extend(noise_show)

    ids = []
    seen = set()
    for row in selected_rows:
        sid = str(row["point_id"])
        if sid not in seen:
            seen.add(sid)
            ids.append(row["point_id"])

    print("=" * 88)
    print("LEIDEN VISUAL GALLERY")
    print("=" * 88)
    print(f"assignments      : {len(rows):,}")
    print(f"clusters         : {len(groups):,}")
    print(f"noise            : {len(noise):,}")
    print(f"selected clusters: {len(selected):,}")
    print(f"Qdrant fetch IDs : {len(ids):,}")

    q = QdrantHTTP(args.qdrant_url, args.api_key)
    payloads = q.retrieve_points(args.collection, ids, args.qdrant_batch_size)
    print(f"payloads fetched : {len(payloads):,}")

    copied = 0
    missing = 0
    sections = []

    def render_section(title, subtitle, all_members, show_rows, slug):
        nonlocal copied, missing
        group_dir = assets_dir / slug
        if not args.inline_images:
            group_dir.mkdir(parents=True, exist_ok=True)
        cards = []
        for row in show_rows:
            pid = row["point_id"]
            payload = payloads.get(str(pid), {})
            raw_crop = first_payload(payload, CROP_KEYS)
            src = resolve_path(raw_crop, project_root)
            if src is None:
                missing += 1
                cards.append(f'''<div class="thumb missing"><div class="missingbox">이미지 없음</div><div class="meta"><b>point</b> {esc(pid)}</div><div class="path">{esc(raw_crop or "crop_path 없음")}</div></div>''')
                continue

            if args.inline_images:
                rel = inline_data_uri(src, args.thumb_size, args.thumb_quality)
                if rel is None:
                    missing += 1
                    cards.append(f'''<div class="thumb missing"><div class="missingbox">디코드 실패</div><div class="meta"><b>point</b> {esc(pid)}</div><div class="path">{esc(raw_crop)}</div></div>''')
                    continue
                # data URI 를 href 에도 넣으면 HTML 크기가 두 배가 된다 — 링크 생략
                link_open, link_close = "", ""
            else:
                suffix = src.suffix.lower() if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"
                dst = group_dir / safe_name(pid, suffix)
                if (not dst.exists() or dst.stat().st_size != src.stat().st_size
                        or dst.stat().st_mtime_ns != src.stat().st_mtime_ns):
                    shutil.copy2(src, dst)
                rel = path_href(dst, html_path)
                link_open, link_close = f'<a href="{rel}" target="_blank">', "</a>"
            copied += 1

            video = first_payload(payload, VIDEO_KEYS)
            frame = payload.get("frame_idx", payload.get("frame"))
            track = payload.get("long_track_id") if payload.get("long_track_id") is not None else payload.get("track_id")
            canonical = payload.get("canonical_person_id")
            label = payload.get("label", payload.get("class_name", ""))
            fields = {"point": pid, "label": label, "image_id": payload.get("image_id"),
                      "detection_id": payload.get("detection_id"), "track": track,
                      "canonical": canonical, "frame": frame}
            if payload.get("score") is not None:
                try:
                    fields["score"] = f'{float(payload["score"]):.3f}'
                except (ValueError, TypeError):
                    fields["score"] = payload["score"]
            bbox = payload.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    fields["bbox"] = ",".join(str(round(float(v))) for v in bbox)
                except (ValueError, TypeError):
                    fields["bbox"] = str(bbox)
            details = ''.join(f'<div class="meta"><b>{esc(k)}</b> {esc(v)}</div>' for k, v in fields.items() if v is not None)
            details += f'<div class="path" title="{esc(raw_crop)}"><b>crop</b> {esc(Path(raw_crop.replace(chr(92), "/")).name)}</div>'
            if video is not None:
                details += f'<div class="path" title="{esc(video)}"><b>video</b> {esc(Path(video.replace(chr(92), "/")).name)}</div>'
            cards.append(f'<div class="thumb">{link_open}<img src="{rel}" loading="lazy">{link_close}{details}</div>')

        return f'''<section class="cluster"><div class="cluster-head"><div><h2>{esc(title)}</h2><div class="sub">{esc(subtitle)}</div></div><div class="count">{len(all_members):,} points</div></div><div class="gallery">{"".join(cards)}</div></section>'''

    for rank, cid, kind, members, show in render_plan:
        sections.append(render_section(f"Cluster {rank} · {kind}", cid, members, show, f"cluster_{rank:03d}"))

    if noise_show:
        sections.append(render_section("Noise sample", f"전체 noise {len(noise):,}개 중 샘플", noise, noise_show, "noise"))

    html_text = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(args.title)}</title><style>
:root{{--bg:#0b1020;--panel:#141b2d;--panel2:#19233b;--line:#2b3758;--text:#eef3ff;--muted:#9cacc9;--accent:#7aa2ff;--good:#5bd5a6}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}.wrap{{max-width:1600px;margin:auto;padding:26px}}h1{{margin:0 0 6px;font-size:30px}}.topsub{{color:var(--muted);margin-bottom:20px}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:22px}}.stat{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.stat .k{{color:var(--muted);font-size:12px}}.stat .v{{font-size:25px;font-weight:700;margin-top:6px}}.cluster{{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin:18px 0;padding:18px}}.cluster-head{{display:flex;justify-content:space-between;gap:15px;align-items:center;margin-bottom:13px}}.cluster h2{{margin:0;font-size:20px}}.sub{{color:var(--muted);font-size:12px;margin-top:4px;word-break:break-all}}.count{{font-size:15px;font-weight:700;color:var(--good);white-space:nowrap}}.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:11px}}.thumb{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px;min-width:0}}.thumb img{{width:100%;height:190px;object-fit:contain;background:#090d16;border-radius:7px;display:block}}.meta{{font-size:11px;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.path{{font-size:10px;color:var(--muted);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.missingbox{{height:190px;display:flex;align-items:center;justify-content:center;border-radius:7px;background:#351d27;color:#ffadb7;font-weight:700}}.note{{background:var(--panel);border-left:4px solid var(--accent);border-radius:10px;padding:13px 15px;color:#dbe5fa;margin-top:18px}}@media(max-width:800px){{.stats{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="wrap"><h1>{esc(args.title)}</h1><div class="topsub">collection={esc(args.collection)} · visual QA</div><div class="stats"><div class="stat"><div class="k">Assignments</div><div class="v">{len(rows):,}</div></div><div class="stat"><div class="k">Total clusters</div><div class="v">{len(groups):,}</div></div><div class="stat"><div class="k">Noise</div><div class="v">{len(noise):,}</div></div><div class="stat"><div class="k">Copied images</div><div class="v">{copied:,}</div></div></div><div class="note">largest / medium / small cluster와 noise 샘플을 함께 보여줍니다. 같은 cluster 안에 다른 사람이 섞이는지 먼저 확인하세요.</div>{"".join(sections)}</div></body></html>'''

    noise_ratio = len(noise) / len(rows) if rows else None
    largest = max((len(members) for members in groups.values()), default=0)
    hist = size_histogram(len(members) for members in groups.values())
    coverage = dict(selected_clusters=len(selected), total_clusters=len(groups),
                    sampled_points=len(selected_rows), shown_points=copied, total_points=len(rows))
    details = ('<section class="cluster"><h2>클러스터 설정 · 통계</h2>' + table(cluster_details(cluster_report))
               + table({"noise 비율": noise_ratio, "최종 최대 클러스터 크기 (assignments)": largest,
                        "누락 수": missing, "깨진 줄": parse_errors})
               + '<h3>표본 커버리지</h3>' + table(coverage)
               + '<h3>크기 히스토그램 (noise 제외)</h3>' + histogram_html(hist) + '</section>')
    html_text = html_text.replace('Copied images', '표시 이미지').replace('<div class="stats">', details + '<div class="stats">', 1)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8", newline="\n")
    if missing:
        print(f"[경고] 선택 표본 이미지 누락: {missing}")
        warnings.append(warning("MISSING_IMAGES", f"선택 표본 이미지 누락: {missing}", target))
    report = {
        **common_summary("build_leiden_gallery.py", html_path, inputs, warnings),
        "assignments": len(rows),
        "clusters": len(groups),
        "noise": len(noise),
        "selected_clusters": len(selected),
        "copied_images": copied,
        "missing_images": missing,
        "html": str(html_path),
        "target": target,
        "collection": args.collection,
        "config": applied_config(settings, collection=args.collection, qdrant_url=args.qdrant_url, target=target),
        "source": config.get("sources"),
        "report_path": str(report_path) if report_path else None,
        "assignments_path": str(assignments_path),
        "valid_assignments": len(rows),
        "parse_errors": parse_errors,
        "noise_ratio": noise_ratio,
        "largest_cluster": largest,
        "size_histogram": hist,
        "coverage": coverage,
        "dry_run": (cluster_report or {}).get("dry_run"),
        "max_cluster_size": config.get("max_cluster_size"),
        "oversized_remaining": cluster_stats.get("oversized_remaining"),
    }
    write_json(out_dir / "gallery_report.json", report)

    print("=" * 88)
    print("GALLERY COMPLETE")
    print("=" * 88)
    print(f"HTML          : {html_path}")
    print(f"copied images : {copied:,}")
    print(f"missing images: {missing:,}")
    print(f"report        : {out_dir / 'gallery_report.json'}")
    result_markers(out_dir / "gallery_report.json", html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
