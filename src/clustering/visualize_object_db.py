"""
visualize_object_db.py

forensic_object Qdrant 로컬 컬렉션을 읽어,
DB에 실제로 들어간 객체 crop 이미지를 HTML 썸네일 갤러리로 시각화한다.

기본 경로:
    Qdrant DB : <project_root>/data/qdrant_local
    collection: forensic_object
    output     : <project_root>/outputs/object_db_preview/index.html

실행 예:
    python .\src\clustering\visualize_object_db.py

옵션 예:
    python .\src\clustering\visualize_object_db.py --limit 500
    python .\src\clustering\visualize_object_db.py --label car
    python .\src\clustering\visualize_object_db.py --limit 0   # 전체
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from qdrant_client import QdrantClient
except ImportError as e:
    raise SystemExit("qdrant-client가 필요합니다: pip install qdrant-client") from e


THIS_FILE = Path(__file__).resolve()

# 이 파일을 TransReID/src/clustering/ 아래에 두는 기준
ROOT_DIR = THIS_FILE.parents[2]

DEFAULT_DB_PATH = ROOT_DIR / "data" / "qdrant_local"
DEFAULT_COLLECTION = "forensic_object"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "object_db_preview"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 프로젝트에 따라 payload key 이름이 다를 수 있으므로 폭넓게 탐색
PATH_KEYS = (
    "crop_path",
    "path",
    "image_path",
    "source_path",
    "file_path",
    "frame_path",
    "representative_crop",
    "representative_path",
)

LABEL_KEYS = (
    "label",
    "class_name",
    "class",
    "object_class",
    "category",
    "category_name",
)

VIDEO_KEYS = (
    "video",
    "video_name",
    "video_path",
    "source_video",
    "video_id",
)

FRAME_KEYS = (
    "frame",
    "frame_idx",
    "frame_id",
    "frame_index",
)


def h(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def first_value(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, "", []):
            return value
    return None


def resolve_image_path(payload: Dict[str, Any]) -> Tuple[Optional[Path], Optional[str]]:
    """
    payload에서 실제 crop 이미지 경로를 찾는다.
    절대경로/상대경로 둘 다 지원.
    """
    for key in PATH_KEYS:
        raw = payload.get(key)
        if not raw or not isinstance(raw, (str, os.PathLike)):
            continue

        p = Path(str(raw))

        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([
                ROOT_DIR / p,
                ROOT_DIR / "data" / p,
                Path.cwd() / p,
            ])

        for c in candidates:
            try:
                if c.is_file() and c.suffix.lower() in IMAGE_EXTS:
                    return c.resolve(), key
            except OSError:
                pass

    # payload 안 nested dict도 한 단계 탐색
    for parent_key, obj in payload.items():
        if not isinstance(obj, dict):
            continue
        for key in PATH_KEYS:
            raw = obj.get(key)
            if not raw or not isinstance(raw, (str, os.PathLike)):
                continue
            p = Path(str(raw))
            candidates = [p] if p.is_absolute() else [
                ROOT_DIR / p,
                ROOT_DIR / "data" / p,
                Path.cwd() / p,
            ]
            for c in candidates:
                try:
                    if c.is_file() and c.suffix.lower() in IMAGE_EXTS:
                        return c.resolve(), f"{parent_key}.{key}"
                except OSError:
                    pass

    return None, None


def get_label(payload: Dict[str, Any]) -> str:
    value = first_value(payload, LABEL_KEYS)
    return str(value) if value is not None else "unknown"


def get_vector_dims(point: Any) -> Dict[str, int]:
    """
    with_vectors=True로 읽은 Qdrant point에서 named vector 차원을 수집.
    """
    out: Dict[str, int] = {}
    vectors = getattr(point, "vector", None)

    if isinstance(vectors, dict):
        for name, vec in vectors.items():
            try:
                out[str(name)] = len(vec)
            except TypeError:
                pass
    elif vectors is not None:
        try:
            out["default"] = len(vectors)
        except TypeError:
            pass

    return out


def scroll_points(
    client: QdrantClient,
    collection: str,
    limit: int,
    page_size: int = 256,
) -> List[Any]:
    """
    limit=0이면 전체 collection.
    """
    result: List[Any] = []
    offset = None

    while True:
        remain = page_size if limit == 0 else min(page_size, limit - len(result))
        if remain <= 0:
            break

        points, next_offset = client.scroll(
            collection_name=collection,
            limit=remain,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        result.extend(points)

        if next_offset is None or not points:
            break

        offset = next_offset

    return result


def build_html(rows: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
    cards = []

    for row in rows:
        payload_json = json.dumps(
            row["payload"],
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        if row["thumb_rel"]:
            image_part = (
                f'<a href="{h(row["thumb_rel"])}" target="_blank">'
                f'<img src="{h(row["thumb_rel"])}" loading="lazy"></a>'
            )
        else:
            image_part = '<div class="missing">이미지 경로를 찾지 못함</div>'

        vec_text = ", ".join(
            f"{h(k)}={v}D" for k, v in row["vector_dims"].items()
        ) or "vector info 없음"

        cards.append(f"""
        <article class="card">
          <div class="image-wrap">{image_part}</div>
          <div class="meta">
            <div class="label">{h(row["label"])}</div>
            <div><b>Point ID:</b> {h(row["point_id"])}</div>
            <div><b>Vector:</b> {vec_text}</div>
            <div><b>Path key:</b> {h(row["path_key"] or "-")}</div>
            <div class="path"><b>Crop:</b> {h(row["source_path"] or "-")}</div>
            <details>
              <summary>payload 보기</summary>
              <pre>{h(payload_json)}</pre>
            </details>
          </div>
        </article>
        """)

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>forensic_object DB Preview</title>
<style>
body {{
  margin: 0;
  font-family: Arial, "Malgun Gothic", sans-serif;
  background: #f5f5f5;
  color: #222;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: white;
  border-bottom: 1px solid #ddd;
  padding: 16px 22px;
}}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
.summary {{ font-size: 14px; line-height: 1.6; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 14px;
  padding: 18px;
}}
.card {{
  background: white;
  border: 1px solid #ddd;
  border-radius: 10px;
  overflow: hidden;
}}
.image-wrap {{
  height: 220px;
  background: #e9e9e9;
  display: flex;
  align-items: center;
  justify-content: center;
}}
.image-wrap img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
}}
.missing {{
  color: #777;
  font-size: 13px;
}}
.meta {{
  padding: 12px;
  font-size: 12px;
  line-height: 1.55;
}}
.label {{
  font-size: 18px;
  font-weight: bold;
  margin-bottom: 7px;
}}
.path {{
  word-break: break-all;
}}
details {{
  margin-top: 8px;
}}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 260px;
  overflow: auto;
  background: #f6f6f6;
  padding: 8px;
  border-radius: 6px;
  font-size: 11px;
}}
</style>
</head>
<body>
<header>
  <h1>forensic_object DB Preview</h1>
  <div class="summary">
    Collection: <b>{h(stats["collection"])}</b><br>
    Qdrant path: {h(stats["db_path"])}<br>
    DB points: <b>{stats["db_count"]:,}</b> /
    Preview points: <b>{stats["preview_count"]:,}</b> /
    Image resolved: <b>{stats["resolved_count"]:,}</b> /
    Missing image: <b>{stats["missing_count"]:,}</b>
  </div>
</header>
<main class="grid">
{''.join(cards)}
</main>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="forensic_object Qdrant DB를 HTML 이미지 갤러리로 시각화"
    )
    ap.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Qdrant local path (default: {DEFAULT_DB_PATH})",
    )
    ap.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"collection name (default: {DEFAULT_COLLECTION})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"HTML/thumbnail output dir (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=300,
        help="preview할 최대 point 수. 0=전체 (default: 300)",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="특정 label만 표시. 예: --label car",
    )
    ap.add_argument(
        "--copy-images",
        action="store_true",
        help="이미지를 outputs 아래 thumbnails로 복사. 기본은 file URI 직접 참조.",
    )
    args = ap.parse_args()

    if args.limit < 0:
        raise ValueError("--limit은 0 이상이어야 합니다.")

    db_path = args.db_path.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FORENSIC OBJECT DB VISUALIZER")
    print(f"Qdrant     : {db_path}")
    print(f"Collection : {args.collection}")
    print(f"Output     : {out_dir}")
    print("=" * 72)

    if not db_path.exists():
        raise SystemExit(f"Qdrant DB 경로가 없습니다: {db_path}")

    client = QdrantClient(path=str(db_path))

    try:
        info = client.get_collection(args.collection)
        db_count = int(getattr(info, "points_count", 0) or 0)

        # label 필터는 payload key 구조가 프로젝트마다 다를 수 있어
        # 우선 point를 가져온 후 Python에서 필터링한다.
        points = scroll_points(
            client,
            args.collection,
            0 if args.label else args.limit,
        )

        if args.label:
            wanted = args.label.strip().lower()
            points = [
                p for p in points
                if get_label(dict(getattr(p, "payload", {}) or {})).lower() == wanted
            ]
            if args.limit:
                points = points[:args.limit]

        thumb_dir = out_dir / "thumbnails"
        if args.copy_images:
            thumb_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        resolved = 0
        missing = 0

        for idx, point in enumerate(points):
            payload = dict(getattr(point, "payload", {}) or {})
            image_path, path_key = resolve_image_path(payload)

            thumb_rel = None
            source_path = None

            if image_path is not None:
                resolved += 1
                source_path = str(image_path)

                if args.copy_images:
                    safe_id = str(getattr(point, "id", idx)).replace("/", "_")
                    dst = thumb_dir / f"{idx:06d}_{safe_id}{image_path.suffix.lower()}"
                    if not dst.exists():
                        shutil.copy2(image_path, dst)
                    thumb_rel = dst.relative_to(out_dir).as_posix()
                else:
                    # 브라우저에서 로컬 파일 직접 참조
                    thumb_rel = image_path.as_uri()
            else:
                missing += 1

            rows.append({
                "point_id": getattr(point, "id", ""),
                "payload": payload,
                "label": get_label(payload),
                "vector_dims": get_vector_dims(point),
                "path_key": path_key,
                "source_path": source_path,
                "thumb_rel": thumb_rel,
            })

        stats = {
            "collection": args.collection,
            "db_path": str(db_path),
            "db_count": db_count,
            "preview_count": len(rows),
            "resolved_count": resolved,
            "missing_count": missing,
        }

        html_text = build_html(rows, stats)
        html_path = out_dir / "index.html"
        html_path.write_text(html_text, encoding="utf-8")

        print()
        print(f"DB points       : {db_count:,}")
        print(f"Preview points  : {len(rows):,}")
        print(f"Image resolved  : {resolved:,}")
        print(f"Missing images  : {missing:,}")
        print(f"HTML            : {html_path}")
        print()
        print("브라우저 열기:")
        print(f'  Start-Process "{html_path}"')

    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
