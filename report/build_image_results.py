#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""이미지 파이프라인 '결과창' — DB 리포트 · 클러스터 갤러리 · 결과 인덱스를 한 번에 만든다.

GUI 이미지 파이프라인의 핵심 4단계(검출 → 임베딩 → 클러스터 → 결과창) 중 마지막 단계.
개별 단계(3. DB HTML, 5/6. 갤러리, 7. 인덱스)를 순서대로 부르고 마지막에 인덱스 HTML 을
RESULT_HTML 마커로 알린다 (GUI 의 '결과 열기' 버튼이 이 마커를 읽는다).

  1) report/build_image_db_html.py      → <out-dir>/image_db_<source>.html
  2) report/build_leiden_gallery.py     → <cluster-dir>/<target>/gallery/leiden_<target>_gallery.html
                                           (assignments 가 있는 target 만; 없으면 건너뜀)
  3) report/build_image_review_index.py → <out-dir>/index_<source>.html   ← RESULT_HTML

자식 스크립트가 찍는 RESULT_* 마커는 '(단계) …' 로 바꿔 흘려보내 GUI 가 마지막(인덱스) 것만 보게 한다.
--dry-run 은 실행할 명령만 출력한다 (테스트·확인용).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
MARKER_RE = re.compile(r"^\s*RESULT_(HTML|SUMMARY):")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("_") or "all"


def plan(args: argparse.Namespace) -> List[List[str]]:
    """실행할 명령 목록 [(설명 포함 argv)]. 각 항목의 첫 원소는 단계 이름(로그용), 나머지가 argv."""
    py = args.python or sys.executable
    out_dir = Path(args.out_dir)
    cluster_dir = Path(args.cluster_dir)
    tag = slug(args.source)
    db_html = out_dir / f"image_db_{tag}.html"
    cmds: List[List[str]] = []

    common_q = ["--qdrant-url", args.qdrant_url] if args.qdrant_url else []

    if args.reuse and db_html.is_file():
        cmds.append(["skip:DB 리포트", f"이미 있음 → 재사용: {db_html}"])
    else:
        cmd = [py, "-u", str(ROOT / "report" / "build_image_db_html.py"), "--config", args.config, "--samples", str(args.samples),
               "--thumb-size", str(args.thumb_size), "--out", str(db_html), "--manifest-dir", args.manifest_dir] + common_q
        if args.source:
            cmd += ["--source", args.source]
        cmds.append(["DB 리포트"] + cmd)

    for target in ("person", "object"):
        assignments = cluster_dir / target / f"{target}_leiden_assignments.jsonl"
        gallery_dir = cluster_dir / target / "gallery"
        html_name = f"leiden_{target}_gallery.html"
        if not assignments.is_file():
            cmds.append([f"skip:{target} 갤러리", f"assignments 없음 → 건너뜀: {assignments} (3. 클러스터 단계를 먼저 실행)"])
            continue
        if args.reuse and (gallery_dir / html_name).is_file():
            cmds.append([f"skip:{target} 갤러리", f"이미 있음 → 재사용: {gallery_dir / html_name}"])
            continue
        cmd = [py, "-u", str(ROOT / "report" / "build_leiden_gallery.py"), "--assignments", str(assignments), "--config", args.config,
               "--target", target, "--output-dir", str(gallery_dir), "--title", f"{target} 클러스터 갤러리 ({args.source or 'all'})",
               "--html-name", html_name, "--inline-images", "--thumb-size", "240", "--top-clusters", str(args.top_clusters),
               "--medium-clusters", str(args.medium_clusters), "--small-clusters", str(args.small_clusters),
               "--images-per-cluster", str(args.images_per_cluster), "--noise-samples", str(args.noise_samples), "--seed", "42"] + common_q
        cmds.append([f"{target} 갤러리"] + cmd)

    index = out_dir / f"index_{tag}.html"
    cmd = [py, "-u", str(ROOT / "report" / "build_image_review_index.py"), "--db-html", str(db_html), "--cluster-dir", str(cluster_dir),
           "--manifest-dir", args.manifest_dir, "--title", args.title, "--out", str(index)]
    cmds.append(["결과 인덱스"] + cmd)
    return cmds


def run_cmd(name: str, argv: List[str]) -> int:
    """자식 스크립트를 돌리며 출력을 그대로 흘리되 RESULT_* 마커는 접두어를 붙여 GUI 마커 해석에서 뺀다."""
    print(f"\n[결과창] {name}: {' '.join(argv)}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if MARKER_RE.match(line):
            line = f"  ({name}) {line.strip()}"
        print(line, flush=True)
    code = proc.wait()
    print(f"[결과창] {name}: {'완료' if code == 0 else f'실패 (exit {code})'} · {time.time() - t0:.1f}s", flush=True)
    return code


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pipeline.yaml")
    p.add_argument("--source", default="prw_image", help="DB 리포트 source 필터 (검출 단계의 --source 와 같은 값)")
    p.add_argument("--cluster-dir", default="outputs/clustering/leiden_image_prw", help="3. 클러스터 단계의 출력 폴더")
    p.add_argument("--out-dir", default="outputs/image_review")
    p.add_argument("--title", default="이미지 DB · 클러스터 결과")
    p.add_argument("--samples", type=int, default=2000, help="DB 리포트 썸네일 표본 수")
    p.add_argument("--thumb-size", type=int, default=160)
    p.add_argument("--top-clusters", type=int, default=20)
    p.add_argument("--medium-clusters", type=int, default=10)
    p.add_argument("--small-clusters", type=int, default=10)
    p.add_argument("--images-per-cluster", type=int, default=40)
    p.add_argument("--noise-samples", type=int, default=80)
    p.add_argument("--manifest-dir", default="data/build_manifests")
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--reuse", action="store_true", help="이미 있는 DB 리포트·갤러리는 다시 만들지 않는다")
    p.add_argument("--python", default=None)
    p.add_argument("--dry-run", action="store_true", help="명령만 출력")
    args = p.parse_args(argv)

    cmds = plan(args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for item in cmds:
        name, rest = item[0], item[1:]
        if name.startswith("skip:"):
            print(f"[결과창] {name[5:]}: {rest[0]}", flush=True)
            continue
        if args.dry_run:
            print(f"[결과창] {name}: {' '.join(rest)}", flush=True)
            continue
        code = run_cmd(name, rest)
        if code != 0:
            print(f"[결과창] 중단: '{name}' 실패. 위 로그를 확인하세요.", flush=True)
            return code
    index = Path(cmds[-1][-1])
    if args.dry_run:
        print(f"[결과창] (dry-run) 인덱스: {index}")
        return 0
    if not index.is_file():
        print(f"[결과창] 인덱스 HTML 이 만들어지지 않았습니다: {index}", flush=True)
        return 1
    print(f"RESULT_HTML: {index.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
