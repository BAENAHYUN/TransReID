"""
health_check.py — person_db 무결성 + 검색 기능 일괄 점검

적재가 끝났다는 것과 검색이 된다는 것은 다르고, 검색이 된다는 것과 모든 경로가
된다는 것도 다르다. 이 스크립트는 DB 상태와 검색 경로를 한 번에 훑고
PASS / WARN / FAIL 로 정리한다.

    python health_check.py                  # DB + 검색 (GPU 불필요)
    python health_check.py --text           # 자연어 검색까지 (GPU 사용)
    python health_check.py --json-out hc.json


GPU 없이 검색을 어떻게 검사하는가
------------------------------
질의 벡터를 새로 만들지 않고 **DB 에 이미 있는 벡터를 꺼내 쓴다**. 이러면
임베더를 올리지 않고도 fused_search / RRF / scope 라우팅 / 필터 / 인덱스를
전부 통과시킬 수 있다. 자기 자신이 1위로 돌아오면 그 경로는 살아 있는 것이다.

임베더 자체(색인/질의 전처리 일치)는 verify_video.py --mode reembed 가 본다.
텍스트 인코더는 저장된 벡터로 대신할 수 없으므로 --text 에서만 검사한다.


검사 항목
--------
  DB   컬렉션 상태 · source 합계 · is_person 분포 · 벡터 커버리지
       payload 인덱스 · 라벨 표기 일관성 · bbox_space · 동영상 track_key
  검색 이미지 사람 / 동영상 사람 / 이미지 객체 / 동영상 객체
       source 필터 · 소스 교차 · (옵션) 자연어
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import PipelineConfig
from qdrant_store import QdrantStore

CONFIG_PATH = ROOT / "pipeline.yaml"

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
MARK = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "----"}


class Report:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, section: str, name: str, status: str, detail: str = "") -> None:
        self.rows.append({"section": section, "name": name,
                          "status": status, "detail": detail})
        print(f"  [{MARK[status]}] {name}" + (f"  —  {detail}" if detail else ""))

    def counts(self) -> Dict[str, int]:
        c = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for r in self.rows:
            c[r["status"]] += 1
        return c


def head(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


# =============================================================================
# DB 점검
# =============================================================================

def check_db(store: QdrantStore, cfg: PipelineConfig, rep: Report) -> Dict[str, Any]:
    from qdrant_client import models

    client = store.client
    coll = cfg.collection
    facts: Dict[str, Any] = {}

    head("1. 컬렉션 상태")

    info = client.get_collection(coll)
    total = info.points_count
    facts["total"] = total
    facts["status"] = str(info.status)

    st = str(info.status).lower()
    rep.add("db", f"status = {info.status}", PASS if st.endswith("green") else WARN,
            "" if st.endswith("green") else "인덱싱 중 — 검색이 느릴 수 있음")

    vecs = info.config.params.vectors
    dims = {n: p.size for n, p in vecs.items()} if isinstance(vecs, dict) else {}
    facts["dims"] = dims

    want = {n: int(s.dim) for n, s in cfg.retrievers.items()}
    if dims == want:
        rep.add("db", "벡터 차원", PASS, str(dims))
    else:
        rep.add("db", "벡터 차원", FAIL,
                f"pipeline.yaml={want} / 실제={dims}")

    # ── source 분포 ──
    head("2. source 분포")

    found = set()
    offset = None
    scanned = 0
    while scanned < 30000:
        pts, offset = client.scroll(
            collection_name=coll, limit=3000, offset=offset,
            with_payload=["source"], with_vectors=False)
        if not pts:
            break
        for p in pts:
            v = (p.payload or {}).get("source")
            if v:
                found.add(str(v))
        scanned += len(pts)
        if offset is None:
            break

    def cnt(f) -> int:
        return client.count(coll, count_filter=f, exact=True).count

    def eq(key, val):
        return models.Filter(must=[models.FieldCondition(
            key=key, match=models.MatchValue(value=val))])

    by_source: Dict[str, int] = {}
    for name in sorted(found):
        by_source[name] = cnt(eq("source", name))
    facts["by_source"] = by_source

    empty = cnt(models.Filter(must=[models.IsEmptyCondition(
        is_empty=models.PayloadField(key="source"))]))
    facts["source_missing"] = empty

    print()
    for k, v in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"      {k:<24} {v:>9,}")
    if empty:
        print(f"      {'(source 없음)':<24} {empty:>9,}")
    print(f"      {'-'*24} {'-'*9}")
    print(f"      {'합계':<24} {sum(by_source.values()) + empty:>9,}")
    print(f"      {'total':<24} {total:>9,}")
    print()

    s = sum(by_source.values()) + empty
    if empty:
        rep.add("db", "source 태깅", FAIL, f"{empty:,}건 미태깅")
    elif s == total:
        rep.add("db", "source 합계 = total", PASS, f"{total:,}")
    else:
        rep.add("db", "source 합계 = total", FAIL,
                f"합계 {s:,} != total {total:,} (차이 {total-s:,})")

    # ── is_person ──
    n_person = cnt(eq("is_person", True))
    n_object = cnt(eq("is_person", False))
    facts["person"] = n_person
    facts["object"] = n_object

    if n_person + n_object == total:
        rep.add("db", "is_person 합계", PASS,
                f"person {n_person:,} / object {n_object:,}")
    else:
        rep.add("db", "is_person 합계", FAIL,
                f"{n_person:,} + {n_object:,} != {total:,}")

    # ── 벡터 커버리지 ──
    head("3. 벡터 커버리지")

    expect = {}
    for name, spec in cfg.retrievers.items():
        sc = str(spec.scope).lower()
        expect[name] = total if sc == "all" else (n_person if sc == "person" else n_object)

    got: Dict[str, Optional[int]] = {}
    for name in cfg.retrievers:
        try:
            got[name] = cnt(models.Filter(must=[
                models.HasVectorCondition(has_vector=name)]))
        except Exception:  # noqa: BLE001
            got[name] = None

    facts["vector_counts"] = got
    facts["vector_expected"] = expect

    print()
    for name in cfg.retrievers:
        sc = str(cfg.retrievers[name].scope)
        g = got[name]
        print(f"      {name:<10} scope={sc:<7} 기대 {expect[name]:>9,}  "
              f"실제 {('%9s' % '조회불가') if g is None else f'{g:>9,}'}")
    print()

    if all(v is None for v in got.values()):
        rep.add("db", "벡터 커버리지", SKIP,
                "HasVectorCondition 미지원 (Qdrant 버전). 검색 테스트로 대체")
    else:
        bad = [n for n, g in got.items() if g is not None and g != expect[n]]
        if not bad:
            rep.add("db", "벡터 커버리지", PASS, "scope 대로 채워져 있음")
        else:
            det = "; ".join(f"{n}: 기대 {expect[n]:,} 실제 {got[n]:,}" for n in bad)
            rep.add("db", "벡터 커버리지", WARN, det)

    # ── payload 인덱스 ──
    head("4. payload 인덱스 · 데이터 일관성")

    idx = set((getattr(info, "payload_schema", None) or {}).keys())
    facts["indexes"] = sorted(idx)
    need = {"is_person", "label", "image_id", "frame_idx",
            "source", "video", "track_key"}
    missing = sorted(need - idx)
    if not missing:
        rep.add("db", "payload 인덱스", PASS, f"{len(idx)}개")
    else:
        rep.add("db", "payload 인덱스", WARN,
                f"없음: {missing} — 이 필드 필터는 전수 스캔이 된다")

    # ── 라벨 표기 일관성 ──
    # COCO 는 'potted plant'(공백), 동영상 객체 파일명은 'potted_plant'(밑줄).
    # 둘 다 들어가 있으면 label 필터가 절반만 잡는다.
    try:
        from rfdetr_adapter import COCO80_NAMES
        multi = [n for n in COCO80_NAMES if " " in n]
        split_labels = []
        for name in multi:
            u = name.replace(" ", "_")
            a = cnt(eq("label", name))
            b = cnt(eq("label", u))
            if a and b:
                split_labels.append(f"{name}({a:,}) vs {u}({b:,})")
            elif b and not a:
                split_labels.append(f"{u}({b:,}) 만 있음 — 공백 표기가 없다")
        facts["label_split"] = split_labels
        if not split_labels:
            rep.add("db", "라벨 표기 일관성", PASS, "공백/밑줄 혼재 없음")
        else:
            rep.add("db", "라벨 표기 일관성", FAIL,
                    "; ".join(split_labels[:4]))
    except Exception as e:  # noqa: BLE001
        rep.add("db", "라벨 표기 일관성", SKIP, str(e))

    # ── bbox_space ──
    n_crop = cnt(eq("bbox_space", "crop"))
    n_img = cnt(eq("bbox_space", "image"))
    facts["bbox_space"] = {"crop": n_crop, "image": n_img,
                           "none": total - n_crop - n_img}
    rep.add("db", "bbox_space 표기", PASS if n_crop else WARN,
            f"crop {n_crop:,} / image {n_img:,} / 없음 {total-n_crop-n_img:,}")

    # ── track_key ──
    n_tk = total - cnt(models.Filter(must=[models.IsEmptyCondition(
        is_empty=models.PayloadField(key="track_key"))]))
    facts["track_key"] = n_tk
    rep.add("db", "track_key 보유", PASS if n_tk else WARN,
            f"{n_tk:,}건 (동영상 사람 크롭)")

    return facts


# =============================================================================
# 검색 점검
# =============================================================================

def pick_point(store, cfg, rep, label: str, must: list) -> Optional[dict]:
    """조건에 맞는 point 하나를 골라 벡터까지 가져온다."""
    from qdrant_client import models
    try:
        pts, _ = store.client.scroll(
            collection_name=cfg.collection,
            scroll_filter=models.Filter(must=must),
            limit=1, with_payload=True, with_vectors=True)
    except Exception as e:  # noqa: BLE001
        rep.add("search", f"{label} 표본 확보", FAIL, f"{type(e).__name__}: {e}")
        return None
    if not pts:
        rep.add("search", f"{label} 표본 확보", SKIP, "해당 point 없음")
        return None
    p = pts[0]
    if not isinstance(p.vector, dict):
        rep.add("search", f"{label} 표본 확보", FAIL, "named vector 아님")
        return None
    return {"id": p.id, "payload": p.payload or {}, "vector": p.vector}


def run_search(
    store, cfg, rep, name: str, sample: dict,
    person_only: Optional[bool], topk: int, extra_filter=None,
    expect_source: Optional[str] = None,
) -> Optional[dict]:
    """저장된 벡터로 검색하고 자기 자신이 1위인지, 결과 성질이 맞는지 본다."""
    try:
        t0 = time.time()
        hits = store.fused_search(
            query_vectors=sample["vector"],
            person_only=person_only,
            limit=topk,
            extra_filter=extra_filter,
        )
        dt = time.time() - t0
    except Exception as e:  # noqa: BLE001
        rep.add("search", name, FAIL, f"{type(e).__name__}: {e}")
        return None

    if not hits:
        rep.add("search", name, FAIL, "결과 0건")
        return None

    self_rank = next((i for i, h in enumerate(hits, 1)
                      if str(h.id) == str(sample["id"])), None)
    top = hits[0]
    tp = top.payload or {}

    # 성질 검사
    problems = []
    if person_only is True:
        bad = sum(1 for h in hits if (h.payload or {}).get("is_person") is not True)
        if bad:
            problems.append(f"person_only 인데 객체 {bad}건")
    elif person_only is False:
        bad = sum(1 for h in hits if (h.payload or {}).get("is_person") is not False)
        if bad:
            problems.append(f"object 인데 사람 {bad}건")
    if expect_source:
        bad = sum(1 for h in hits if (h.payload or {}).get("source") != expect_source)
        if bad:
            problems.append(f"source 필터인데 다른 소스 {bad}건")

    srcs = {}
    for h in hits:
        s = (h.payload or {}).get("source", "?")
        srcs[s] = srcs.get(s, 0) + 1

    detail = (f"자기 {self_rank or '없음'}위 · 1위 {float(top.score):.4f} · "
              f"{dt*1000:.0f}ms · {len(hits)}건")

    if problems:
        rep.add("search", name, FAIL, detail + " · " + "; ".join(problems))
    elif self_rank == 1:
        rep.add("search", name, PASS, detail)
    elif self_rank:
        rep.add("search", name, WARN, detail + " · 자기 자신이 1위가 아님")
    else:
        rep.add("search", name, FAIL, detail + " · 자기 자신이 결과에 없음")

    return {"name": name, "self_rank": self_rank,
            "top_score": float(top.score), "ms": round(dt * 1000),
            "sources": srcs,
            "top_label": tp.get("label"), "top_source": tp.get("source")}


def check_search(store, cfg, rep, facts: dict, topk: int) -> List[dict]:
    from qdrant_client import models

    head("5. 검색 기능")

    def eq(key, val):
        return models.FieldCondition(key=key, match=models.MatchValue(value=val))

    by_source = facts.get("by_source", {})
    img_sources = [s for s in by_source if s in ("COCO", "OpenImages-v7")]
    vid_person = [s for s in by_source
                  if s.startswith(("UCF", "SCVD")) and not s.endswith("-obj")]
    vid_object = [s for s in by_source if s.endswith("-obj")]

    out: List[dict] = []
    print()

    # T1 이미지 사람
    if img_sources:
        s = pick_point(store, cfg, rep, "이미지 사람",
                       [eq("source", img_sources[0]), eq("is_person", True)])
        if s:
            out.append(run_search(store, cfg, rep,
                                  f"이미지 사람 검색 ({img_sources[0]})",
                                  s, True, topk))

    # T2 동영상 사람
    if vid_person:
        s = pick_point(store, cfg, rep, "동영상 사람",
                       [eq("source", vid_person[0]), eq("is_person", True)])
        if s:
            out.append(run_search(store, cfg, rep,
                                  f"동영상 사람 검색 ({vid_person[0]})",
                                  s, True, topk))

    # T3 이미지 객체
    if img_sources:
        s = pick_point(store, cfg, rep, "이미지 객체",
                       [eq("source", img_sources[0]), eq("is_person", False)])
        if s:
            out.append(run_search(store, cfg, rep,
                                  f"이미지 객체 검색 ({img_sources[0]})",
                                  s, False, topk))

    # T4 동영상 객체 (새로 들어온 것)
    if vid_object:
        s = pick_point(store, cfg, rep, "동영상 객체",
                       [eq("source", vid_object[0]), eq("is_person", False)])
        if s:
            out.append(run_search(store, cfg, rep,
                                  f"동영상 객체 검색 ({vid_object[0]})",
                                  s, False, topk))
    else:
        rep.add("search", "동영상 객체 검색", SKIP,
                "-obj source 없음 — 동영상 객체가 아직 적재되지 않았다")

    # T5 source 필터
    if vid_person:
        s = pick_point(store, cfg, rep, "source 필터",
                       [eq("source", vid_person[0]), eq("is_person", True)])
        if s:
            out.append(run_search(
                store, cfg, rep, f"source 필터 ({vid_person[0]} 한정)",
                s, True, topk,
                extra_filter=models.Filter(must=[eq("source", vid_person[0])]),
                expect_source=vid_person[0]))

    # T6 소스 교차 — 통합 DB 가 실제로 가로지르는지
    if vid_person and img_sources:
        s = pick_point(store, cfg, rep, "교차",
                       [eq("source", vid_person[0]), eq("is_person", True)])
        if s:
            r = run_search(store, cfg, rep,
                           "소스 교차 (동영상 질의 → 전체)", s, True, 50)
            if r:
                n_img = sum(v for k, v in r["sources"].items() if k in img_sources)
                rep.add("search", "  └ 이미지 소스 도달",
                        PASS if n_img else WARN,
                        f"top-50 중 이미지 {n_img}건 · {r['sources']}")
                out.append(r)

    return out


def check_text_search(store, cfg, rep, topk: int) -> Optional[dict]:
    head("6. 자연어 검색 (GPU)")
    print()
    try:
        from registry import EmbedderRegistry
        reg = EmbedderRegistry(cfg)
    except Exception as e:  # noqa: BLE001
        rep.add("text", "임베더 로드", FAIL, f"{type(e).__name__}: {e}")
        return None

    query = "a man wearing a blue shirt walking outdoors"
    qv: Dict[str, Any] = {}

    for name in ("irra", "siglip2"):
        if name not in cfg.retrievers:
            continue
        try:
            emb = reg.get(name) if hasattr(reg, "get") else reg[name]
            fn = None
            for cand in ("encode_text", "embed_text", "text"):
                if hasattr(emb, cand):
                    fn = getattr(emb, cand)
                    break
            if fn is None:
                rep.add("text", f"{name} 텍스트 인코더", SKIP,
                        "encode_text/embed_text 메서드 없음")
                continue
            v = fn([query]) if _accepts_list(fn) else fn(query)
            import numpy as np
            arr = np.asarray(v, dtype="float32").reshape(-1)
            qv[name] = arr.tolist()
            rep.add("text", f"{name} 텍스트 인코딩", PASS, f"dim={arr.size}")
        except Exception as e:  # noqa: BLE001
            rep.add("text", f"{name} 텍스트 인코딩", FAIL, f"{type(e).__name__}: {e}")

    if not qv:
        rep.add("text", "자연어 검색", FAIL, "질의 벡터를 만들지 못함")
        return None

    try:
        t0 = time.time()
        hits = store.fused_search(query_vectors=qv, person_only=True, limit=topk)
        dt = time.time() - t0
    except Exception as e:  # noqa: BLE001
        rep.add("text", "자연어 검색", FAIL, f"{type(e).__name__}: {e}")
        return None

    if not hits:
        rep.add("text", "자연어 검색", FAIL, "결과 0건")
        return None

    srcs: Dict[str, int] = {}
    for h in hits:
        s = (h.payload or {}).get("source", "?")
        srcs[s] = srcs.get(s, 0) + 1
    rep.add("text", "자연어 검색", PASS,
            f"'{query[:34]}…' · {len(hits)}건 · {dt*1000:.0f}ms · {srcs}")
    return {"query": query, "sources": srcs, "ms": round(dt * 1000)}


def _accepts_list(fn) -> bool:
    import inspect
    try:
        p = list(inspect.signature(fn).parameters.values())
        return bool(p) and "list" in str(p[0].annotation).lower()
    except Exception:  # noqa: BLE001
        return True


# =============================================================================
# main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="person_db 무결성 + 검색 기능 점검")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--text", action="store_true",
                    help="자연어 검색까지 검사 (임베더 로드, GPU 사용)")
    ap.add_argument("--no-search", action="store_true", help="DB 점검만")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    cfg = PipelineConfig.load(str(CONFIG_PATH))
    store = QdrantStore(cfg)
    try:
        store.client._client.timeout = args.timeout  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass

    rep = Report()
    t0 = time.time()

    print("=" * 76)
    print(f"  person_db HEALTH CHECK   collection={cfg.collection}")
    print("=" * 76)

    facts = check_db(store, cfg, rep)

    searches: List[dict] = []
    if not args.no_search:
        searches = [s for s in check_search(store, cfg, rep, facts, args.topk) if s]

    text_result = check_text_search(store, cfg, rep, args.topk) if args.text else None

    # ── 요약 ──
    c = rep.counts()
    head("요약")
    print()
    print(f"  PASS {c[PASS]}   WARN {c[WARN]}   FAIL {c[FAIL]}   SKIP {c[SKIP]}"
          f"      ({time.time()-t0:.1f}s)")
    print()

    if c[FAIL]:
        print("  FAIL")
        for r in rep.rows:
            if r["status"] == FAIL:
                print(f"    - {r['name']}: {r['detail']}")
        print()
    if c[WARN]:
        print("  WARN")
        for r in rep.rows:
            if r["status"] == WARN:
                print(f"    - {r['name']}: {r['detail']}")
        print()
    if c[SKIP]:
        print("  SKIP")
        for r in rep.rows:
            if r["status"] == SKIP:
                print(f"    - {r['name']}: {r['detail']}")
        print()

    if not c[FAIL] and not c[WARN]:
        print("  이상 없음.")
        print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "facts": facts, "checks": rep.rows,
            "searches": searches, "text": text_result,
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"  JSON 저장: {args.json_out}")
        print()

    return 1 if c[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
