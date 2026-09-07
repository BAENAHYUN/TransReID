"""
verify_video.py — 동영상 크롭이 person_db 에서 제대로 검색되는지 확인한다.

적재가 끝났다는 것과 검색이 된다는 것은 다른 얘기다. 벡터가 들어갔어도
전처리가 어긋났거나 색 공간이 뒤집혔으면 에러 없이 검색 품질만 조용히 나빠진다.
COCO 는 자기검색 score 1.1000 으로 이걸 확인했지만 동영상은 아직 안 했다.


트랙을 파일시스템에서 고른다 (Qdrant scroll 을 쓰지 않는다)
--------------------------------------------------------
source 필터로 scroll 하면 68만 건을 훑는다. payload 인덱스를 만들어도 빌드가
끝나기 전에는 소용이 없고, 옵티마이저가 도는 동안에는 서버 쪽에서
'scroll_by_id timed out after 60s' 로 죽는다.

트랙 구조는 디스크에 그대로 있고 point ID 는 계산할 수 있으므로, 스캔 없이
ID 로 바로 꺼낸다:

    detection_id = "{id_source}|{video}|{track}|{frame_idx}|{person_idx}"
    point_id     = uuid5(NAMESPACE_URL, "{collection}|detection_id={detection_id}")

이 방식은 컬렉션이 yellow 여도 즉시 동작한다. 덤으로 **적재 누락 검사**도 된다 —
계산한 ID 를 retrieve 해서 안 나오면 그 크롭은 DB 에 없는 것이다.


무엇을 재는가
------------
트랙 하나에서 가운데 프레임을 골라 그 벡터로 DB 를 검색한다. 같은 트랙의 다른
프레임은 같은 사람의 연속된 모습이므로 **상위에 올라와야 정상**이다.

    1) 자기 자신이 1위인가          인덱스와 벡터가 온전한지
    2) 같은 트랙이 상위 K 에 몇 개   임베딩이 동일인을 붙잡는지
    3) 같은 영상 / 다른 소스 비율    무엇과 헷갈리는지

2) 가 낮으면 동영상 임베딩이 사람을 구분하지 못한다는 뜻이고, 그 상태로
클러스터링을 돌려봐야 의미가 없다.


두 가지 모드
-----------
--mode stored (기본)
    Qdrant 에 저장된 벡터를 그대로 꺼내 질의한다. GPU 불필요.
    저장 상태와 인덱스, 트랙 응집도를 본다.

--mode reembed
    크롭 파일을 다시 읽어 임베딩한 뒤 질의한다. GPU 사용.
    **색인 전처리와 질의 전처리가 일치하는지**를 본다 — stored 모드로는 절대
    잡을 수 없는 종류의 버그다. 자기 자신이 1위가 아니면 여기가 깨진 것이다.
    두 모드 결과가 다르면 전처리 불일치를 의심할 것.


사용법
-----
    python verify_video.py --tracks data/video_person_tracks_v4 \
        --id-source UCF-Crime --source UCF-Crime-Normal --html video_check.html

    python verify_video.py --tracks data/scvd_person_tracks_v1 \
        --id-source SCVD --samples 30

    python verify_video.py --tracks data/video_person_tracks_v4 \
        --id-source UCF-Crime --source UCF-Crime-Normal --mode reembed --samples 8

--id-source 는 **적재할 때 쓴 이름**이다 (detection_id 에 박혀 있다).
--source 는 payload 의 현재 이름이며, 결과 분류에만 쓴다. 생략하면 같다고 본다.
UCF 는 적재 후 이름을 바꿨으므로 둘이 다르다.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import PipelineConfig
from qdrant_store import QdrantStore

CONFIG_PATH = ROOT / "pipeline.yaml"

CROP_RE = re.compile(
    r"^frame_(?P<frame>\d+)_(?P<label>[A-Za-z][A-Za-z0-9_]*?)_(?P<idx>\d+)"
    r"\.(?:jpg|jpeg|png|bmp|webp)$", re.IGNORECASE,
)
TRACK_RE = re.compile(r"^track[_-]?(?P<num>\d+)$", re.IGNORECASE)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =============================================================================
# 파일시스템에서 트랙 표본 뽑기 + point ID 계산
# =============================================================================

def point_id_for(detection_id: str, collection: str) -> str:
    """QdrantStore._stable_point_id 와 동일한 규칙."""
    return str(uuid.uuid5(
        uuid.NAMESPACE_URL, f"{collection}|detection_id={detection_id}"
    ))


def sample_tracks_fs(
    tracks_dir: Path,
    id_source: str,
    collection: str,
    n_tracks: int,
    min_frames: int,
    seed: int,
    max_per_video: int = 3,
) -> List[Tuple[str, List[dict]]]:
    """트랙 디렉터리를 훑어 프레임이 min_frames 이상인 트랙을 n_tracks 개 고른다.

    전체를 다 읽지 않는다. 영상 목록을 먼저 섞고, 필요한 트랙 수를 채우면 멈춘다.

    max_per_video 로 영상당 트랙 수를 제한하는 이유
    ---------------------------------------------
    제한이 없으면 트랙이 많은 영상 하나가 표본을 다 먹는다. min_frames=8 로
    돌렸을 때 894 와 783 두 편에서 25개가 다 나왔다. 한 장면의 조명·화질·
    카메라 각도에만 의존한 수치가 되어 일반화가 안 된다. 영상당 몇 개로 끊고
    여러 편에서 모아야 데이터셋 전체를 대표한다.
    """
    if not tracks_dir.is_dir():
        raise FileNotFoundError(f"트랙 디렉터리가 없습니다: {tracks_dir}")

    rng = random.Random(seed)
    videos = sorted(p for p in tracks_dir.iterdir() if p.is_dir())
    if not videos:
        raise RuntimeError(f"영상 디렉터리가 없습니다: {tracks_dir}")
    rng.shuffle(videos)

    picked: List[Tuple[str, List[dict]]] = []
    n_scanned_videos = 0

    for vdir in videos:
        if len(picked) >= n_tracks:
            break
        video = vdir.name
        n_scanned_videos += 1

        tdirs = [p for p in sorted(vdir.iterdir()) if p.is_dir() and TRACK_RE.match(p.name)]
        rng.shuffle(tdirs)
        taken_here = 0

        for tdir in tdirs:
            if len(picked) >= n_tracks or taken_here >= max_per_video:
                break

            frames = []
            for f in sorted(tdir.iterdir()):
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                m = CROP_RE.match(f.name)
                if m is None:
                    continue
                frame_idx = int(m.group("frame"))
                person_idx = int(m.group("idx"))
                det_id = f"{id_source}|{video}|{tdir.name}|{frame_idx}|{person_idx}"
                frames.append({
                    "point_id": point_id_for(det_id, collection),
                    "detection_id": det_id,
                    "crop_path": str(f),
                    "frame_idx": frame_idx,
                    "person_idx": person_idx,
                    "video": video,
                    "track": tdir.name,
                })

            if len(frames) >= min_frames:
                frames.sort(key=lambda r: r["frame_idx"])
                track_key = f"{id_source}|{video}|{tdir.name}"
                picked.append((track_key, frames))
                taken_here += 1

    n_videos_used = len({tk.split("|")[1] for tk, _ in picked})
    print(f"  영상 {n_scanned_videos}편 훑어 {min_frames}프레임 이상 트랙 "
          f"{len(picked)}개 확보 (영상 {n_videos_used}편에서, "
          f"영상당 최대 {max_per_video})")
    if not picked:
        raise RuntimeError(
            f"{min_frames}프레임 이상인 트랙을 찾지 못했습니다. "
            f"--min-frames 를 낮추세요."
        )
    return picked


# =============================================================================
# 벡터 확보
# =============================================================================

def vectors_by_id(store: QdrantStore, cfg: PipelineConfig, point_id: str) -> Dict[str, list]:
    """ID 로 직접 꺼낸다. 스캔이 없으므로 yellow 상태에서도 빠르다."""
    recs = store.client.retrieve(
        collection_name=cfg.collection,
        ids=[point_id], with_vectors=True, with_payload=False,
    )
    if not recs:
        raise LookupError(
            "DB 에 이 point 가 없습니다. detection_id 규칙이 다르거나 "
            "적재에서 누락된 크롭입니다."
        )
    vec = recs[0].vector
    if not isinstance(vec, dict):
        raise RuntimeError("named vector 가 아닙니다. 컬렉션 구성이 예상과 다릅니다.")
    return vec


class ReEmbedder:
    """크롭 파일을 다시 읽어 임베딩한다 (색인/질의 전처리 일치 확인용).

    build_db.py 와 **같은 경로**(rfdetr_adapter -> Router -> registry)를 쓴다.
    직접 구현하면 그 자체가 전처리 불일치의 원인이 되므로 재사용해야 한다.
    """

    def __init__(self, cfg: PipelineConfig):
        from registry import EmbedderRegistry
        from router import Router

        print("임베더 로드 중 ...")
        t0 = time.time()
        self.cfg = cfg
        self.registry = EmbedderRegistry(cfg)
        self.router = Router(cfg, self.registry, input_format="rgb")
        print(f"  완료 ({time.time() - t0:.1f}s)")

    def embed(self, crop_path: str, label: str = "person") -> Dict[str, list]:
        from rfdetr_adapter import from_rfdetr

        p = Path(crop_path)
        if not p.is_file():
            p = ROOT / crop_path
        if not p.is_file():
            raise FileNotFoundError(f"크롭 파일이 없습니다: {crop_path}")

        dets, fmt = from_rfdetr(
            [{"path": str(p), "bbox": [0, 0, 0, 0], "confidence": 1.0,
              "class_name": label, "source_image": p.name, "frame_idx": 0}],
            load_mode="path",
        )
        if fmt != "rgb":
            raise RuntimeError(f"input_format 이 rgb 가 아닙니다: {fmt}")
        vecs = self.router.embed(dets)
        return {k: v.tolist() for k, v in vecs[0].items()}


# =============================================================================
# 판정
# =============================================================================

def evaluate(
    store: QdrantStore,
    query_vectors: Dict[str, list],
    self_id: str,
    track_key: str,
    video: str,
    payload_source: str,
    topk: int,
    prefetch: int,
) -> dict:
    hits = store.fused_search(
        query_vectors=query_vectors,
        person_only=True,
        limit=topk,
        prefetch_limit=prefetch,
    )

    self_rank = None
    same_track = same_video = same_source = other_source = 0
    rows = []

    for rank, h in enumerate(hits, 1):
        pay = h.payload or {}
        htk, hvid, hsrc = pay.get("track_key"), pay.get("video"), pay.get("source")

        if str(h.id) == str(self_id):
            self_rank = rank
            kind = "self"
        elif htk and htk == track_key:
            same_track += 1
            kind = "same-track"
        elif hvid and hvid == video:
            same_video += 1
            kind = "same-video"
        elif hsrc == payload_source:
            same_source += 1
            kind = "same-source"
        else:
            other_source += 1
            kind = hsrc or "?"

        rows.append({
            "rank": rank, "score": round(float(h.score), 5), "kind": kind,
            "source": hsrc, "video": hvid, "frame_idx": pay.get("frame_idx"),
            "crop_path": pay.get("crop_path"),
        })

    return {
        "track_key": track_key, "video": video, "self_rank": self_rank,
        "same_track": same_track, "same_video": same_video,
        "same_source": same_source, "other_source": other_source,
        "top_score": rows[0]["score"] if rows else None, "rows": rows,
    }


# =============================================================================
# 출력
# =============================================================================

def print_report(results: List[dict], source: str, topk: int, mode: str,
                 n_missing: int) -> None:
    print()
    print("=" * 78)
    print(f"  동영상 검색 정합성 · source={source} · mode={mode} · top-{topk}")
    print("=" * 78)
    print()
    print(f"  {'트랙':<40} {'프레임':>6} {'자기':>5} {'같은트랙':>8} {'타소스':>7}")
    print(f"  {'-'*40} {'-'*6} {'-'*5} {'-'*8} {'-'*7}")

    for r in results:
        tk = r["track_key"]
        short = tk if len(tk) <= 40 else "…" + tk[-39:]
        sr = str(r["self_rank"]) if r["self_rank"] else "없음"
        print(f"  {short:<40} {r.get('n_frames', 0):>6} {sr:>5} "
              f"{r['same_track']:>8} {r['other_source']:>7}")

    n = len(results)
    if not n:
        return

    self_1 = sum(1 for r in results if r["self_rank"] == 1)
    self_any = sum(1 for r in results if r["self_rank"] is not None)
    avg_track = sum(r["same_track"] for r in results) / n
    avg_video = sum(r["same_video"] for r in results) / n
    avg_other = sum(r["other_source"] for r in results) / n
    avg_frames = sum(r.get("n_frames", 0) for r in results) / n
    avg_top = sum(r["top_score"] or 0 for r in results) / n

    # 같은 트랙에서 실제로 올라올 수 있는 최대치 (자기 자신 제외)
    ceiling = min(topk - 1, max(0, avg_frames - 1))

    print()
    print("-" * 78)
    print(f"  표본 트랙            : {n}   (평균 {avg_frames:.1f}프레임)")
    print(f"  자기 자신 1위        : {self_1}/{n}  ({100*self_1/n:.0f}%)")
    print(f"  자기 자신 top-{topk} 내   : {self_any}/{n}")
    print(f"  1위 평균 점수        : {avg_top:.4f}")
    print(f"  같은 트랙 평균       : {avg_track:.1f}  (이 표본의 상한 {ceiling:.1f})")
    print(f"  같은 영상 다른 트랙  : {avg_video:.1f}")
    print(f"  다른 소스 혼입       : {avg_other:.1f} / {topk}")
    if n_missing:
        print(f"  DB 에 없던 크롭      : {n_missing}  <- 적재 누락")
    print()

    print("  판정")
    if self_1 == n:
        print("    OK   자기 자신이 항상 1위. 벡터와 인덱스가 온전하다.")
    elif self_any == n:
        print("    확인 자기 자신이 top-K 에는 있으나 1위가 아니다.")
        if mode == "reembed":
            print("         reembed 에서 1위가 아니면 색인/질의 전처리가 어긋났을")
            print("         수 있다. stored 모드 결과와 비교할 것.")
        else:
            print("         같은 트랙 이웃 프레임이 더 높게 나온 것이라면 정상이다.")
    else:
        print("    문제 자기 자신이 top-K 밖이다. 적재나 인덱스를 의심할 것.")

    # ── 트랙 길이를 고려한 판정 ──
    #
    # 상한(topk-1)을 다 채우는 것이 목표가 아니다. 68만 건 DB 에서 상위 20개 중
    # 절반 이상이 같은 트랙이면 이미 강한 결과이고, 20개 전부가 같은 트랙이면
    # 오히려 비교 대상이 DB 에 없다는 뜻이다. 또 짧은 트랙은 올릴 프레임 자체가
    # 없어 평균을 끌어내리므로, 프레임이 충분한 트랙만 따로 본다.
    LONG = 20
    longs = [r for r in results if r.get("n_frames", 0) >= LONG]

    if longs:
        cap = topk - 1
        lavg = sum(r["same_track"] for r in longs) / len(longs)
        lratio = lavg / cap
        print(f"    ({LONG}프레임 이상 트랙 {len(longs)}개 기준 같은트랙 "
              f"{lavg:.1f}/{cap})")
        if lratio >= 0.55:
            print("    OK   긴 트랙에서 같은 트랙이 상위를 채운다. 임베딩이")
            print("         동일인을 제대로 붙잡고 있다.")
        elif lratio >= 0.3:
            print("    확인 긴 트랙인데 같은 트랙이 절반에 못 미친다. 트랙 내")
            print("         외형 변화가 크거나 추적 ID 가 섞였을 수 있다.")
        else:
            print("    문제 긴 트랙에서도 같은 트랙이 거의 안 올라온다.")
            print("         크롭 품질과 전처리를 먼저 확인할 것.")
    else:
        print(f"    참고 {LONG}프레임 이상 트랙이 표본에 없어 판단이 어렵다.")
        print("         --min-frames 를 올려 다시 볼 것.")

    if avg_other >= topk * 0.5:
        print("    확인 다른 소스(이미지 DB)가 상위 절반 이상이다. 동영상 크롭이")
        print("         저해상도라 일반 사진 쪽으로 끌릴 수 있다.")

    # ── 이상 트랙 지목 ──
    #
    # 프레임이 충분한데 같은 트랙이 거의 안 올라오는 트랙. 추적기가 두 사람을
    # 한 트랙에 섞어 넣은(ID switching) 경우가 대표적이다. 같은 길이의 다른
    # 트랙이 만점을 받는 상황에서 혼자 낮으면 임베딩이 아니라 데이터 문제다.
    suspects = [
        r for r in results
        if r.get("n_frames", 0) >= LONG and r["same_track"] < (topk - 1) * 0.35
    ]
    if suspects:
        print()
        print("  눈으로 확인할 트랙 (프레임은 충분한데 자기 트랙이 안 올라옴)")
        print("    추적 ID 스위칭 — 한 트랙에 두 사람이 섞였을 가능성이 있다.")
        for r in sorted(suspects, key=lambda x: x["same_track"]):
            print(f"    {r['track_key']}")
            print(f"      {r['n_frames']}프레임 · 같은트랙 {r['same_track']}"
                  f" · 같은영상 {r['same_video']} · 타소스 {r['other_source']}")
        print("    --html 로 뽑아 해당 줄의 초록 테두리를 확인할 것.")
    print()


def write_html(results: List[dict], path: Path, source: str, mode: str, topk: int) -> None:
    """상위 결과를 크롭 이미지 격자로 저장한다. 숫자만으로는 판단이 안 된다."""
    def uri(p: Optional[str]) -> str:
        if not p:
            return ""
        f = Path(p)
        if not f.is_file():
            f = ROOT / p
        return f.resolve().as_uri() if f.is_file() else ""

    css = """
    body{background:#0f1117;color:#e6eaf2;font-family:system-ui,sans-serif;
         margin:0;padding:28px;font-size:14px}
    h1{font-size:19px;margin:0 0 4px} .sub{color:#7d8799;font-size:12px;margin-bottom:22px}
    .track{margin-bottom:30px;border-top:1px solid #262d3b;padding-top:14px}
    .tname{font-family:ui-monospace,monospace;font-size:12px;color:#9fb4e8;margin-bottom:9px}
    .row{display:flex;gap:7px;overflow-x:auto;padding-bottom:7px}
    .c{flex:0 0 auto;width:82px;text-align:center}
    .c img{width:82px;height:170px;object-fit:cover;border-radius:5px;
           border:2px solid #2a3040;background:#171b26;display:block}
    .c.self img{border-color:#facc15}
    .c.same-track img{border-color:#4ade80}
    .c.same-video img{border-color:#60a5fa}
    .c.other img{border-color:#f87171}
    .m{font-family:ui-monospace,monospace;font-size:9.5px;color:#8892a6;
       margin-top:3px;line-height:1.4}
    .legend{display:flex;gap:16px;font-size:11.5px;color:#8892a6;margin-bottom:22px;
            flex-wrap:wrap}
    .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;
        vertical-align:-1px}
    """
    KINDCLS = {"self": "self", "same-track": "same-track", "same-video": "same-video"}

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>동영상 검색 검증 · {source}</title><style>{css}</style>",
        "<h1>동영상 검색 정합성</h1>",
        f"<div class='sub'>source={source} · mode={mode} · top-{topk} · "
        f"트랙 {len(results)}개</div>",
        "<div class='legend'>"
        "<span><i class='sw' style='background:#facc15'></i>질의 자신</span>"
        "<span><i class='sw' style='background:#4ade80'></i>같은 트랙</span>"
        "<span><i class='sw' style='background:#60a5fa'></i>같은 영상</span>"
        "<span><i class='sw' style='background:#f87171'></i>다른 소스</span>"
        "</div>",
    ]

    for r in results:
        parts.append("<div class='track'>")
        parts.append(
            f"<div class='tname'>{r['track_key']} &nbsp;·&nbsp; "
            f"{r.get('n_frames',0)}프레임 &nbsp;·&nbsp; "
            f"자기 {r['self_rank'] or '없음'}위 &nbsp;·&nbsp; "
            f"같은트랙 {r['same_track']} &nbsp;·&nbsp; "
            f"타소스 {r['other_source']}</div>"
        )
        parts.append("<div class='row'>")
        for row in r["rows"]:
            cls = KINDCLS.get(row["kind"], "" if row["kind"] == "same-source" else "other")
            u = uri(row.get("crop_path"))
            img = (f"<img src='{u}' loading='lazy'>" if u else
                   "<div style='width:82px;height:170px;background:#1a1f2b;"
                   "border-radius:5px'></div>")
            parts.append(
                f"<div class='c {cls}'>{img}"
                f"<div class='m'>{row['rank']}위<br>{row['score']:.4f}<br>"
                f"f{row.get('frame_idx','')}</div></div>"
            )
        parts.append("</div></div>")

    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"HTML 저장: {path.resolve()}")


# =============================================================================
# main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="동영상 크롭이 person_db 에서 제대로 검색되는지 확인"
    )
    ap.add_argument("--tracks", default="data/video_person_tracks_v4",
                    help="트랙 디렉터리")
    ap.add_argument("--id-source", default="UCF-Crime",
                    help="detection_id 에 박힌 원래 source 이름 (적재 시 쓴 이름)")
    ap.add_argument("--source", default=None,
                    help="payload 의 현재 source 값 (결과 분류용). "
                         "생략하면 --id-source 와 같다고 본다")
    ap.add_argument("--mode", default="stored", choices=("stored", "reembed"))
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--prefetch", type=int, default=200)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--max-per-video", type=int, default=3,
                    help="영상당 최대 트랙 수. 트랙이 많은 영상 하나가 표본을 "
                         "다 먹는 것을 막는다 (기본 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--html", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    payload_source = args.source or args.id_source

    cfg = PipelineConfig.load(str(CONFIG_PATH))
    store = QdrantStore(cfg)
    try:
        store.client._client.timeout = args.timeout  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass

    print("=" * 78)
    print(f"collection : {cfg.collection}")
    print(f"tracks     : {args.tracks}")
    print(f"id-source  : {args.id_source}   payload-source: {payload_source}")

    try:
        info = store.client.get_collection(cfg.collection)
        print(f"status     : {info.status}  points {info.points_count:,}")
        if str(info.status).lower().endswith("yellow"):
            print("             (인덱싱 중 — 검색이 느릴 수 있으나 결과는 유효)")
    except Exception as e:  # noqa: BLE001
        print(f"status     : 확인 실패 ({e})")

    print()
    picked = sample_tracks_fs(
        Path(args.tracks), args.id_source, cfg.collection,
        n_tracks=args.samples, min_frames=args.min_frames, seed=args.seed,
        max_per_video=args.max_per_video,
    )
    print()

    embedder = ReEmbedder(cfg) if args.mode == "reembed" else None

    results: List[dict] = []
    n_missing = 0
    t0 = time.time()

    for i, (track_key, frames) in enumerate(picked, 1):
        q = frames[len(frames) // 2]          # 트랙 가운데 프레임
        label = f"{q['video']}/{q['track']}"

        try:
            if embedder is not None:
                qv = embedder.embed(q["crop_path"])
            else:
                qv = vectors_by_id(store, cfg, q["point_id"])

            r = evaluate(
                store, qv, q["point_id"], track_key, q["video"],
                payload_source, topk=args.topk, prefetch=args.prefetch,
            )
            r["n_frames"] = len(frames)
            results.append(r)
            print(f"  [{i:02d}/{len(picked)}] {label:<44} "
                  f"{len(frames):>3}f · 자기 {str(r['self_rank'] or '없음'):>4}위 · "
                  f"같은트랙 {r['same_track']:>2} · 타소스 {r['other_source']:>2}")

        except LookupError as e:
            n_missing += 1
            print(f"  [{i:02d}/{len(picked)}] {label:<44} 누락: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:02d}/{len(picked)}] {label:<44} 실패: {type(e).__name__}: {e}")

    print(f"\n소요: {time.time() - t0:.1f}s")

    if not results:
        print("\n결과가 없습니다.")
        if n_missing:
            print(f"모든 point 를 DB 에서 찾지 못했습니다 ({n_missing}건).")
            print(f"--id-source 가 적재 때 쓴 이름과 같은지 확인하세요 "
                  f"(현재 '{args.id_source}').")
        return 1

    print_report(results, payload_source, args.topk, args.mode, n_missing)

    if args.html:
        write_html(results, Path(args.html), payload_source, args.mode, args.topk)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 저장: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
