"""
ingest_source.py — person_db 에 새 데이터 소스를 추가 적재한다.

build_db.py 는 COCO 전용이다 (STATS_PATH 가 data/crops/filter_stats.json 로
하드코딩). 이 스크립트는 **어떤 소스든** 같은 컬렉션에 이어붙이기 위한 범용
도구다. 동영상 트랙 디렉터리도, 나중에 들어올 새 데이터셋도 여기로 처리한다.

    # 0) 기존 452,869건에 source=COCO 를 박는다 (동영상 넣기 전에 반드시)
    python ingest_source.py --backfill-source COCO

    # 1) 동영상 트랙 적재
    python ingest_source.py --tracks data/video_person_tracks_v4 --source-name UCF-Crime
    python ingest_source.py --tracks data/scvd_person_tracks_v1  --source-name SCVD \
        --manifest data/scvd_ingest_state_v2/person_track_manifest.jsonl

    # 2) 확인
    python ingest_source.py --stats-only


왜 별도 스크립트인가
------------------
build_db.py 를 고쳐 쓰면 COCO 적재 경로가 흔들린다. 이미 452,869건이 그
코드로 들어갔고, 그 재현성이 이 프로젝트에서 가장 비싼 자산이다. 새 소스는
새 파일에서 처리하고 build_db.py 는 건드리지 않는다.


--recreate 가 없다
-----------------
의도적이다. 이 스크립트는 컬렉션을 삭제할 수단을 제공하지 않는다.
기존 데이터를 날리려면 build_db.py --recreate --fresh 를 명시적으로 써야 한다.
"추가 적재하려다 전부 지웠다"가 물리적으로 불가능해진다.


동영상 크롭에는 bbox 가 없다
--------------------------
트랙 파이프라인(scvd_person_track_v2.py)이 저장한 것은 영상 단위 manifest 뿐이다.
크롭 파일명에서 복원되는 것은 frame_idx 와 person_idx 뿐이고, 프레임 좌표계
bbox 와 confidence 는 어디에도 남지 않았다.

임베딩에는 지장이 없다 — 크롭 이미지 자체를 넣기 때문이다. bbox 는 payload
출처 기록과 point ID fallback 용도인데, 후자는 detection_id 를 명시해 회피한다.

payload 에는 크롭 자기 좌표계 크기 [0, 0, w, h] 를 넣고 **bbox_space="crop"**
을 함께 기록한다. COCO 는 원본 이미지 좌표계이므로 "image" 다. 이 플래그가
없으면 나중에 누군가 동영상 bbox 를 프레임 좌표로 읽고 조용히 틀린다.


detection_id 를 왜 직접 만드는가
------------------------------
rfdetr_adapter.make_detection_id 는 crop 파일 경로를 ID 로 쓴다. 동영상 크롭
경로는 고유하긴 하지만 절대경로라 기계마다 다르다. Windows 에서 적재한 뒤
Linux 서버에서 재적재하면 같은 크롭이 별개 point 가 된다.

대신 동영상에는 경로보다 안정적인 신원이 있다:

    {source}|{video}|{track}|{frame_idx}|{person_idx}

이걸 detection_id 로 쓴다. 어느 기계에서 돌려도 같은 값이 나오므로
재실행이 중복이 아니라 덮어쓰기가 된다.


track_key
--------
Detection.track_id 는 int 라 "track_0001" -> 1 이 되는데, 이 값은 영상 안에서만
고유하다. 서로 다른 영상의 track_id=1 이 충돌한다. 그래서 payload 에
전역 고유한 track_key = "{source}|{video}|{track}" 를 따로 넣는다.
나중에 트랙 단위 집계 검색은 track_id 가 아니라 이 키로 묶어야 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router
from rfdetr_adapter import from_rfdetr
from qdrant_store import QdrantStore

CONFIG_PATH = ROOT / "pipeline.yaml"

# frame_00001770_person_01.jpg
CROP_RE = re.compile(
    r"^frame_(?P<frame>\d+)_(?P<label>[A-Za-z][A-Za-z0-9_]*?)_(?P<idx>\d+)"
    r"\.(?:jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE,
)

TRACK_RE = re.compile(r"^track[_-]?(?P<num>\d+)$", re.IGNORECASE)

# Normal_Videos_003_x264_frame_00000000_dog_0002.jpg
# n001_converted_frame_00000000_chair_0008.jpg
# ..._frame_00000000_potted_plant_0000.jpg   (라벨에 밑줄이 있어도 됨)
FLAT_CROP_RE = re.compile(
    r"^(?P<video>.+)_frame_(?P<frame>\d+)_(?P<label>.+)_(?P<idx>\d+)"
    r"\.(?:jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(raw: str) -> str:
    """파일명 라벨을 COCO 표기로 되돌린다.

    파일명에는 공백을 쓸 수 없어 'potted plant' 가 'potted_plant' 로 저장돼 있다.
    COCO(filter_stats.json)는 공백 표기를 쓰므로, 그대로 넣으면 같은 클래스가
    'potted plant' 와 'potted_plant' 두 갈래로 갈라져 label 필터가 절반만 잡는다.

    밑줄을 공백으로 바꿔 COCO 80 에 있으면 그 이름을 쓰고, 없으면 원본을 둔다.
    """
    from rfdetr_adapter import COCO80_NAMES

    low = raw.strip().lower()
    if low in COCO80_NAMES:
        return low
    spaced = low.replace("_", " ")
    if spaced in COCO80_NAMES:
        return spaced
    return low


# =============================================================================
# 유틸
# =============================================================================

def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# =============================================================================
# manifest
# =============================================================================

def load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    """person_track_manifest.jsonl -> {video_stem: {split, category, dataset}}

    한 줄 예:
      {"video": "Train/Normal/n001_converted.avi", "status": "ok",
       "source_dataset": "SCVD", "split": "Train"}

    video 필드의 중간 경로에서 category(Normal/Violence 등)를 뽑는다.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        print(f"  manifest 없음, split/category 생략: {path}")
        return out

    n_bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue

            vid = str(rec.get("video", ""))
            if not vid:
                continue

            parts = Path(vid.replace("\\", "/")).parts
            stem = Path(parts[-1]).stem

            # 경로에서 category 를 유추하는 것은 SCVD 형식("Train/Normal/x.avi")
            # 에서만 옳다. UCF manifest 는 video 가 전체 경로라
            # ("C:/.../data/videos/Normal_Videos_003_x264.mp4") 그대로 쓰면
            # category="videos" 라는 쓰레기가 payload 에 들어간다.
            category = rec.get("category")
            if category is None and rec.get("split") and len(parts) >= 2:
                category = parts[-2]

            out[stem] = {
                "split": rec.get("split"),
                "category": category,
                "dataset": rec.get("source_dataset"),
                "status": rec.get("status"),
            }

    if n_bad:
        print(f"  manifest 파싱 실패 {n_bad}줄 (무시)")
    print(f"  manifest 로드: {len(out)} 영상")
    return out


# =============================================================================
# 스캔 — 트랙 디렉터리
# =============================================================================

def scan_tracks(
    root: Path,
    source: str,
    manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    read_size: bool = True,
    default_label: str = "person",
) -> List[Dict[str, Any]]:
    """
    {root}/{video}/{track_NNNN}/frame_XXXXXXXX_label_NN.jpg 를 스캔해
    from_rfdetr 가 먹는 레코드 리스트를 만든다.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"트랙 디렉터리가 없습니다: {root}")

    manifest = manifest or {}
    records: List[Dict[str, Any]] = []
    skipped_name = 0
    n_videos = n_tracks = 0

    if read_size:
        from PIL import Image

    videos = sorted(p for p in root.iterdir() if p.is_dir())
    for vdir in videos:
        video = vdir.name
        meta = manifest.get(video, {})
        n_videos += 1

        for tdir in sorted(p for p in vdir.iterdir() if p.is_dir()):
            m = TRACK_RE.match(tdir.name)
            if m is None:
                continue
            track_num = int(m.group("num"))
            track_key = f"{source}|{video}|{tdir.name}"
            n_tracks += 1

            for f in sorted(tdir.iterdir()):
                if f.suffix.lower() not in IMG_EXTS:
                    continue

                cm = CROP_RE.match(f.name)
                if cm is None:
                    skipped_name += 1
                    continue

                frame_idx = int(cm.group("frame"))
                label = cm.group("label").lower()
                person_idx = int(cm.group("idx"))

                if read_size:
                    try:
                        with Image.open(f) as im:
                            w, h = im.size
                    except Exception:  # noqa: BLE001
                        w = h = 0
                else:
                    w = h = 0

                records.append({
                    # from_rfdetr 가 읽는 키
                    "path": str(f),
                    "bbox": [0, 0, float(w), float(h)],
                    "confidence": 1.0,          # 트랙 단계에서 보존되지 않음
                    "class_name": label or default_label,
                    "source_image": f"{video}/{f.name}",
                    "frame_idx": frame_idx,
                    "track_num": track_num,
                    # 기계 독립 · 재실행 안전
                    "detection_id":
                        f"{source}|{video}|{tdir.name}|{frame_idx}|{person_idx}",
                    # 아래는 from_rfdetr 가 안 넘기므로 이후 extra 에 직접 주입
                    "_source": source,
                    "_video": video,
                    "_track_key": track_key,
                    "_person_idx": person_idx,
                    "_split": meta.get("split"),
                    "_category": meta.get("category"),
                })

    print(f"  영상 {n_videos} · 트랙 {n_tracks} · 크롭 {len(records):,}")
    if skipped_name:
        print(f"  파일명 패턴 불일치로 제외: {skipped_name}건 "
              f"(기대 형식: frame_00001770_person_01.jpg)")
    if not records:
        raise RuntimeError(f"적재할 크롭이 없습니다: {root}")
    return records


# =============================================================================
# 스캔 — 평면 크롭 (동영상 객체)
# =============================================================================

def scan_flat_crops(
    root: Path,
    source: str,
    manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    read_size: bool = True,
) -> List[Dict[str, Any]]:
    """
    {root}/{dir}/{video}_frame_XXXXXXXX_{label}_NNNN.jpg 를 스캔한다.

    동영상 객체 크롭에는 트랙이 없다. 사람은 추적을 붙였지만 객체는 프레임마다
    독립 검출이라 track_key 를 만들지 않는다.

    video 를 디렉터리명이 아니라 **파일명 접두사**에서 뽑는 이유
    -------------------------------------------------------
    SCVD 는 둘이 다르다:

        디렉터리 : scvd_object_search_v1/crops/Train_Normal_n001_converted/
        파일     : n001_converted_frame_00000000_chair_0008.jpg
        사람 트랙: scvd_person_tracks_v1/n001_converted/...

    디렉터리명을 쓰면 같은 영상인데 사람 point 는 video='n001_converted',
    객체 point 는 video='Train_Normal_n001_converted' 가 되어 join 이 깨진다.
    파일명 접두사를 쓰면 사람 쪽과 맞는다. UCF 는 둘이 같으므로 영향 없다.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"크롭 디렉터리가 없습니다: {root}")

    manifest = manifest or {}
    records: List[Dict[str, Any]] = []
    skipped_name = 0
    labels_raw: Dict[str, str] = {}

    if read_size:
        from PIL import Image

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subdirs:
        raise RuntimeError(f"하위 디렉터리가 없습니다: {root}")

    for sub in subdirs:
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue

            m = FLAT_CROP_RE.match(f.name)
            if m is None:
                skipped_name += 1
                continue

            video = m.group("video")
            frame_idx = int(m.group("frame"))
            raw_label = m.group("label")
            det_idx = int(m.group("idx"))

            label = normalize_label(raw_label)
            if label != raw_label.lower():
                labels_raw[raw_label.lower()] = label

            meta = manifest.get(video, {})

            if read_size:
                try:
                    with Image.open(f) as im:
                        w, h = im.size
                except Exception:  # noqa: BLE001
                    w = h = 0
            else:
                w = h = 0

            records.append({
                "path": str(f),
                "bbox": [0, 0, float(w), float(h)],
                "confidence": 1.0,          # 검출 단계에서 보존되지 않음
                "class_name": label,
                "source_image": f"{video}/{f.name}",
                "frame_idx": frame_idx,
                "detection_id":
                    f"{source}|{video}|{frame_idx}|{label}|{det_idx}",
                "_source": source,
                "_video": video,
                "_track_key": None,         # 객체는 추적하지 않는다
                "_person_idx": None,
                "_split": meta.get("split"),
                "_category": meta.get("category"),
            })

    print(f"  디렉터리 {len(subdirs)} · 크롭 {len(records):,}")
    if labels_raw:
        print(f"  라벨 정규화: {dict(list(labels_raw.items())[:6])}")
    if skipped_name:
        print(f"  파일명 패턴 불일치로 제외: {skipped_name}건 "
              f"(기대 형식: <video>_frame_00000000_<label>_0000.jpg)")
    if not records:
        raise RuntimeError(f"적재할 크롭이 없습니다: {root}")
    return records


# =============================================================================
# 스캔 — filter_stats.json 형식
# =============================================================================

def scan_stats(path: Path, source: str) -> List[Dict[str, Any]]:
    """detect_rf.py 가 만든 filter_stats.json 을 읽는다 (COCO 와 같은 형식)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    crops = data.get("crops") or []
    for r in crops:
        r["_source"] = source
        r.setdefault("_video", None)
        r.setdefault("_track_key", None)
    print(f"  크롭 {len(crops):,}")
    if not crops:
        raise RuntimeError(f"적재할 크롭이 없습니다: {path}")
    return crops


# =============================================================================
# Detection 후처리 — from_rfdetr 가 안 넘기는 payload 주입
# =============================================================================

def inject_extra(detections, records, bbox_space: str) -> None:
    """
    from_rfdetr 는 extra 에 crop_id / crop_path / max_person_iou 만 넣는다.
    source / video / track_key 등을 여기서 추가한다.

    QdrantStore.RESERVED_PAYLOAD_KEYS 와 겹치면 upsert 가 거부하므로
    (image_id, frame_idx, label, is_person, score, bbox, detection_id, track_id)
    그 이름은 절대 쓰지 않는다.
    """
    if len(detections) != len(records):
        raise RuntimeError(
            f"detection / record 개수 불일치: {len(detections)} != {len(records)}"
        )

    for det, r in zip(detections, records):
        det.extra["source"] = r["_source"]
        det.extra["bbox_space"] = bbox_space

        if r.get("_video"):
            det.extra["video"] = r["_video"]
        if r.get("_track_key"):
            det.extra["track_key"] = r["_track_key"]
        if r.get("_person_idx") is not None:
            det.extra["person_idx"] = int(r["_person_idx"])
        if r.get("_split"):
            det.extra["split"] = r["_split"]
        if r.get("_category"):
            det.extra["category"] = r["_category"]


# =============================================================================
# source 백필 / 통계
# =============================================================================

def point_id_for(record: Dict[str, Any], collection: str) -> str:
    """QdrantStore._stable_point_id 와 동일한 규칙으로 point ID 를 재계산한다.

        detection_id -> f"{collection}|detection_id={detection_id}" -> uuid5

    build_db.py / build_db_query.py 는 from_rfdetr 를 거쳤으므로 detection_id 는
    make_detection_id(record) 와 같다. 즉 stats JSON 만 있으면 그 빌드가 만든
    point 를 정확히 지목할 수 있다.
    """
    import uuid
    from rfdetr_adapter import make_detection_id

    detection_id = make_detection_id(record, int(record.get("frame_idx", 0)))
    return str(uuid.uuid5(
        uuid.NAMESPACE_URL, f"{collection}|detection_id={detection_id}"
    ))


def backfill_from_stats(
    store: QdrantStore,
    cfg: PipelineConfig,
    payload: Dict[str, Any],
    stats_path: Path,
    batch: int = 1000,
) -> None:
    """stats JSON 이 만든 point 에 **정확히** payload 를 박는다.

    왜 필터가 아니라 ID 인가
    ----------------------
    "source 가 없는 point 전부"에 거는 방식은 컬렉션에 데이터셋이 하나일 때만
    맞다. person_db 에는 이미 둘이 있다:

        data/crops/filter_stats.json        452,869   (build_db.py)
        data/query_crops/filter_stats.json  191,490   (build_db_query.py)
                                            -------
                                            644,359

    필터로 뭉뚱그리면 쿼리측 19만건까지 COCO 로 잘못 찍힌다. 각 빌드의 stats
    파일에서 point ID 를 되계산해 그 집합에만 태깅해야 정확하다.

    또한 필터 한 방 set_payload 는 대량/인덱싱중(status=yellow) 상황에서 일부만
    적용되고 끝나는 경우가 관측됐다 (644,359 중 약 4만건만 처리). 명시적 ID
    배치로 나누면 진행 상황이 보이고 중단해도 이어서 돌릴 수 있다.
    """
    if not stats_path.is_file():
        raise FileNotFoundError(f"stats 파일이 없습니다: {stats_path}")

    print(f"stats 로드: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        records = json.load(f).get("crops") or []

    total = len(records)
    print(f"  레코드 {total:,}")
    if total == 0:
        print("  비어 있습니다.")
        return

    coll = cfg.collection
    client = store.client

    print(f"  point ID 재계산 중 ...")
    ids = [point_id_for(r, coll) for r in records]

    uniq = len(set(ids))
    if uniq != total:
        print(f"  주의: ID 중복 {total - uniq:,}건 (같은 point 를 여러 번 지목)")

    print(f"{payload} 부여 중 ({total:,}건, batch={batch}) ...")
    t0 = time.time()
    done = 0

    for i in range(0, total, batch):
        chunk = ids[i:i + batch]
        client.set_payload(
            collection_name=coll,
            payload=payload,
            points=chunk,
            wait=True,
        )
        done += len(chunk)
        if done % (batch * 10) == 0 or done == total:
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  {done:>9,}/{total:,}  {100*done/total:6.2f}%  "
                  f"{rate:7.0f}/s  ETA {format_seconds(eta)}")

    print(f"완료: {done:,}건  ({format_seconds(time.time() - t0)})")


def backfill_from_tracks(
    store: QdrantStore,
    cfg: PipelineConfig,
    name: str,
    tracks_dir: Path,
    id_source: str,
    batch: int = 1000,
) -> None:
    """트랙 디렉터리가 만든 point 에 source=name 을 박는다 (이름 변경용).

    id_source 가 따로 있는 이유
    -------------------------
    detection_id 는 적재 시점의 source 이름을 포함한다:

        {source}|{video}|{track}|{frame}|{person_idx}

    따라서 이미 적재된 point 를 다시 지목하려면 **적재할 때 쓴 이름**으로
    스캔해야 한다. 바꾸고 싶은 새 이름은 payload 에만 쓴다.

        --from-tracks data/video_person_tracks_v4 \
        --tracks-id-source UCF-Crime          <- ID 계산용 (옛 이름)
        --backfill-source  UCF-Crime-Normal   <- payload 에 쓸 새 이름

    주의: 이 방식은 payload 만 바꾼다. point ID 는 옛 이름에 묶인 채 남는다.
    나중에 같은 트랙을 재적재할 일이 생기면 반드시 --source-name 에 옛 이름
    (id_source)을 주어야 같은 point 를 덮어쓴다. 새 이름으로 적재하면 ID 가
    달라져 중복이 생긴다.
    """
    print(f"트랙 스캔 (ID 계산용 source='{id_source}') ...")
    records = scan_tracks(Path(tracks_dir), id_source, read_size=False)

    total = len(records)
    coll = cfg.collection
    client = store.client

    print(f"  point ID 재계산 중 ...")
    ids = [point_id_for(r, coll) for r in records]

    if id_source != name:
        print(f"  payload source: '{id_source}' -> '{name}'")

    print(f"source='{name}' 부여 중 ({total:,}건, batch={batch}) ...")
    payload = {"source": name}
    t0 = time.time()
    done = 0
    for i in range(0, total, batch):
        chunk = ids[i:i + batch]
        client.set_payload(
            collection_name=coll, payload={"source": name},
            points=chunk, wait=True,
        )
        done += len(chunk)
        if done % (batch * 10) == 0 or done == total:
            print(f"  {done:>9,}/{total:,}  {100*done/total:6.2f}%")

    print(f"완료: {done:,}건  ({format_seconds(time.time() - t0)})")
    if id_source != name:
        print(
            f"\n기억할 것: 이 트랙들의 detection_id 는 여전히 '{id_source}' 를 씁니다.\n"
            f"  재적재할 일이 생기면 --source-name {id_source} 로 스캔한 뒤\n"
            f"  --backfill-source {name} 로 이름을 다시 덮으세요."
        )


def backfill_field_by_source(
    store: QdrantStore,
    cfg: PipelineConfig,
    field: str,
    value: Any,
    sources: Sequence[str],
) -> None:
    """지정한 source 들의 point 에 임의 payload 필드를 채운다.

    쓰임새: build_db.py 로 적재한 이미지 소스에는 나중에 도입된 필드가 없다.
    대표적으로 bbox_space — 동영상은 "crop", 이미지는 "image" 인데 이미지 쪽은
    필드 자체가 비어 있다. "없으면 image" 라는 규칙이 코드가 아니라 사람 기억에만
    남으면, 언젠가 bbox_space=="image" 필터가 0건을 돌려주고 아무도 이유를 모른다.

    source 에 인덱스가 있으므로 필터 방식으로도 빠르다. 다만 필터 기반
    set_payload 는 반환 후에도 백그라운드로 적용되므로, 직후 count 가 덜 올라와
    보이는 것이 정상이다. 몇 분 뒤 --stats-only 로 확인할 것.
    """
    from qdrant_client import models

    client = store.client
    coll = cfg.collection

    for name in sources:
        flt = models.Filter(must=[models.FieldCondition(
            key="source", match=models.MatchValue(value=name))])
        n = client.count(coll, count_filter=flt, exact=True).count
        if n == 0:
            print(f"  {name:<22} 대상 없음 — 건너뜀")
            continue

        print(f"  {name:<22} {n:>9,}건에 {field}={value!r} ... ", end="", flush=True)
        t0 = time.time()
        client.set_payload(
            collection_name=coll,
            payload={field: value},
            points=flt,
            wait=True,
        )
        print(f"요청 완료 ({time.time() - t0:.1f}s)")

    print("\n필터 기반 갱신은 비동기로 적용됩니다. "
          "몇 분 뒤 --stats-only 로 확인하세요.")


def backfill_untagged(
    store: QdrantStore,
    cfg: PipelineConfig,
    name: str,
    batch: int = 1000,
) -> None:
    """source 가 없는 point 를 scroll 로 훑어 배치로 태깅한다.

    stats 파일이 없는 출처를 정리할 때만 쓴다. 무엇이 남았는지 모르는 상태에서
    이름을 붙이는 것이므로, 가능하면 --from-stats 를 먼저 쓸 것.
    """
    from qdrant_client import models

    client = store.client
    coll = cfg.collection

    flt = models.Filter(must=[models.IsEmptyCondition(
        is_empty=models.PayloadField(key="source"))])

    remaining = client.count(coll, count_filter=flt, exact=True).count
    print(f"source 없는 point: {remaining:,}")
    if remaining == 0:
        print("대상이 없습니다.")
        return

    print(f"source='{name}' 부여 중 (scroll batch={batch}) ...")
    t0 = time.time()
    done = 0

    while True:
        # 태깅하면 필터에서 빠지므로 항상 처음부터 훑어도 무한루프가 아니다.
        points, _ = client.scroll(
            collection_name=coll,
            scroll_filter=flt,
            limit=batch,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break

        client.set_payload(
            collection_name=coll,
            payload={"source": name},
            points=[p.id for p in points],
            wait=True,
        )
        done += len(points)
        el = time.time() - t0
        rate = done / el if el > 0 else 0
        print(f"  {done:>9,}  {rate:7.0f}/s")

    left = client.count(coll, count_filter=flt, exact=True).count
    print(f"완료: {done:,}건 태깅 · 남은 미태깅 {left:,}")


def ensure_extra_indexes(store: QdrantStore, cfg: PipelineConfig) -> None:
    """ingest_source.py 가 추가한 payload 필드에 인덱스를 만든다.

    qdrant_store._ensure_payload_indexes() 는 build_db.py 시절 필드만 안다:

        is_person(bool) · label(keyword) · image_id(keyword) · frame_idx(integer)

    source / video / track_key 등은 이 스크립트가 새로 넣은 것이라 인덱스가 없다.
    인덱스 없는 필드로 필터를 걸면 Qdrant 가 전체 point 를 훑는다. 68만 건에
    on_disk 벡터까지 얹히면 기본 타임아웃(5초)을 넘겨 그냥 실패한다.

    특히 앞으로 쓸 것들이라 미리 만들어 둔다:
        source     데이터셋별 필터 / 통계
        video      영상 단위 조회
        track_key  트랙 집계, 클러스터링 평가 (전역 고유 키)
    """
    from qdrant_client import models

    desired = {
        "source": models.PayloadSchemaType.KEYWORD,
        "video": models.PayloadSchemaType.KEYWORD,
        "track_key": models.PayloadSchemaType.KEYWORD,
        "split": models.PayloadSchemaType.KEYWORD,
        "category": models.PayloadSchemaType.KEYWORD,
        "bbox_space": models.PayloadSchemaType.KEYWORD,
        "person_idx": models.PayloadSchemaType.INTEGER,
    }

    client = store.client
    coll = cfg.collection

    info = client.get_collection(coll)
    existing = set((getattr(info, "payload_schema", None) or {}).keys())

    print(f"기존 인덱스: {sorted(existing) or '(없음)'}")

    todo = [f for f in desired if f not in existing]
    if not todo:
        print("추가할 인덱스가 없습니다.")
        return

    print(f"생성할 인덱스: {todo}")
    if str(getattr(info, "status", "")).endswith("yellow"):
        print(
            "\n주의: collection status 가 yellow(인덱싱 중) 입니다.\n"
            "  지금 만들어도 되지만 green 이 된 뒤가 더 빠릅니다.\n"
        )

    for field in todo:
        t0 = time.time()
        print(f"  {field} ... ", end="", flush=True)
        try:
            client.create_payload_index(
                collection_name=coll,
                field_name=field,
                field_schema=desired[field],
                wait=True,
            )
            print(f"완료 ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"실패: {e}")

    info = client.get_collection(coll)
    now = set((getattr(info, "payload_schema", None) or {}).keys())
    print(f"\n현재 인덱스: {sorted(now)}")


def print_source_stats(store: QdrantStore, cfg: PipelineConfig) -> None:
    from qdrant_client import models

    client = store.client
    coll = cfg.collection

    info = client.get_collection(coll)
    total = info.points_count
    print(f"\ncollection : {coll}")
    print(f"total      : {total:,}  (status={info.status})")

    empty = client.count(
        coll,
        count_filter=models.Filter(must=[models.IsEmptyCondition(
            is_empty=models.PayloadField(key="source"))]),
        exact=True,
    ).count

    # 실제로 존재하는 source 값을 표본으로 발견한 뒤 각각 정확히 센다.
    # (하드코딩 목록만 세면 모르는 값이 "기타"로 뭉뚱그려져 놓친다)
    found: set = set()
    offset = None
    scanned = 0
    while scanned < 20000:
        pts, offset = client.scroll(
            collection_name=coll, limit=2000, offset=offset,
            with_payload=["source"], with_vectors=False,
        )
        if not pts:
            break
        for p in pts:
            v = (p.payload or {}).get("source")
            if v:
                found.add(str(v))
        scanned += len(pts)
        if offset is None:
            break

    print(f"\nsource 분포:  (표본 {scanned:,}건에서 값 발견)")
    seen = 0
    for name in sorted(found):
        n = client.count(
            coll,
            count_filter=models.Filter(must=[models.FieldCondition(
                key="source", match=models.MatchValue(value=name))]),
            exact=True,
        ).count
        print(f"  {name:<14} {n:>10,}")
        seen += n

    if empty:
        print(f"  {'(없음)':<14} {empty:>10,}   <- 백필 필요")
    other = total - seen - empty
    if other > 0:
        print(f"  {'(표본밖)':<14} {other:>10,}   <- 표본에 안 잡힌 source 값 존재")

    n_person = client.count(
        coll,
        count_filter=models.Filter(must=[models.FieldCondition(
            key="is_person", match=models.MatchValue(value=True))]),
        exact=True,
    ).count
    print(f"\nis_person=True : {n_person:,}")
    print(f"is_person=False: {total - n_person:,}\n")


# =============================================================================
# main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="person_db 에 새 데이터 소스를 추가 적재 (컬렉션 삭제 불가)"
    )

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tracks", metavar="DIR",
                      help="트랙 디렉터리 (video/track_NNNN/frame_*.jpg)")
    mode.add_argument("--crops", metavar="DIR",
                      help="평면 크롭 디렉터리 (<dir>/<video>_frame_N_<label>_N.jpg). "
                           "동영상 객체 크롭용 — 트랙이 없다")
    mode.add_argument("--stats", metavar="JSON",
                      help="filter_stats.json 형식 파일")
    mode.add_argument("--backfill-source", metavar="NAME",
                      help="기존 point 에 source=NAME 을 부여하고 종료. "
                           "--from-stats 와 함께 쓰는 것을 강력히 권장")
    mode.add_argument("--stats-only", action="store_true",
                      help="컬렉션 현황만 출력하고 종료")
    mode.add_argument("--set-field", metavar="KEY=VALUE",
                      help="--for-sources 로 지정한 source 들에 payload 필드를 "
                           "채우고 종료. 예: --set-field bbox_space=image")
    mode.add_argument("--ensure-indexes", action="store_true",
                      help="source / video / track_key 등에 payload 인덱스를 "
                           "만들고 종료. 인덱스가 없으면 이 필드로 거는 필터가 "
                           "전수 스캔이 되어 타임아웃 난다.")

    ap.add_argument("--from-stats", metavar="JSON", default=None,
                    help="백필 대상을 이 stats JSON 이 만든 point 로 한정한다. "
                         "생략하면 'source 가 없는 모든 point' 가 대상이 되는데, "
                         "컬렉션에 데이터셋이 둘 이상이면 잘못 태깅된다.")
    ap.add_argument("--from-tracks", metavar="DIR", default=None,
                    help="백필 대상을 이 트랙 디렉터리가 만든 point 로 한정한다 "
                         "(트랙 소스 이름 변경용)")
    ap.add_argument("--tracks-id-source", metavar="NAME", default=None,
                    help="--from-tracks 로 ID 를 계산할 때 쓸 원래 source 이름. "
                         "생략하면 --backfill-source 와 같다고 본다. "
                         "이름을 바꾸는 경우 반드시 **적재할 때 쓴 옛 이름**을 준다.")
    ap.add_argument("--backfill-batch", type=int, default=1000,
                    help="백필 배치 크기 (기본 1000)")
    ap.add_argument("--for-sources", metavar="NAME", action="append", default=None,
                    help="--set-field 를 적용할 source (여러 번 지정 가능). "
                         "--from-stats 가 있으면 그쪽이 우선한다")
    ap.add_argument("--timeout", type=int, default=300,
                    help="Qdrant 클라이언트 타임아웃(초). 기본 300")

    ap.add_argument("--source-name",
                    help="이 소스의 이름 (COCO / SCVD / UCF-Crime ...). "
                         "--tracks / --stats 에 필수")
    ap.add_argument("--manifest",
                    help="person_track_manifest.jsonl (split/category 보강)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--checkpoint", default=None,
                    help="체크포인트 디렉터리 (기본: data/embedding_checkpoint_<source>)")
    ap.add_argument("--fresh", action="store_true",
                    help="체크포인트를 지우고 0부터")
    ap.add_argument("--dry-run", action="store_true",
                    help="스캔만 하고 적재하지 않음")
    ap.add_argument("--limit", type=int, default=None,
                    help="앞에서 N건만 처리 (테스트용)")
    ap.add_argument("--no-read-size", action="store_true",
                    help="크롭 이미지 크기를 읽지 않음 (bbox=[0,0,0,0])")
    ap.add_argument("--bbox-space", default=None,
                    help="crop | image (기본: tracks=crop, stats=image)")
    args = ap.parse_args()

    print("=" * 70)
    print("LOAD CONFIG")
    print("=" * 70)
    cfg = PipelineConfig.load(str(CONFIG_PATH))
    print("collection:", cfg.collection)

    store = QdrantStore(cfg)
    try:
        store.client._client.timeout = args.timeout  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass

    # ── 임의 필드 백필만 ──
    if args.set_field:
        if "=" not in args.set_field:
            ap.error("--set-field 는 KEY=VALUE 형식입니다 (예: bbox_space=image)")
        if not args.for_sources and not args.from_stats:
            ap.error("--set-field 에는 --from-stats 또는 --for-sources 가 필요합니다")

        key, raw = args.set_field.split("=", 1)
        key = key.strip()
        low = raw.strip().lower()
        if low in ("true", "false"):
            val: Any = (low == "true")
        else:
            try:
                val = int(raw)
            except ValueError:
                val = raw.strip()

        print("\n" + "=" * 70)
        print(f"SET FIELD  {key} = {val!r}")
        print("=" * 70)

        if args.from_stats:
            # 권장 경로: point ID 를 계산해 배치로 쓴다.
            backfill_from_stats(
                store, cfg, {key: val},
                Path(args.from_stats), batch=args.backfill_batch,
            )
        else:
            # 필터 경로. 대상이 크면 서버가 백그라운드로 적용하는 동안 다음 요청이
            # 밀려 타임아웃 난다 (COCO 45만건에서 실제로 발생). 가능하면
            # --from-stats 를 쓸 것.
            print(
                "경고: 필터 방식으로 실행합니다. 대상이 10만건을 넘으면\n"
                "  서버가 백그라운드 적용하는 동안 다음 요청이 밀려 타임아웃 납니다.\n"
                "  stats JSON 이 있으면 --from-stats 쪽이 훨씬 안전합니다.\n"
            )
            backfill_field_by_source(store, cfg, key, val, args.for_sources)
        return 0

    # ── 인덱스 생성만 ──
    if args.ensure_indexes:
        print("\n" + "=" * 70)
        print("PAYLOAD INDEXES")
        print("=" * 70)
        ensure_extra_indexes(store, cfg)
        return 0

    # ── 현황만 ──
    if args.stats_only:
        print_source_stats(store, cfg)
        return 0

    # ── 백필만 ──
    if args.backfill_source:
        print("\n" + "=" * 70)
        print(f"BACKFILL source = {args.backfill_source}")
        print("=" * 70)

        if args.from_stats and args.from_tracks:
            ap.error("--from-stats 와 --from-tracks 는 함께 쓸 수 없습니다")

        if args.from_stats:
            backfill_from_stats(
                store, cfg, {"source": args.backfill_source},
                Path(args.from_stats), batch=args.backfill_batch,
            )
        elif args.from_tracks:
            backfill_from_tracks(
                store, cfg, args.backfill_source,
                Path(args.from_tracks),
                id_source=args.tracks_id_source or args.backfill_source,
                batch=args.backfill_batch,
            )
        else:
            print(
                "경고: --from-stats 없이 실행합니다.\n"
                "  'source 가 없는 모든 point' 가 대상이 됩니다.\n"
                "  컬렉션에 데이터셋이 둘 이상이면 서로 다른 출처가 한 이름으로\n"
                "  뭉뚱그려집니다. 각 빌드의 filter_stats.json 이 있다면\n"
                "  --from-stats 로 하나씩 태깅하는 편이 정확합니다.\n"
            )
            backfill_untagged(
                store, cfg, args.backfill_source, batch=args.backfill_batch
            )

        print_source_stats(store, cfg)
        return 0

    # ── 적재 ──
    if not args.source_name:
        ap.error("--tracks / --crops / --stats 에는 --source-name 이 필요합니다")

    source = args.source_name.strip()
    if not source:
        ap.error("--source-name 이 비었습니다")

    print("\n" + "=" * 70)
    print(f"SCAN  source={source}")
    print("=" * 70)

    manifest = load_manifest(Path(args.manifest)) if args.manifest else {}

    if args.tracks:
        records = scan_tracks(
            Path(args.tracks), source,
            manifest=manifest,
            read_size=not args.no_read_size,
        )
        bbox_space = args.bbox_space or "crop"
    elif args.crops:
        records = scan_flat_crops(
            Path(args.crops), source,
            manifest=manifest,
            read_size=not args.no_read_size,
        )
        bbox_space = args.bbox_space or "crop"
    else:
        records = scan_stats(Path(args.stats), source)
        bbox_space = args.bbox_space or "image"

    if args.limit:
        records = records[:args.limit]
        print(f"  --limit {args.limit} 적용 -> {len(records):,}건")

    total = len(records)

    # detection_id 중복 사전 점검 (Qdrant 에서 조용히 덮어쓰기 되는 것 방지)
    ids = [r.get("detection_id") or r.get("path") for r in records]
    if len(set(ids)) != len(ids):
        from collections import Counter
        dup = [k for k, v in Counter(ids).most_common(5) if v > 1]
        raise RuntimeError(
            f"detection_id 중복 {len(ids) - len(set(ids))}건. 예: {dup}\n"
            "이대로 적재하면 뒤엣것이 앞엣것을 덮어써 조용히 사라집니다."
        )
    print(f"  detection_id 고유성 확인: {total:,}건 OK")

    # 라벨 분포
    from collections import Counter
    labels = Counter(str(r.get("class_name", "?")).lower() for r in records)
    print("  라벨 분포:", dict(labels.most_common(8)))

    person_labels = {str(x).lower() for x in cfg.person_labels}
    n_person = sum(n for l, n in labels.items() if l in person_labels)
    print(f"  person {n_person:,} / object {total - n_person:,}")

    if args.dry_run:
        print("\n--dry-run: 여기서 종료합니다.")
        print("샘플 레코드:")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))
        return 0

    # ── 체크포인트 ──
    ckpt_dir = Path(args.checkpoint) if args.checkpoint else (
        ROOT / "data" / f"embedding_checkpoint_{source.replace('/', '_')}"
    )
    state_path = ckpt_dir / "state.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    import hashlib

    def sha256_file(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    run_info = {
        "pipeline_sha256": sha256_file(CONFIG_PATH),
        "source": source,
        "total": total,
        "batch_size": args.batch_size,
        "bbox_space": bbox_space,
    }

    if args.fresh and state_path.exists():
        state_path.unlink()
        print(f"  체크포인트 삭제: {state_path}")

    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("run_info") != run_info:
            raise RuntimeError(
                f"기존 체크포인트가 현재 설정과 다릅니다: {state_path}\n"
                "  pipeline.yaml 이 바뀌었거나 스캔 결과 건수가 달라졌습니다.\n"
                "  처음부터 다시 하려면 --fresh 를 쓰세요."
            )
        start_index = int(state.get("next_index", 0))
    else:
        state = {"run_info": run_info, "next_index": 0}
        start_index = 0

    if start_index >= total:
        print(f"\n이미 완료된 소스입니다 ({total:,}건). --fresh 로 재적재 가능.")
        print_source_stats(store, cfg)
        return 0

    # 안전 재생: 마지막 배치를 다시 돌린다 (deterministic ID 라 덮어쓰기 = 안전)
    if start_index > 0:
        replay_from = max(0, start_index - args.batch_size)
        print(f"\nresume: checkpoint={start_index:,} · safety replay from={replay_from:,}")
        start_index = replay_from

    # ── Qdrant / 모델 ──
    print("\n" + "=" * 70)
    print("QDRANT")
    print("=" * 70)
    store.ensure_collection(recreate=False)   # 삭제 경로 없음
    print("ready:", cfg.collection)
    before_count = store.client.get_collection(cfg.collection).points_count
    print(f"현재 points: {before_count:,}")

    _, fmt = from_rfdetr(records[:1], load_mode="path")
    print("input_format:", fmt)

    print("\n" + "=" * 70)
    print("LOAD EMBEDDERS")
    print("=" * 70)
    registry = EmbedderRegistry(cfg)
    router = Router(cfg, registry, input_format=fmt)
    print("Router ready")

    # ── 적재 루프 ──
    print("\n" + "=" * 70)
    print(f"INGEST  {source}  ({total - start_index:,} 건 남음)")
    print("=" * 70)

    run_start = time.time()
    processed = 0
    uploaded_total = 0

    for start in range(start_index, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch_records = records[start:end]
        t0 = time.time()

        detections, batch_fmt = from_rfdetr(
            batch_records,
            load_mode="path",
            track_id_key="track_num" if args.tracks else None,
        )

        if batch_fmt != fmt:
            raise RuntimeError(f"input_format 변경 감지: {fmt} -> {batch_fmt}")
        if len(detections) != len(batch_records):
            raise RuntimeError(
                f"from_rfdetr 변환 개수 불일치: "
                f"{len(batch_records)} -> {len(detections)}"
            )

        inject_extra(detections, batch_records, bbox_space)

        vectors = router.embed(detections)

        if len(vectors) != len(detections):
            raise RuntimeError(
                f"vector 개수 불일치: {len(detections)} != {len(vectors)}"
            )

        uploaded_total += store.upsert(
            detections, vectors, batch_size=args.batch_size
        )

        # Qdrant 성공 후에만 기록
        atomic_write_json(state_path, {"run_info": run_info, "next_index": end})

        processed += end - start
        elapsed = time.time() - run_start
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - end) / rate if rate > 0 else 0

        print(
            f"  {end:>8,}/{total:,}  {100*end/total:6.2f}%  "
            f"{rate:6.1f}/s  batch {time.time()-t0:5.1f}s  ETA {format_seconds(eta)}"
        )

    after_count = store.client.get_collection(cfg.collection).points_count

    print("\n" + "=" * 70)
    print("INGEST COMPLETED")
    print("=" * 70)
    print(f"source      : {source}")
    print(f"처리        : {total:,}")
    print(f"upsert 반환 : {uploaded_total:,}")
    print(f"points      : {before_count:,} -> {after_count:,}  "
          f"(+{after_count - before_count:,})")
    print(f"소요        : {format_seconds(time.time() - run_start)}")

    if after_count - before_count < total * 0.9 and before_count > 0:
        print(
            "\n주의: 증가분이 처리 건수보다 훨씬 적습니다. deterministic ID 로 "
            "기존 point 를 덮어썼을 수 있습니다 (재적재라면 정상)."
        )

    print_source_stats(store, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
