"""eval/track_gt_eval.py — 추적·스티칭 준정답 시트(sheet) + IDF1/HOTA/ID switch 평가(eval)  (P6, 기준표 §2)

준정답(semi-GT) 방식: 영상 파이프라인 출력(outputs/processed_videos/<영상>/final_routed_tracks.json)의 박스 위치는 그대로 두고,
사람은 "어느 박스 묶음(구간)이 같은 사람인가"(gt_id) 만 라벨한다. 따라서 이 평가는 검출 품질이 아니라 연관(association) 품질 —
추적기의 ID 유지, 스티처의 병합 — 을 잰다. 같은 검출 위에서 추적기·스티처를 바꾼 출력을 GT 박스와 IoU≥0.5 로 대응시켜 비교할 수 있다.
(검출기까지 바꾸면 기존 검출기가 놓친 사람은 GT 에 없어 새 검출은 FP 가 된다 — 검출기 비교는 eval/detect_eval_prw.py 의 PRW 박스 GT 로.)

라벨 단위 = 구간(segment): 추적기 id(track_id) 는 시간이 지나면 다른 사람에게 재사용되므로(실측: 한 id 가 5 명) 그대로 쓸 수 없다.
같은 track_id 가 끊기지 않는(프레임 간격 ≤ max_gap) 연속 구간을 하나의 단위("<track_id>.<k>")로 잡는다 — 스티처 출력(long id)과 무관한
추적기만의 단위라서, 스티처가 창 경계에서 갈라 놓은 사람(SUSHI 는 512 프레임 창을 독립 처리)은 한 구간이 여러 long id 를 걸치게 되고 시트에 그렇게 표시된다.

sheet : 영상별 <gt-dir>/<영상>/{proposals.json(manifest 포함), boxes.jsonl, sheet.html}. 브라우저에서 라벨하고 "labels.json 내려받기" → 같은 폴더.
        항목마다 "검토" 체크가 있고, 검토된 항목만 사람 정답으로 쓴다(제안값을 그대로 둔 항목은 정답이 아니다).
eval  : GT = boxes.jsonl + labels.json(검토 항목). 검토율(coverage) < --min-coverage 이면 pseudo(제안값 = 정답 가정; 원장 기록 안 함, 이름에 __pseudo).
        예측 세 가지: raw = 추적기 id 그대로(재사용 포함) · before = 구간(추적기 tracklet) · after = 긴 트랙 id(스티처 출력).
        지표(TrackEval 정의): IDF1/IDP/IDR(Identity 전역 매칭), HOTA/DetA/AssA(α 0.05~0.95, 프레임당 매칭 1회 후 α 필터), CLEAR(MOTA, IDSW = 마지막 대응과 다른 id 로 매칭,
        fragments = 매칭 구간의 끊김 수), 보조 진단 splits(한 GT 가 대응한 예측 id 수 − 1 의 합) · over_merges(둘 이상 GT 에 대응한 예측 id 수),
        pred_dup_boxes(한 프레임에 같은 예측 id 두 개 = 동시 존재 과병합 신호). ignore 라벨은 MOTChallenge 방식(정상+ignore GT 를 함께 일대일 매칭 → ignore 에 붙은 예측 제거).
        산출물: <output-dir>/<name>/track_eval.json + report.md, 원장 stage=track (metrics = after, *_before / *_raw 병기, idsw_ratio = after/before).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import gt_sheet as S  # noqa: E402

PRODUCER = "track_gt_eval"
KIND = "track_labels"
DEFAULT_GT_DIR = ROOT / "eval" / "gt" / "tracks"
DEFAULT_PROCESSED = ROOT / "outputs" / "processed_videos"
DEFAULT_VIDEOS_ROOT = ROOT / "data" / "videos"
DEFAULT_OUT = ROOT / "eval" / "results" / "track_gt"
DEFAULT_MAX_GAP = 30
HOTA_ALPHAS = [round(0.05 * i, 2) for i in range(1, 20)]
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv")
VARIANTS = (("raw", "raw"), ("before", "segment"), ("after", "long"))
EPS = float(np.finfo("float").eps)


# ---------------------------------------------------------------- 입력
def load_routed(processed_root: Path, video: str) -> Tuple[List[Dict[str, Any]], str]:
    """final_routed → stitched → tracks.jsonl 순으로 읽는다. 반환 (rows, source_file)."""
    d = processed_root / video
    for name in ("final_routed_tracks.json", "stitched_tracks.json"):
        p = d / name
        if p.is_file():
            data = S.load_json(p, [])
            rows = data if isinstance(data, list) else (data.get("tracks") or [])
            return [r for r in rows if isinstance(r, dict)], name
    p = d / "tracks.jsonl"
    if p.is_file():
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows, "tracks.jsonl"
    raise SystemExit(f"추적 출력이 없습니다: {d} (final_routed_tracks.json / stitched_tracks.json / tracks.jsonl)")


def short_id(row: Dict[str, Any]) -> int:
    return int(row.get("short_track_id", row.get("track_id")))


def has_long_id(rows: Sequence[Dict[str, Any]]) -> bool:
    return any(r.get("long_track_id") is not None for r in rows)


def long_id(row: Dict[str, Any]) -> int:
    v = row.get("long_track_id")
    if v is None:
        v = row.get("short_track_id", row.get("track_id"))
    return int(v)


def route_of(row: Dict[str, Any]) -> str:
    return str(row.get("final_db_route") or row.get("db_route") or row.get("route") or "person")


def _best_rows(rows: Sequence[Dict[str, Any]]) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """track_id → frame → 행 (같은 프레임에 같은 id 가 둘이면 신뢰도 높은 것)."""
    per: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if "bbox" not in r or r.get("frame_idx") is None:
            continue
        tid = short_id(r)
        fi = int(r["frame_idx"])
        prev = per[tid].get(fi)
        if prev is None or float(r.get("confidence") or 0) > float(prev.get("confidence") or 0):
            per[tid][fi] = r
    return per


def segment_rows(rows: Sequence[Dict[str, Any]], max_gap: int = DEFAULT_MAX_GAP) -> Dict[str, Dict[str, Any]]:
    """구간(segment): 같은 track_id · 프레임 간격 ≤ max_gap 인 연속 구간 = "<track_id>.<k>". 스티처의 long id 와 무관하게 자른다.
    long_id = 구간 안 다수 long id(시트 카드 배치·기본 제안용), long_ids = 걸친 long id 전부(둘 이상이면 스티처가 갈라 놓은 것)."""
    per = _best_rows(rows)
    out: Dict[str, Dict[str, Any]] = {}
    for tid in sorted(per):
        k = 0
        cur: Optional[Dict[str, Any]] = None
        for fi in sorted(per[tid]):
            r = per[tid][fi]
            if cur is None or fi - cur["last_frame"] > max_gap:
                sid = f"{tid}.{k}"
                k += 1
                cur = out[sid] = {"segment_id": sid, "track_id": tid, "route_votes": Counter(), "long_votes": Counter(), "class_name": r.get("class_name"),
                                  "frames": [], "boxes": [], "secs": [], "last_frame": fi}
            cur["frames"].append(fi)
            cur["boxes"].append([float(x) for x in r["bbox"][:4]])
            cur["secs"].append(r.get("timestamp_sec"))
            cur["long_votes"][long_id(r)] += 1
            cur["route_votes"][route_of(r)] += 1
            cur["last_frame"] = fi
    for t in out.values():
        hs = [b[3] - b[1] for b in t["boxes"]]
        t["long_id"] = t["long_votes"].most_common(1)[0][0]
        t["long_ids"] = sorted(t["long_votes"])
        t["route"] = "person" if t["route_votes"].get("person") else t["route_votes"].most_common(1)[0][0]
        t.update({"n_frames": len(t["frames"]), "first_frame": t["frames"][0], "last_frame": t["frames"][-1], "first_sec": t["secs"][0],
                  "last_sec": t["secs"][-1], "median_h": float(np.median(hs)) if hs else 0.0})
        del t["long_votes"], t["route_votes"]
    return out


def find_video_file(processed_root: Path, videos_root: Optional[Path], video: str) -> Optional[Path]:
    st = S.load_json(processed_root / video / "status.json", {}) or {}
    cand = st.get("video")
    if cand and Path(cand).is_file():
        return Path(cand)
    if videos_root is not None:
        for suf in VIDEO_SUFFIXES:
            p = videos_root / f"{video}{suf}"
            if p.is_file():
                return p
        hits = sorted(videos_root.glob(f"{video}.*")) if videos_root.is_dir() else []
        if hits:
            return hits[0]
    return None


def sample_indices(frames: Sequence[int], n: int) -> List[int]:
    if not frames:
        return []
    if len(frames) <= n:
        return list(frames)
    pos = np.linspace(0, len(frames) - 1, n)
    return [frames[int(round(p))] for p in pos]


def read_frames(video_path: Path, wanted: Iterable[int]) -> Dict[int, np.ndarray]:
    """필요한 프레임만 순차 디코딩으로 모은다 (seek 반복보다 빠르고 안정적)."""
    import cv2
    want = sorted(set(int(i) for i in wanted))
    out: Dict[int, np.ndarray] = {}
    if not want:
        return out
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return out
    try:
        target = 0
        idx = 0
        last = want[-1]
        while idx <= last and target < len(want):
            ok = cap.grab()
            if not ok:
                break
            if idx == want[target]:
                ok, frame = cap.retrieve()
                if ok:
                    out[idx] = frame
                target += 1
            idx += 1
    finally:
        cap.release()
    return out


# ---------------------------------------------------------------- sheet
INTRO = """
<b>추적 준정답 시트</b> — 각 카드는 스티처가 만든 <b>긴 트랙</b>, 그 안의 행은 추적기가 만든 <b>구간</b>(끊기지 않은 박스 묶음)입니다. 박스 위치는 건드리지 않고 "누구인가"만 정합니다.
<ol>
<li><b>같은 사람</b>이면 같은 <code>gt_id</code>. 기본값은 긴 트랙 id (P12 등). 다른 카드의 사람과 같으면 그 카드의 id 를 적어 주세요 (스티처가 놓친 연결).</li>
<li>한 카드 안의 행이 <b>다른 사람</b>이면 그 행의 gt_id 를 바꿉니다 (예: P12b). 스티처의 과병합이 여기서 잡힙니다.</li>
<li><span class="tag warn">창 경계</span> 표시는 추적기는 한 사람으로 이었는데 스티처가 긴 트랙을 나눈 구간입니다. 같은 사람이 맞으면 그대로(기본값), 아니면 바뀌는 프레임을 적으세요.</li>
<li>사람이 아니거나 너무 작아 판단 불가면 상태를 <b>ignore</b>. (평가에서 그 박스에 붙는 예측은 정답도 오답도 아닌 것으로 뺍니다.)</li>
<li>한 행(구간) 안에서 사람이 바뀌면 <b>바뀌는 프레임</b>과 그 뒤의 gt_id 를 적습니다.</li>
<li>행을 확인했으면 <b>검토</b>에 체크(필드를 고치면 자동 체크). 검토하지 않은 행은 정답으로 쓰이지 않습니다. 끝나면 <b>labels.json 내려받기</b> → 이 시트와 같은 폴더.</li>
</ol>
"""


def build_sheet(video: str, segs: Dict[str, Dict[str, Any]], frames: Dict[int, np.ndarray], samples: int, thumb_h: int,
                source_file: str, video_path: Optional[Path], manifest: str) -> Tuple[str, List[Dict[str, Any]]]:
    by_long: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for t in segs.values():
        by_long[t["long_id"]].append(t)
    proposals: List[Dict[str, Any]] = []
    main_cards: List[str] = []
    minor_cards: List[str] = []
    for lid in sorted(by_long, key=lambda k: min(t["first_frame"] for t in by_long[k])):
        parts = sorted(by_long[lid], key=lambda t: t["first_frame"])
        routes = {t["route"] for t in parts}
        route = "person" if "person" in routes else (sorted(routes)[0] if routes else "?")
        default_gt = f"P{lid}" if route == "person" else ""
        default_status = "person" if route == "person" else "ignore"
        rows_html = []
        for t in parts:
            sid = t["segment_id"]
            thumbs = []
            for fi in sample_indices(t["frames"], samples):
                fr = frames.get(fi)
                uri = ""
                if fr is not None:
                    b = t["boxes"][t["frames"].index(fi)]
                    uri = S.thumb_b64_from_array(S.crop_with_margin(fr, b), height=thumb_h)
                thumbs.append(S.img_tag(uri, f"frame {fi}"))
            proposals.append({"segment_id": sid, "track_id": t["track_id"], "long_id": lid, "long_ids": t["long_ids"], "route": t["route"],
                              "n_frames": t["n_frames"], "first_frame": t["first_frame"], "last_frame": t["last_frame"], "first_sec": t["first_sec"],
                              "last_sec": t["last_sec"], "median_h": round(t["median_h"], 1), "default_gt_id": default_gt,
                              "default_status": default_status})
            span = (f' <span class="tag warn">창 경계: 긴 트랙 {", ".join("L" + str(x) for x in t["long_ids"])}</span>' if len(t["long_ids"]) > 1 else "")
            rows_html.append(f"""
<div class="row" data-block="{S.esc(sid)}">
  <div class="thumbs">{''.join(thumbs)}</div>
  <div class="fields">
    <div class="meta">구간 <b>#{S.esc(sid)}</b> (추적기 id {t['track_id']}) <span class="tag {S.esc(t['route'])}">{S.esc(t['route'])}</span>{span} · 프레임 {t['first_frame']}–{t['last_frame']} ({S.fmt_time(t['first_sec'])}–{S.fmt_time(t['last_sec'])}) · 박스 {t['n_frames']} · 높이 {t['median_h']:.0f}px</div>
    <label>gt_id <input type="text" class="gt" data-item="{S.esc(sid)}" data-field="gt_id" value="{S.esc(default_gt)}" placeholder="P12"></label>
    <label>상태 <select data-item="{S.esc(sid)}" data-field="status">
      <option value="person"{' selected' if default_status == 'person' else ''}>person (사람)</option>
      <option value="ignore"{' selected' if default_status == 'ignore' else ''}>ignore (사람 아님/판단 불가)</option></select></label>
    <label>바뀌는 프레임 <input type="number" data-item="{S.esc(sid)}" data-field="split_frame" min="0" placeholder="선택"> → gt_id <input type="text" class="gt" data-item="{S.esc(sid)}" data-field="split_gt_id" placeholder="P13"></label>
    <label>메모 <input type="text" data-item="{S.esc(sid)}" data-field="note" style="width:180px"> {S.reviewed_box(sid)}</label>
  </div>
</div>""")
        card = f"""
<div class="card" data-group="L{lid}">
  <h2>긴 트랙 L{lid} <span class="tag {S.esc(route)}">{S.esc(route)}</span> <span class="meta">구간 {len(parts)} · {S.fmt_time(parts[0]['first_sec'])}–{S.fmt_time(parts[-1]['last_sec'])}</span></h2>
  {''.join(rows_html)}
</div>"""
        (main_cards if route == "person" else minor_cards).append(card)
    body = "".join(main_cards)
    if minor_cards:
        body += f"""
<details data-group="rejects"><summary>파이프라인이 사람이 아니라고 본 긴 트랙 {len(minor_cards)}개 (기본 ignore — 실제 사람이면 상태를 person 으로 바꾸고 gt_id 를 적으세요)</summary>
<p><button type="button" data-mark-all="1">이 묶음을 모두 확인함 (전부 ignore 가 맞음)</button></p>
{''.join(minor_cards)}
</details>"""
    intro = INTRO + f'<div class="meta">영상 {S.esc(video)} · 원본 {S.esc(source_file)} · 긴 트랙 {len(by_long)} · 구간 {len(segs)} · 영상 파일 {S.esc(S.display_name(video_path) or "(없음 — 썸네일 없음)")} · manifest {S.esc(manifest)}</div>'
    html = S.html_document(f"추적 준정답 — {video}", intro, body, kind=KIND, store_key=f"track:{video}", export_name="labels.json",
                           meta={"video": video, "source_file": source_file, "n_segments": len(segs), "n_long": len(by_long), "manifest": manifest})
    return html, proposals


def cmd_sheet(args: argparse.Namespace) -> Dict[str, Any]:
    processed_root = Path(args.processed_root).resolve()
    videos_root = Path(args.videos_root).resolve() if args.videos_root else None
    gt_dir = Path(args.gt_dir).resolve()
    max_gap = args.max_gap if args.max_gap is not None else DEFAULT_MAX_GAP
    summary = []
    for video in args.videos:
        rows, src = load_routed(processed_root, video)
        segs = segment_rows(rows, max_gap)
        if not segs:
            print(f"[sheet] {video}: 트랙 없음 — 건너뜀")
            continue
        video_path = find_video_file(processed_root, videos_root, video)
        wanted = [fi for t in segs.values() for fi in sample_indices(t["frames"], args.samples)]
        frames = read_frames(video_path, wanted) if video_path else {}
        if video_path and not frames:
            print(f"[sheet] {video}: 영상을 읽지 못했습니다 ({video_path}) — 썸네일 없이 생성")
        manifest = S.manifest_of(KIND, segs.keys(), {"video": video, "source_file": src, "max_gap": max_gap,
                                                     "boxes": sum(t["n_frames"] for t in segs.values())})
        html, proposals = build_sheet(video, segs, frames, args.samples, args.thumb_height, src, video_path, manifest)
        out = gt_dir / video
        S.write_text(out / "sheet.html", html)
        long_ids = {t["long_id"] for t in segs.values()}
        person_long = {t["long_id"] for t in segs.values() if t["route"] == "person"}
        n_span = sum(1 for t in segs.values() if len(t["long_ids"]) > 1)
        S.write_json(out / "proposals.json", {"kind": KIND, "manifest": manifest, "video": video, "source_file": src, "processed_root": str(processed_root),
                                              "max_gap": max_gap, "video_path": str(video_path) if video_path else None, "generated_at": S.now_iso(),
                                              "n_segments": len(segs), "n_tracker_ids": len({t["track_id"] for t in segs.values()}),
                                              "n_long": len(long_ids), "n_person_long": len(person_long), "n_segments_spanning_long": n_span,
                                              "short_tracks": proposals})
        with (out / "boxes.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for t in segs.values():
                for fi, b in zip(t["frames"], t["boxes"]):
                    f.write(json.dumps({"frame_idx": fi, "segment_id": t["segment_id"], "track_id": t["track_id"], "long_id": t["long_id"],
                                        "bbox": [round(x, 2) for x in b]}) + "\n")
        n_person = sum(1 for t in segs.values() if t["route"] == "person")
        print(f"[sheet] {video}: 긴 트랙 {len(long_ids)} (person {len(person_long)}) · 구간 {len(segs)} (person {n_person}, 추적기 id {len({t['track_id'] for t in segs.values()})}, "
              f"창 경계 걸침 {n_span}) · 썸네일 프레임 {len(frames)} · manifest {manifest} → {out / 'sheet.html'}")
        summary.append({"video": video, "sheet": str(out / "sheet.html"), "n_segments": len(segs), "n_person_segments": n_person, "manifest": manifest})
    print(f"RESULT_SUMMARY: {gt_dir}")
    return {"videos": summary, "gt_dir": str(gt_dir)}


# ---------------------------------------------------------------- GT / 예측 구성
def load_boxes(gt_video_dir: Path) -> Dict[str, Dict[str, Any]]:
    """boxes.jsonl → 구간 id → {long_id, track_id, frames, boxes}."""
    segs: Dict[str, Dict[str, Any]] = {}
    p = gt_video_dir / "boxes.jsonl"
    if not p.is_file():
        raise SystemExit(f"boxes.jsonl 이 없습니다 (먼저 sheet 를 만드세요): {p}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = str(r.get("segment_id") or r["track_id"])
            t = segs.setdefault(sid, {"long_id": int(r.get("long_id", r.get("track_id", 0))), "track_id": r.get("track_id"), "frames": [], "boxes": []})
            t["frames"].append(int(r["frame_idx"]))
            t["boxes"].append([float(x) for x in r["bbox"]])
    return segs


def build_gt(segs: Dict[str, Dict[str, Any]], proposals: Dict[str, Any], labels: Dict[str, Dict[str, Any]]) -> Tuple[Dict[int, List[Tuple[str, List[float]]]], Dict[int, List[List[float]]], Dict[str, Any]]:
    """구간 박스 + 라벨 → (frame → [(gt_id, bbox)], frame → [ignore bbox], info).
    검토(reviewed)된 항목만 라벨을 쓰고, 나머지는 제안값(pseudo). info.coverage = 검토 항목 / 전체."""
    defaults = {str(t.get("segment_id") or t.get("track_id")): t for t in (proposals.get("short_tracks") or [])}
    gt: Dict[int, List[Tuple[str, List[float]]]] = defaultdict(list)
    ignore: Dict[int, List[List[float]]] = defaultdict(list)
    n_person = n_ignore = n_split = n_reviewed = 0
    ids = set()
    seen: Dict[Tuple[int, str], int] = {}
    dup = 0
    for sid, t in segs.items():
        d = defaults.get(str(sid), {})
        lab = labels.get(str(sid)) or {}
        reviewed = S.is_reviewed(lab)
        if not reviewed:
            lab = {}
        else:
            n_reviewed += 1
        status = str(lab.get("status") or d.get("default_status") or "person").strip().lower()
        gid = str(lab.get("gt_id") if lab.get("gt_id") not in (None, "") else d.get("default_gt_id") or f"P{t['long_id']}").strip()
        if status != "person" or not gid:
            n_ignore += 1
            for fi, b in zip(t["frames"], t["boxes"]):
                ignore[fi].append(b)
            continue
        split_frame = lab.get("split_frame")
        split_gid = str(lab.get("split_gt_id") or "").strip()
        try:
            split_frame = int(split_frame) if split_frame not in (None, "") else None
        except (TypeError, ValueError):
            split_frame = None
        if split_frame is not None and split_gid:
            n_split += 1
        n_person += 1
        for fi, b in zip(t["frames"], t["boxes"]):
            g = split_gid if (split_frame is not None and split_gid and fi >= split_frame) else gid
            ids.add(g)
            if (fi, g) in seen:                        # 같은 프레임에 같은 gt_id 두 박스 = 라벨 오류 (동시 존재하는 두 사람을 한 id 로)
                dup += 1
                continue
            seen[(fi, g)] = 1
            gt[fi].append((g, b))
    total = len(segs)
    info = {"gt_segments": n_person, "ignored_segments": n_ignore, "splits_labeled": n_split, "gt_ids": len(ids),
            "gt_boxes": sum(len(v) for v in gt.values()), "gt_dup_boxes": dup, "segments": total, "reviewed_segments": n_reviewed,
            "coverage": round(n_reviewed / total, 4) if total else 0.0, "labeled": n_reviewed > 0}
    return gt, ignore, info


def pred_from_rows(rows: Sequence[Dict[str, Any]], key: str, max_gap: int = DEFAULT_MAX_GAP) -> Tuple[Dict[int, List[Tuple[str, List[float]]]], int]:
    """예측 출력 → (frame → [(pred_id, bbox)], 중복 박스 수). key = raw(추적기 id) | segment(구간) | long(긴 트랙 id).
    같은 프레임에 같은 id 가 둘이면 신뢰도 높은 것만 남기고 개수를 센다(동시 존재 과병합 신호)."""
    out: Dict[int, List[Tuple[str, List[float]]]] = defaultdict(list)
    if key == "segment":
        for t in segment_rows(rows, max_gap).values():
            for fi, b in zip(t["frames"], t["boxes"]):
                out[fi].append((t["segment_id"], b))
        return out, 0
    best: Dict[Tuple[int, int], Dict[str, Any]] = {}
    dup = 0
    for r in rows:
        if "bbox" not in r or r.get("frame_idx") is None:
            continue
        pid = short_id(r) if key == "raw" else long_id(r)
        fi = int(r["frame_idx"])
        prev = best.get((fi, pid))
        if prev is None:
            best[(fi, pid)] = r
        else:
            dup += 1
            if float(r.get("confidence") or 0) > float(prev.get("confidence") or 0):
                best[(fi, pid)] = r
    for (fi, pid), r in best.items():
        out[fi].append((str(pid), [float(x) for x in r["bbox"][:4]]))
    return out, dup


def load_pred_file(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        data = S.load_json(path, [])
        rows = data if isinstance(data, list) else (data.get("tracks") or [])
    return rows


# ---------------------------------------------------------------- 지표
def iou_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    A = np.asarray(a, dtype=np.float64)[:, None, :]
    B = np.asarray(b, dtype=np.float64)[None, :, :]
    ix1 = np.maximum(A[..., 0], B[..., 0])
    iy1 = np.maximum(A[..., 1], B[..., 1])
    ix2 = np.minimum(A[..., 2], B[..., 2])
    iy2 = np.minimum(A[..., 3], B[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(A[..., 2] - A[..., 0], 0, None) * np.clip(A[..., 3] - A[..., 1], 0, None)
    area_b = np.clip(B[..., 2] - B[..., 0], 0, None) * np.clip(B[..., 3] - B[..., 1], 0, None)
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _assign_max(score: np.ndarray) -> List[Tuple[int, int]]:
    """점수 최대 매칭 (Hungarian). 점수 0 이하 쌍은 제외."""
    if score.size == 0:
        return []
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-score)
    return [(int(i), int(j)) for i, j in zip(r, c) if score[i, j] > EPS]


def remove_ignored(pred: Dict[Any, List[Tuple[str, List[float]]]], gt: Dict[Any, List[Tuple[str, List[float]]]],
                   ignore: Dict[Any, List[List[float]]], thr: float = 0.5) -> Tuple[Dict[Any, List[Tuple[str, List[float]]]], int]:
    """MOTChallenge 전처리: 프레임마다 (정상 GT + ignore GT) 와 예측을 함께 일대일 매칭(IoU ≥ thr)한 뒤 ignore 에 붙은 예측만 뺀다.
    정상 GT 에 더 잘 맞는 예측은 ignore 가 근처에 있어도 살아남는다. 반환 (pred, 제거 수)."""
    out: Dict[Any, List[Tuple[str, List[float]]]] = {}
    removed = 0
    for fi, items in pred.items():
        ig = ignore.get(fi)
        if not ig or not items:
            out[fi] = list(items)
            continue
        normal = [b for _, b in gt.get(fi, [])]
        all_gt = normal + list(ig)
        m = iou_matrix(all_gt, [b for _, b in items])
        m = np.where(m >= thr - EPS, m, 0.0)
        drop = set()
        for gi, pj in _assign_max(m):
            if gi >= len(normal):
                drop.add(pj)
        keep = [it for k, it in enumerate(items) if k not in drop]
        removed += len(drop)
        out[fi] = keep
    return out, removed


class Frame:
    __slots__ = ("gt_ids", "pr_ids", "iou")

    def __init__(self, gt_ids: List[str], pr_ids: List[str], iou: np.ndarray):
        self.gt_ids, self.pr_ids, self.iou = gt_ids, pr_ids, iou


def build_frames(gt: Dict[Any, List[Tuple[str, List[float]]]], pred: Dict[Any, List[Tuple[str, List[float]]]]) -> List[Frame]:
    keys = sorted(set(gt) | set(pred), key=lambda k: (str(k[0]), int(k[1])) if isinstance(k, tuple) else ("", int(k)))
    frames = []
    for k in keys:
        g = gt.get(k, [])
        p = pred.get(k, [])
        frames.append(Frame([x[0] for x in g], [x[0] for x in p], iou_matrix([x[1] for x in g], [x[1] for x in p])))
    return frames


def clear_metrics(frames: Sequence[Frame], thr: float = 0.5) -> Dict[str, Any]:
    """CLEAR (TrackEval clear.py): 직전 프레임의 대응을 우선(점수 +1000)하는 Hungarian 매칭 → TP/FP/FN, MOTA,
    IDSW = 마지막으로 대응했던 예측 id(프레임 무관)와 다른 id 로 매칭된 횟수, fragments = GT 별 매칭 run 수 − 1 (첫 획득 전 미검출은 제외).
    보조 진단: splits = Σ_GT(대응 예측 id 수 − 1), over_merges = 둘 이상 GT 에 대응한 예측 id 수."""
    prev_tracker: Dict[str, str] = {}            # gt id → 마지막으로 매칭된 pred id (IDSW 판정)
    prev_step: Dict[str, str] = {}               # gt id → 직전 프레임에 매칭된 pred id (매칭 우선권)
    matched_runs: Dict[str, List[bool]] = defaultdict(list)
    gt_to_preds: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pred_to_gts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tp = fp = fn = idsw = 0
    n_gt = n_pr = 0
    for f in frames:
        n_gt += len(f.gt_ids)
        n_pr += len(f.pr_ids)
        matched: List[Tuple[int, int]] = []
        if f.iou.size:
            sim = np.where(f.iou >= thr - EPS, f.iou, 0.0)
            bonus = np.zeros_like(sim)
            for gi, g in enumerate(f.gt_ids):
                p = prev_step.get(g)
                if p is not None:
                    for pj, q in enumerate(f.pr_ids):
                        if q == p:
                            bonus[gi, pj] = 1000.0
            score = np.where(sim > 0, bonus + sim, 0.0)
            matched = _assign_max(score)
        tp += len(matched)
        fp += len(f.pr_ids) - len(matched)
        fn += len(f.gt_ids) - len(matched)
        cur: Dict[str, str] = {}
        for gi, pj in matched:
            g, p = f.gt_ids[gi], f.pr_ids[pj]
            if g in prev_tracker and prev_tracker[g] != p:
                idsw += 1
            prev_tracker[g] = p
            cur[g] = p
            gt_to_preds[g][p] += 1
            pred_to_gts[p][g] += 1
        for g in f.gt_ids:
            matched_runs[g].append(g in cur)
        prev_step = cur
    frag = 0
    for g, flags in matched_runs.items():
        runs = 0
        prev = False
        for m in flags:
            if m and not prev:
                runs += 1
            prev = m
        frag += max(runs - 1, 0)
    splits = sum(len(v) - 1 for v in gt_to_preds.values() if len(v) > 1)
    split_ids = sum(1 for v in gt_to_preds.values() if len(v) > 1)
    over = sum(1 for v in pred_to_gts.values() if len(v) > 1)
    over_pairs = sum(len(v) - 1 for v in pred_to_gts.values() if len(v) > 1)
    mota = 1.0 - (fn + fp + idsw) / n_gt if n_gt else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "idsw": idsw, "fragments": frag, "splits": splits, "split_gt_ids": split_ids,
            "over_merges": over, "over_merge_pairs": over_pairs, "mota": round(mota, 4), "gt_boxes": n_gt, "pred_boxes": n_pr, "gt_ids": len(matched_runs)}


def idf1_metrics(frames: Sequence[Frame], thr: float = 0.5) -> Dict[str, float]:
    """Identity (Ristani 2016 / TrackEval identity.py): 쌍별 겹침 프레임 수로 gt id ↔ pred id 일대일 대응(더미 행·열) → IDTP/IDFP/IDFN."""
    ov: Dict[Tuple[str, str], int] = defaultdict(int)
    len_g: Dict[str, int] = defaultdict(int)
    len_p: Dict[str, int] = defaultdict(int)
    for f in frames:
        for g in f.gt_ids:
            len_g[g] += 1
        for p in f.pr_ids:
            len_p[p] += 1
        if f.iou.size:
            gi_idx, pj_idx = np.nonzero(f.iou >= thr - EPS)
            for gi, pj in zip(gi_idx, pj_idx):
                ov[(f.gt_ids[gi], f.pr_ids[pj])] += 1
    G, P = sorted(len_g), sorted(len_p)
    total_g, total_p = sum(len_g.values()), sum(len_p.values())
    if not G or not P:
        return {"idf1": 0.0, "idp": 0.0, "idr": 0.0, "idtp": 0, "idfp": total_p, "idfn": total_g}
    from scipy.optimize import linear_sum_assignment
    n = len(G) + len(P)
    cost = np.zeros((n, n), dtype=np.float64)
    for i, g in enumerate(G):
        for j, p in enumerate(P):
            o = ov.get((g, p), 0)
            cost[i, j] = (len_g[g] - o) + (len_p[p] - o)
    big = float(total_g + total_p + 1)
    cost[:len(G), len(P):] = big
    cost[len(G):, :len(P)] = big
    for i, g in enumerate(G):
        cost[i, len(P) + i] = len_g[g]
    for j, p in enumerate(P):
        cost[len(G) + j, j] = len_p[p]
    r, c = linear_sum_assignment(cost)
    idtp = 0
    for i, j in zip(r, c):
        if i < len(G) and j < len(P):
            idtp += ov.get((G[i], P[j]), 0)
    idfp, idfn = total_p - idtp, total_g - idtp
    idp = idtp / total_p if total_p else 0.0
    idr = idtp / total_g if total_g else 0.0
    idf1 = 2 * idtp / (total_g + total_p) if (total_g + total_p) else 0.0
    return {"idf1": round(idf1, 4), "idp": round(idp, 4), "idr": round(idr, 4), "idtp": int(idtp), "idfp": int(idfp), "idfn": int(idfn)}


def hota_metrics(frames: Sequence[Frame], alphas: Sequence[float] = HOTA_ALPHAS) -> Dict[str, float]:
    """HOTA (TrackEval hota.py): 전역 정렬 점수 = Σ_frames IoU/(행합+열합−IoU) 를 (|G|+|P|−그 합) 으로 나눈 것(α 무관),
    프레임마다 score = 정렬 점수 × IoU 로 Hungarian 1회, α 마다 IoU ≥ α 인 매칭만 TP → DetA·AssA(TPA/(TPA+FNA+FPA) 의 TP 가중 평균) → 기하평균, α 평균."""
    len_g: Dict[str, int] = defaultdict(int)
    len_p: Dict[str, int] = defaultdict(int)
    pot: Dict[Tuple[str, str], float] = defaultdict(float)
    for f in frames:
        for g in f.gt_ids:
            len_g[g] += 1
        for p in f.pr_ids:
            len_p[p] += 1
        if f.iou.size:
            denom = f.iou.sum(0)[None, :] + f.iou.sum(1)[:, None] - f.iou
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(denom > 0, f.iou / denom, 0.0)
            gi_idx, pj_idx = np.nonzero(sim > 0)
            for gi, pj in zip(gi_idx, pj_idx):
                pot[(f.gt_ids[gi], f.pr_ids[pj])] += float(sim[gi, pj])
    total_g, total_p = sum(len_g.values()), sum(len_p.values())
    if total_g == 0 or total_p == 0:
        return {"hota": 0.0, "deta": 0.0, "assa": 0.0}
    matches: Dict[float, Dict[Tuple[str, str], int]] = {a: defaultdict(int) for a in alphas}
    tps: Dict[float, int] = {a: 0 for a in alphas}
    for f in frames:
        if not f.iou.size:
            continue
        score = np.zeros_like(f.iou)
        for gi, g in enumerate(f.gt_ids):
            for pj, p in enumerate(f.pr_ids):
                pm = pot.get((g, p), 0.0)
                score[gi, pj] = (pm / (len_g[g] + len_p[p] - pm)) * f.iou[gi, pj] if pm > 0 else 0.0
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-score)
        for gi, pj in zip(r, c):
            iou = f.iou[gi, pj]
            for a in alphas:
                if iou >= a - EPS:
                    matches[a][(f.gt_ids[gi], f.pr_ids[pj])] += 1
                    tps[a] += 1
    hotas, detas, assas = [], [], []
    for a in alphas:
        tp = tps[a]
        fn, fp = total_g - tp, total_p - tp
        deta = tp / (tp + fn + fp) if (tp + fn + fp) else 0.0
        assa = 0.0
        if tp:
            s = 0.0
            for (g, p), c in matches[a].items():
                s += c * (c / (len_g[g] + len_p[p] - c))
            assa = s / tp
        hotas.append((deta * assa) ** 0.5)
        detas.append(deta)
        assas.append(assa)
    return {"hota": round(float(np.mean(hotas)), 4), "deta": round(float(np.mean(detas)), 4), "assa": round(float(np.mean(assas)), 4)}


def evaluate(gt: Dict[Any, List[Tuple[str, List[float]]]], pred: Dict[Any, List[Tuple[str, List[float]]]],
             ignore: Dict[Any, List[List[float]]], thr: float = 0.5) -> Dict[str, Any]:
    pred2, removed = remove_ignored(pred, gt, ignore, thr)
    frames = build_frames(gt, pred2)
    m: Dict[str, Any] = {}
    m.update(idf1_metrics(frames, thr))
    m.update(hota_metrics(frames))
    m.update(clear_metrics(frames, thr))
    m["pred_ids"] = len({p for f in frames for p in f.pr_ids})
    m["ignored_pred_boxes"] = removed
    return m


# ---------------------------------------------------------------- eval
def namespaced(video: str, d: Dict[int, list], with_ids: bool) -> Dict[Tuple[str, int], list]:
    out = {}
    for fi, items in d.items():
        out[(video, int(fi))] = [(f"{video}/{x[0]}", x[1]) for x in items] if with_ids else list(items)
    return out


def cmd_eval(args: argparse.Namespace) -> Dict[str, Any]:
    gt_dir = Path(args.gt_dir).resolve()
    processed_root = Path(args.processed_root).resolve() if args.processed_root else None
    videos = list(args.videos) if args.videos else sorted(p.name for p in gt_dir.iterdir() if (p / "boxes.jsonl").is_file())
    if not videos:
        raise SystemExit(f"평가할 영상이 없습니다: {gt_dir} (먼저 sheet 를 만드세요)")
    started = time.time()
    all_gt: Dict[Any, list] = {}
    all_ig: Dict[Any, list] = {}
    preds: Dict[str, Dict[Any, list]] = {v: {} for v, _ in VARIANTS}
    per_video: Dict[str, Any] = {}
    gt_info_total: Dict[str, int] = defaultdict(int)
    pred_sources, coverages, gaps, inputs = [], [], {}, []
    pred_rows_all = load_pred_file(Path(args.pred_file)) if args.pred_file else None
    if pred_rows_all is not None and len(videos) > 1 and any(not r.get("video") for r in pred_rows_all):
        raise SystemExit("--pred-file 로 여러 영상을 평가하려면 행마다 video 필드가 있어야 합니다")
    after_is_raw = False
    dup_total = 0
    for video in videos:
        vdir = gt_dir / video
        segs = load_boxes(vdir)
        proposals = S.load_json(vdir / "proposals.json", {}) or {}
        max_gap = args.max_gap if args.max_gap is not None else int(proposals.get("max_gap") or DEFAULT_MAX_GAP)
        gaps[video] = max_gap
        labels, lmeta = S.read_labels_meta(vdir / "labels.json")
        if labels:
            S.check_manifest(lmeta, proposals.get("manifest"), video, args.ignore_manifest)
        gt, ig, info = build_gt(segs, proposals, labels)
        coverages.append(info["coverage"])
        for k, v in info.items():
            if isinstance(v, int) and k != "coverage":
                gt_info_total[k] += v
        for fname in ("proposals.json", "boxes.jsonl", "labels.json"):
            if (vdir / fname).is_file():
                inputs.append({"role": f"gt:{video}:{fname}", "path": str(vdir / fname), "sha1": S.file_sha1(vdir / fname)})
        if pred_rows_all is not None:
            rows = [r for r in pred_rows_all if not r.get("video") or Path(str(r.get("video"))).stem == video]
            src = str(args.pred_file)
            if not has_long_id(rows):
                after_is_raw = True
        else:
            root = processed_root or Path(proposals.get("processed_root") or DEFAULT_PROCESSED)
            rows, src_name = load_routed(root, video)
            src = str(root / video / src_name)
        pred_sources.append(src)
        pv: Dict[str, Any] = {"gt": info, "pred_source": src, "max_gap": max_gap}
        for variant, key in VARIANTS:
            pr, dup = pred_from_rows(rows, key, max_gap)
            preds[variant].update(namespaced(video, pr, True))
            pv[variant] = evaluate(gt, pr, ig, args.iou)
            pv[variant]["pred_dup_boxes"] = dup
            if variant == "after":
                dup_total += dup
        per_video[video] = pv
        all_gt.update(namespaced(video, gt, True))
        all_ig.update(namespaced(video, ig, False))
        print(f"[eval] {video}: GT 사람 {info['gt_ids']} (구간 {info['gt_segments']}, ignore {info['ignored_segments']}, 검토 {info['reviewed_segments']}/{info['segments']}"
              f"{', 중복 gt 박스 ' + str(info['gt_dup_boxes']) if info['gt_dup_boxes'] else ''}) · raw IDF1 {pv['raw']['idf1']:.3f} IDSW {pv['raw']['idsw']} 과병합 {pv['raw']['over_merges']} · "
              f"before IDF1 {pv['before']['idf1']:.3f} IDSW {pv['before']['idsw']} · after IDF1 {pv['after']['idf1']:.3f} IDSW {pv['after']['idsw']} 과병합 {pv['after']['over_merges']}"
              f"{' 동시중복 ' + str(pv['after']['pred_dup_boxes']) if pv['after']['pred_dup_boxes'] else ''}")
    coverage = round(float(np.mean(coverages)), 4) if coverages else 0.0
    min_cov = float(args.min_coverage)
    labeled = coverage >= min_cov and all(c >= min_cov for c in coverages)
    partial = (not labeled) and coverage > 0
    pseudo = not labeled
    overall = {variant: evaluate(all_gt, preds[variant], all_ig, args.iou) for variant, _ in VARIANTS}
    after, before, raw = overall["after"], overall["before"], overall["raw"]
    metrics: Dict[str, Any] = {k: after[k] for k in ("idf1", "idp", "idr", "hota", "deta", "assa", "mota", "idsw", "fragments", "splits",
                                                     "over_merges", "tp", "fp", "fn")}
    for k in ("idf1", "hota", "mota", "idsw", "fragments", "splits", "over_merges"):
        metrics[f"{k}_before"] = before[k]
        metrics[f"{k}_raw"] = raw[k]
    metrics["idsw_ratio"] = round(after["idsw"] / before["idsw"], 4) if before["idsw"] else None
    metrics["pred_dup_boxes"] = dup_total
    metrics["gt_dup_boxes"] = int(gt_info_total.get("gt_dup_boxes", 0))
    metrics["coverage"] = coverage
    metrics["elapsed_sec"] = round(time.time() - started, 2)
    base = args.name or ("stitched" if not args.pred_file else Path(args.pred_file).stem)
    name = base + ("__pseudo" if pseudo else "")
    out = {"producer": PRODUCER, "generated_at": S.now_iso(), "name": name, "pseudo_gt": pseudo, "partial_gt": partial,
           "config": {"gt_dir": str(gt_dir), "processed_root": str(processed_root) if processed_root else None, "pred_file": args.pred_file,
                      "videos": videos, "iou": args.iou, "max_gap": gaps, "min_coverage": min_cov, "tracking_config": args.tracking_config,
                      "tracking_config_sha256": None, "after_is_raw": after_is_raw},
           "gt": {"videos": len(videos), "labeled_videos": sum(1 for c in coverages if c >= min_cov), "coverage": coverage, **dict(gt_info_total),
                  "protocol": "semi-GT: 검출 박스 고정, 사람 라벨 = 구간(track_id 연속 구간)별 gt_id(검토 항목만); IoU≥0.5; raw=추적기 id, before=구간, after=긴 트랙; ignore 는 MOTChallenge 방식"},
           "inputs": inputs, "pred_sources": pred_sources, "metrics": metrics, "raw": raw, "before": before, "after": after, "per_video": per_video}
    if args.tracking_config and Path(args.tracking_config).is_file():
        from bench import ledger
        out["config"]["tracking_config_sha256"] = ledger.file_sha256(args.tracking_config)
    out_dir = Path(args.output_dir).resolve() / name
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = S.write_json(out_dir / "track_eval.json", out)
    S.write_text(out_dir / "report.md", report_md(out))
    tag = " (pseudo GT — 검토된 라벨 없음, 원장 기록 안 함)" if (pseudo and not partial) else (f" (partial — 검토율 {coverage:.0%} < {min_cov:.0%}, pseudo 로 취급)" if partial else f" (검토율 {coverage:.0%})")
    print(f"\n[eval] {name}{tag}")
    for label, m in (("raw   (추적기 id)", raw), ("before(구간)", before), ("after (스티처)", after)):
        print(f"  {label:<16} IDF1 {m['idf1']:.4f} HOTA {m['hota']:.4f} MOTA {m['mota']:.4f} IDSW {m['idsw']} 단절 {m['fragments']} 갈라짐 {m['splits']} 과병합 {m['over_merges']}")
    print(f"  IDSW after/before = {metrics['idsw_ratio']} · after 동시 중복 박스 {dup_total} · GT 중복 박스 {metrics['gt_dup_boxes']}")
    if after_is_raw:
        print("  (예측 파일에 long_track_id 가 없어 after = raw 입니다 — before/after 차이는 스티처 효과가 아닙니다)")
    if not pseudo or args.record_pseudo:
        from bench import ledger
        ledger.record(lambda: [ledger.entry_from_track_result(out, report=json_path, command=args.command_line, versions=ledger.versions_info())],
                      args.ledger, args.no_ledger)
    print(f"RESULT_SUMMARY: {json_path}")
    return out


def report_md(out: Dict[str, Any]) -> str:
    r, b, a = out["raw"], out["before"], out["after"]
    g = out["gt"]
    lines = [f"# 추적·스티칭 평가 — {out['name']} ({out['generated_at']})", "",
             f"- GT: 영상 {g['videos']} (검토율 {g['coverage']:.0%}), 사람 id {g.get('gt_ids')}, 구간 {g.get('gt_segments')}, ignore {g.get('ignored_segments')}"
             + (" — **pseudo GT (제안값 = 정답 가정)**" if out["pseudo_gt"] else ""),
             f"- 예측: {', '.join(out['pred_sources'])}", "",
             "| 지표 | raw (추적기 id) | before (구간 = 추적기 tracklet) | after (긴 트랙 = 스티처 출력) |", "|---|---|---|---|"]
    for k, label in (("idf1", "IDF1"), ("idp", "IDP"), ("idr", "IDR"), ("hota", "HOTA"), ("deta", "DetA"), ("assa", "AssA"), ("mota", "MOTA"),
                     ("idsw", "ID switch"), ("fragments", "단절(fragments)"), ("splits", "갈라짐(splits)"), ("over_merges", "과병합(over-merges)"),
                     ("tp", "TP"), ("fp", "FP"), ("fn", "FN"), ("pred_ids", "예측 id 수")):
        lines.append(f"| {label} | {r.get(k)} | {b.get(k)} | {a.get(k)} |")
    lines.append(f"| IDSW after/before | | | {out['metrics'].get('idsw_ratio')} |")
    lines.append(f"| 동시 중복 박스 | | | {out['metrics'].get('pred_dup_boxes')} |")
    lines += ["", "## 영상별", "", "| 영상 | GT id | 검토 | raw IDF1 / IDSW / 과병합 | before IDF1 / IDSW / 과병합 | after IDF1 / IDSW / 과병합 |", "|---|---|---|---|---|---|"]
    for v, pv in out["per_video"].items():
        cells = " | ".join(f"{pv[x]['idf1']} / {pv[x]['idsw']} / {pv[x]['over_merges']}" for x in ("raw", "before", "after"))
        lines.append(f"| {v} | {pv['gt']['gt_ids']} | {pv['gt']['reviewed_segments']}/{pv['gt']['segments']} | {cells} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="추적·스티칭 준정답 시트 + IDF1/HOTA 평가 (P6)")
    p.add_argument("cmd", choices=["sheet", "eval"], help="sheet: 라벨링 시트 생성 / eval: labels.json 으로 평가")
    p.add_argument("--videos", nargs="*", default=None, help="영상 stem 목록 (eval 에서 비우면 gt-dir 의 전부)")
    p.add_argument("--processed-root", default=str(DEFAULT_PROCESSED), help="영상 파이프라인 출력 루트 (sheet: GT 박스 원본 / eval: 예측)")
    p.add_argument("--videos-root", default=str(DEFAULT_VIDEOS_ROOT), help="원본 영상 폴더 (썸네일용; status.json 의 경로가 없을 때)")
    p.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    p.add_argument("--samples", type=int, default=3, help="sheet: 구간당 썸네일 수")
    p.add_argument("--thumb-height", type=int, default=160)
    p.add_argument("--max-gap", type=int, default=None, help=f"구간을 끊는 프레임 간격 (기본 {DEFAULT_MAX_GAP}; eval 은 proposals.json 값)")
    p.add_argument("--pred-file", default=None, help="eval: 다른 추적기 출력 (jsonl/json, frame_idx·track_id·bbox[·long_track_id·video])")
    p.add_argument("--tracking-config", default=None, help="eval: 기록용 — 예측을 만든 pipeline_tracking*.yaml (sha256 을 남김)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--min-coverage", type=float, default=1.0, help="eval: 이 검토율 미만이면 pseudo 로 취급 (부분 라벨을 쓰려면 낮춘다)")
    p.add_argument("--ignore-manifest", action="store_true", help="eval: labels.json 의 manifest 가 시트와 달라도 강행")
    p.add_argument("--name", default=None, help="eval: 결과·원장 이름 (pseudo 면 __pseudo 가 붙음)")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--ledger", default=None, help="실행 원장 JSONL (기본 bench/ledger.jsonl; '' 또는 none 이면 기록 안 함)")
    p.add_argument("--no-ledger", action="store_true")
    p.add_argument("--record-pseudo", action="store_true", help="라벨 없는(pseudo/unlabeled) 결과도 --ledger 에 기록 (러너의 조각 원장용; 기본 원장에는 들어가지 않음)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    p = build_parser()
    args = p.parse_args(argv)
    args.command_line = list(argv) if argv is not None else sys.argv[1:]
    if args.cmd == "sheet":
        if not args.videos:
            p.error("sheet 는 --videos 가 필요합니다")
        return cmd_sheet(args)
    return cmd_eval(args)


if __name__ == "__main__":
    main()
