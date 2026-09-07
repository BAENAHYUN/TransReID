import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RF_DIR = ROOT_DIR / "src" / "RF-DETR"

SCVD_ROOT  = ROOT_DIR / "data" / "scvd" / "SCVD_converted"
TRAIN_ROOT = SCVD_ROOT / "Train"
TEST_ROOT  = SCVD_ROOT / "Test"

TEMP_ROOT       = ROOT_DIR / "data" / "scvd_person_temp_v1"
VIDEO_CROP_ROOT = ROOT_DIR / "data" / "scvd_track_crops_raw_v1"
TRACK_ROOT      = ROOT_DIR / "data" / "scvd_person_tracks_v1"
REJECT_ROOT     = ROOT_DIR / "data" / "scvd_person_rejected_v1"

STATE_ROOT = ROOT_DIR / "data" / "scvd_ingest_state_v2"
MANIFEST   = STATE_ROOT / "person_track_manifest.jsonl"

TARGET     = RF_DIR / "person_video_track_search_v4_optimized.py"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


class FilteredVideoDir:
    """Minimal VIDEO_DIR replacement used by the V4 tracker."""
    def __init__(self, paths):
        self.paths = list(paths)

    def rglob(self, pattern):
        suffix = pattern.lower().replace("*", "")
        return [p for p in self.paths if p.suffix.lower() == suffix]


class NullQdrant:
    """
    V4 tracker normally writes legacy track points to forensic_video_tracks_v4.
    SCVD V2 only needs track crops; final vectors are written later to the
    unified person collection, so legacy Qdrant writes are deliberately disabled.
    """
    def upsert(self, *args, **kwargs):
        return None


def load_module():
    if not TARGET.exists():
        raise FileNotFoundError(f"Missing tracker script: {TARGET}")

    spec = importlib.util.spec_from_file_location("person_v4_scvd_v2", TARGET)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_videos(split: str) -> list[Path]:
    """split: 'train' | 'test' | 'all'"""
    roots = []
    if split in ("train", "all"):
        if TRAIN_ROOT.exists():
            roots.append(TRAIN_ROOT)
    if split in ("test", "all"):
        if TEST_ROOT.exists():
            roots.append(TEST_ROOT)

    videos = []
    for root in roots:
        videos.extend(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
    return sorted(videos)


def rel(vp: Path) -> str:
    return vp.relative_to(SCVD_ROOT).as_posix()


def detect_split(vp: Path) -> str:
    """영상 경로에서 split 자동 감지."""
    try:
        parts = vp.relative_to(SCVD_ROOT).parts
        return parts[0] if parts else "Unknown"
    except Exception:
        return "Unknown"


def done_set() -> set[str]:
    done: set[str] = set()
    if not MANIFEST.exists():
        return done
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
            if d.get("status") == "ok" and d.get("video"):
                done.add(d["video"])
        except Exception:
            pass
    return done


def append_manifest(vp: Path, status: str = "ok"):
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    rec = {
        "video":          rel(vp),
        "status":         status,
        "source_dataset": "SCVD",
        "split":          detect_split(vp),
    }
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="SCVD 인물 트랙 크롭 추출")
    ap.add_argument("--max-videos", type=int, default=None,
                    help="처리 영상 수 제한 (디버그)")
    ap.add_argument("--split", choices=["train", "test", "all"], default="all",
                    help="처리할 split (기본: all = Train + Test)")
    ap.add_argument("--force", action="store_true",
                    help="manifest skip 무시하고 전체 재추출")
    args = ap.parse_args()

    videos = get_videos(args.split)
    done   = done_set()

    if not args.force:
        videos = [p for p in videos if rel(p) not in done]

    if args.max_videos:
        videos = videos[:args.max_videos]

    # 카운트
    split_counts: dict[str, int] = {}
    for vp in videos:
        s = detect_split(vp)
        split_counts[s] = split_counts.get(s, 0) + 1

    print("=" * 72)
    print(f"SCVD PERSON TRACK V2  (split={args.split})")
    print("=" * 72)
    print(f"처리 대상: {len(videos):,}개")
    for s, c in sorted(split_counts.items()):
        print(f"  {s}: {c:,}개")
    print(f"Track root: {TRACK_ROOT}")
    print("Legacy Qdrant: DISABLED")

    if not videos:
        print("Nothing to do.")
        return

    mod = load_module()

    mod.VIDEO_DIR       = FilteredVideoDir(videos)
    mod.TEMP_ROOT       = TEMP_ROOT
    mod.FRAME_DIR       = TEMP_ROOT / "frames"
    mod.CROP_DIR        = TEMP_ROOT / "crops"
    mod.VIDEO_CROP_ROOT = VIDEO_CROP_ROOT
    mod.TRACK_ROOT      = TRACK_ROOT
    mod.REJECT_ROOT     = REJECT_ROOT

    mod.get_client      = lambda: NullQdrant()
    mod.init_collection = lambda client: None

    try:
        mod.build_database(max_videos=None)
    except Exception:
        print(
            "\nTracking interrupted. No new V2 manifest entries were committed.\n"
            "A rerun is safe: deterministic output folders will be reused."
        )
        raise

    for vp in videos:
        append_manifest(vp, "ok")

    print("\nSCVD PERSON TRACK V2 COMPLETE")
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()
