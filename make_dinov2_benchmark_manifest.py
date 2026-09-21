from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parent
CROP_ROOT = ROOT / "data" / "scvd_object_tracks_v1"
OUT = ROOT / "dinov2_benchmark_manifest.csv"

RANDOM_SEED = 42
MAX_GROUPS = 100
POSITIVES_PER_GROUP = 3
NEGATIVES_PER_GROUP = 10

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(RANDOM_SEED)


def get_track_dirs():
    tracks = []

    for source_dir in CROP_ROOT.iterdir():
        if not source_dir.is_dir():
            continue

        for track_dir in source_dir.iterdir():
            if not track_dir.is_dir():
                continue

            images = [
                p for p in track_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTS
            ]

            if len(images) >= 2:
                tracks.append(
                    {
                        "source": source_dir.name,
                        "track": track_dir.name,
                        "images": sorted(images),
                    }
                )

    return tracks


def main():
    tracks = get_track_dirs()

    print("usable tracks:", len(tracks))

    random.shuffle(tracks)

    rows = []
    used_groups = 0

    for track in tracks:
        if used_groups >= MAX_GROUPS:
            break

        imgs = track["images"]

        if len(imgs) < 2:
            continue

        query = imgs[0]

        positives = imgs[1:1 + POSITIVES_PER_GROUP]

        if not positives:
            continue

        # 다른 track에서 negative 선택
        candidate_neg_tracks = [
            t for t in tracks
            if t is not track
        ]

        random.shuffle(candidate_neg_tracks)

        negatives = []

        for neg_track in candidate_neg_tracks:
            if not neg_track["images"]:
                continue

            negatives.append(
                random.choice(
                    neg_track["images"]
                )
            )

            if len(negatives) >= NEGATIVES_PER_GROUP:
                break

        if len(negatives) < 1:
            continue

        gid = (
            f"{track['source']}/"
            f"{track['track']}"
        )

        rows.append(
            [gid, "query", str(query)]
        )

        for p in positives:
            rows.append(
                [gid, "positive", str(p)]
            )

        for n in negatives:
            rows.append(
                [gid, "negative", str(n)]
            )

        used_groups += 1

    with OUT.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "group_id",
                "role",
                "path",
            ]
        )

        writer.writerows(rows)

    print()
    print("groups :", used_groups)
    print("rows   :", len(rows))
    print("output :", OUT)


if __name__ == "__main__":
    main()