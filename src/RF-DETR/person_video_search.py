import argparse
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from detect_rf import (
    load_detect_model,
    detect_and_crop,
)

from reid_clip import (
    load_reid_model,
    get_embedding as get_reid_embedding,
)

from face_insight import (
    load_face_model,
    detect_faces,
)

from siglip_semantic import (
    load_semantic_model,
    get_image_embedding as get_semantic_embedding,
)


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

VIDEO_DIR = ROOT_DIR / "data" / "videos_test"

TEMP_ROOT = ROOT_DIR / "data" / "video_person_temp"

FRAME_DIR = TEMP_ROOT / "frames"
CROP_DIR = TEMP_ROOT / "crops"

COLLECTION_NAME = "forensic_video_persons"

QDRANT_URL = "http://127.0.0.1:6333"

SAMPLE_INTERVAL_SEC = 1.0

REID_DIM = 1280
FACE_DIM = 512
SEMANTIC_DIM = 768

REID_WEIGHT = 0.5
FACE_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.2

SEARCH_TOP_K = 50


# ============================================================
# Utils
# ============================================================

def normalize(x):

    if x is None:
        return None

    x = np.asarray(
        x,
        dtype=np.float32,
    ).reshape(-1)

    n = np.linalg.norm(x)

    if n == 0:
        return x

    return x / n


def format_time(seconds):

    seconds = int(seconds)

    minute = seconds // 60
    second = seconds % 60

    return f"{minute:02d}:{second:02d}"


def reset_temp():

    if TEMP_ROOT.exists():

        shutil.rmtree(
            TEMP_ROOT
        )

    FRAME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Qdrant
# ============================================================

def get_client():

    return QdrantClient(
        url=QDRANT_URL
    )


def init_collection(
    client
):

    if client.collection_exists(
        COLLECTION_NAME
    ):

        print(
            f"Collection exists: "
            f"{COLLECTION_NAME}"
        )

        return

    client.create_collection(

        collection_name=
            COLLECTION_NAME,

        vectors_config={

            "reid":
                VectorParams(
                    size=REID_DIM,
                    distance=Distance.COSINE,
                ),

            "face":
                VectorParams(
                    size=FACE_DIM,
                    distance=Distance.COSINE,
                ),

            "semantic":
                VectorParams(
                    size=SEMANTIC_DIM,
                    distance=Distance.COSINE,
                ),
        },
    )

    print(
        f"Created collection: "
        f"{COLLECTION_NAME}"
    )


# ============================================================
# Models
# ============================================================

def load_all_models():

    print()
    print("=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    detect_model = (
        load_detect_model()
    )

    reid_model_data = (
        load_reid_model()
    )

    # 네 load_reid_model()이
    # (model, device)를 반환할 가능성 대응

    if isinstance(
        reid_model_data,
        tuple,
    ):

        reid_model = (
            reid_model_data[0]
        )

        reid_device = (
            reid_model_data[1]
        )

    else:

        reid_model = (
            reid_model_data
        )

        reid_device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    face_model = (
        load_face_model()
    )

    semantic_data = (
        load_semantic_model()
    )

    # load_semantic_model()
    # → model, processor, device 예상

    if (
        not isinstance(
            semantic_data,
            tuple,
        )
        or len(semantic_data) < 3
    ):

        raise RuntimeError(
            "load_semantic_model() "
            "must return "
            "(model, processor, device)"
        )

    semantic_model = (
        semantic_data[0]
    )

    semantic_processor = (
        semantic_data[1]
    )

    semantic_device = (
        semantic_data[2]
    )

    print()
    print("All models loaded.")

    return (
        detect_model,
        reid_model,
        reid_device,
        face_model,
        semantic_model,
        semantic_processor,
        semantic_device,
    )


# ============================================================
# Frame sampling
# ============================================================

def extract_frames(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 30.0

    frame_step = max(
        1,
        int(
            fps
            * SAMPLE_INTERVAL_SEC
        )
    )

    duration = (
        total_frames
        / fps
    )

    print(
        f"FPS          : "
        f"{fps:.2f}"
    )

    print(
        f"Total frames : "
        f"{total_frames}"
    )

    print(
        f"Duration     : "
        f"{duration:.2f} sec"
    )

    print(
        f"Frame step   : "
        f"{frame_step}"
    )

    frame_number = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if (
            frame_number
            % frame_step
            == 0
        ):

            timestamp = (
                frame_number
                / fps
            )

            yield (
                frame_number,
                timestamp,
                frame,
            )

        frame_number += 1

    cap.release()


# ============================================================
# RF-DETR
# ============================================================

def detect_frame_person_crops(
    detect_model,
    frame,
    frame_number,
):

    frame_path = (
        FRAME_DIR
        / f"frame_{frame_number:08d}.jpg"
    )

    cv2.imwrite(
        str(frame_path),
        frame,
    )

    prefix = (
        f"frame_{frame_number:08d}"
    )

    before = set(
        CROP_DIR.glob(
            "*"
        )
    )

    result = detect_and_crop(

        detect_model,

        str(
            frame_path
        ),

        output_dir=
            str(CROP_DIR),

        prefix=
            prefix,
    )

    # --------------------------------------------------------
    # 함수가 crop 경로를 직접 반환하는 경우
    # --------------------------------------------------------

    crop_paths = []

    if result is not None:

        if isinstance(
            result,
            (list, tuple),
        ):

            for item in result:

                if isinstance(
                    item,
                    (str, Path),
                ):

                    p = Path(item)

                    if p.exists():
                        crop_paths.append(
                            p
                        )

    # --------------------------------------------------------
    # 반환값이 없으면
    # output_dir에 새로 생성된 파일 탐색
    # --------------------------------------------------------

    if not crop_paths:

        after = set(
            CROP_DIR.glob(
                "*"
            )
        )

        new_files = (
            after - before
        )

        crop_paths = sorted(
            [
                p
                for p in new_files
                if p.suffix.lower()
                in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                ]
            ]
        )

    return crop_paths


# ============================================================
# Embeddings
# ============================================================

def extract_embeddings(
    crop_path,
    reid_model,
    reid_device,
    face_model,
    semantic_model,
    semantic_processor,
    semantic_device,
):

    # --------------------------------------------------------
    # CLIP-ReID
    # --------------------------------------------------------

    try:

        reid = (
            get_reid_embedding(

                reid_model,
                reid_device,
                str(crop_path),

                cam_id=0,
            )
        )

        reid = normalize(
            reid
        )

    except Exception as e:

        print(
            f"[REID ERROR] "
            f"{crop_path.name}: "
            f"{e}"
        )

        reid = None

    # --------------------------------------------------------
    # InsightFace
    # --------------------------------------------------------

    face = None

    try:

        faces = detect_faces(

            face_model,

            str(crop_path),

            det_thresh=0.5,
        )

        if faces:

            # detect_faces 결과가
            # ndarray 또는 객체일 수 있으므로 대응

            first = faces[0]

            if isinstance(
                first,
                np.ndarray,
            ):

                face = first

            elif hasattr(
                first,
                "normed_embedding",
            ):

                face = (
                    first
                    .normed_embedding
                )

            elif hasattr(
                first,
                "embedding",
            ):

                face = (
                    first.embedding
                )

            elif isinstance(
                first,
                dict,
            ):

                face = (
                    first.get(
                        "normed_embedding"
                    )
                    or first.get(
                        "embedding"
                    )
                )

            if face is not None:

                face = normalize(
                    face
                )

    except Exception as e:

        print(
            f"[FACE WARNING] "
            f"{crop_path.name}: "
            f"{e}"
        )

        face = None

    # --------------------------------------------------------
    # SigLIP2
    # --------------------------------------------------------

    try:

        semantic = (
            get_semantic_embedding(

                semantic_model,
                semantic_processor,
                semantic_device,

                str(crop_path),
            )
        )

        semantic = normalize(
            semantic
        )

    except Exception as e:

        print(
            f"[SEMANTIC ERROR] "
            f"{crop_path.name}: "
            f"{e}"
        )

        semantic = None

    # --------------------------------------------------------
    # Dimension check
    # --------------------------------------------------------

    if (
        reid is not None
        and reid.shape[0]
        != REID_DIM
    ):

        print(
            f"[REID DIM ERROR] "
            f"{reid.shape}"
        )

        reid = None

    if (
        face is not None
        and face.shape[0]
        != FACE_DIM
    ):

        print(
            f"[FACE DIM ERROR] "
            f"{face.shape}"
        )

        face = None

    if (
        semantic is not None
        and semantic.shape[0]
        != SEMANTIC_DIM
    ):

        print(
            f"[SEMANTIC DIM ERROR] "
            f"{semantic.shape}"
        )

        semantic = None

    return (
        reid,
        face,
        semantic,
    )


# ============================================================
# BUILD DATABASE
# ============================================================

def build_database():

    reset_temp()

    (
        detect_model,
        reid_model,
        reid_device,
        face_model,
        semantic_model,
        semantic_processor,
        semantic_device,
    ) = load_all_models()

    client = get_client()

    init_collection(
        client
    )

    video_files = []

    for extension in [
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.mkv",
        "*.wmv",
    ]:

        video_files.extend(
            VIDEO_DIR.rglob(
                extension
            )
        )

    video_files = sorted(
        video_files
    )

    print()
    print(
        f"Videos found: "
        f"{len(video_files)}"
    )

    total_frames = 0
    person_frames = 0
    no_person_frames = 0
    total_person_crops = 0
    total_points = 0
    total_faces = 0

    batch = []

    for (
        video_index,
        video_path,
    ) in enumerate(
        video_files,
        start=1,
    ):

        print()
        print("=" * 70)

        print(
            f"[{video_index}/"
            f"{len(video_files)}] "
            f"{video_path.name}"
        )

        print("=" * 70)

        for (
            frame_number,
            timestamp,
            frame,
        ) in extract_frames(
            video_path
        ):

            total_frames += 1

            try:

                crops = (
                    detect_frame_person_crops(

                        detect_model,
                        frame,
                        frame_number,
                    )
                )

            except Exception as e:

                print(
                    f"[DETECT ERROR] "
                    f"frame={frame_number}: "
                    f"{e}"
                )

                continue

            if not crops:

                no_person_frames += 1
                continue

            person_frames += 1

            print(
                f"Frame "
                f"{frame_number} "
                f"Time "
                f"{format_time(timestamp)} "
                f"Persons "
                f"{len(crops)}"
            )

            for (
                person_index,
                crop_path,
            ) in enumerate(
                crops
            ):

                total_person_crops += 1

                (
                    reid,
                    face,
                    semantic,
                ) = extract_embeddings(

                    crop_path,

                    reid_model,
                    reid_device,

                    face_model,

                    semantic_model,
                    semantic_processor,
                    semantic_device,
                )

                # 사람 검색의 기본축은 ReID
                if reid is None:

                    continue

                vectors = {
                    "reid":
                        reid.tolist(),
                }

                if face is not None:

                    vectors[
                        "face"
                    ] = (
                        face.tolist()
                    )

                    total_faces += 1

                if semantic is not None:

                    vectors[
                        "semantic"
                    ] = (
                        semantic.tolist()
                    )

                payload = {

                    "source":
                        "video_person",

                    "video_name":
                        video_path.name,

                    "video_path":
                        str(
                            video_path
                            .relative_to(
                                ROOT_DIR
                            )
                        ),

                    "frame_number":
                        int(
                            frame_number
                        ),

                    "timestamp":
                        float(
                            timestamp
                        ),

                    "timestamp_text":
                        format_time(
                            timestamp
                        ),

                    "person_index":
                        int(
                            person_index
                        ),

                    "crop_path":
                        str(
                            crop_path
                            .relative_to(
                                ROOT_DIR
                            )
                        ),

                    "has_face":
                        (
                            face is not None
                        ),
                }

                point = PointStruct(

                    id=str(
                        uuid.uuid4()
                    ),

                    vector=
                        vectors,

                    payload=
                        payload,
                )

                batch.append(
                    point
                )

                if len(batch) >= 32:

                    client.upsert(

                        collection_name=
                            COLLECTION_NAME,

                        points=
                            batch,

                        wait=True,
                    )

                    total_points += (
                        len(batch)
                    )

                    print(
                        f"Uploaded points: "
                        f"{total_points}"
                    )

                    batch.clear()

    if batch:

        client.upsert(

            collection_name=
                COLLECTION_NAME,

            points=
                batch,

            wait=True,
        )

        total_points += (
            len(batch)
        )

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Sampled frames : "
        f"{total_frames}"
    )

    print(
        f"Person frames  : "
        f"{person_frames}"
    )

    print(
        f"No-person      : "
        f"{no_person_frames}"
    )

    print(
        f"Person crops   : "
        f"{total_person_crops}"
    )

    print(
        f"Faces detected : "
        f"{total_faces}"
    )

    print(
        f"Qdrant points  : "
        f"{total_points}"
    )


# ============================================================
# Query
# ============================================================

def prepare_query(
    query_image,
    models,
):

    (
        detect_model,
        reid_model,
        reid_device,
        face_model,
        semantic_model,
        semantic_processor,
        semantic_device,
    ) = models

    query_image = Path(
        query_image
    )

    if not query_image.exists():

        raise FileNotFoundError(
            query_image
        )

    query_crop_dir = (
        TEMP_ROOT
        / "query"
    )

    query_crop_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = detect_and_crop(

        detect_model,

        str(
            query_image
        ),

        output_dir=
            str(
                query_crop_dir
            ),

        prefix=
            "query",
    )

    crops = []

    if isinstance(
        result,
        (list, tuple),
    ):

        for item in result:

            if isinstance(
                item,
                (str, Path),
            ):

                p = Path(item)

                if p.exists():

                    crops.append(
                        p
                    )

    if not crops:

        crops = sorted(
            [
                p
                for p
                in query_crop_dir.glob(
                    "*"
                )
                if p.suffix.lower()
                in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                ]
            ]
        )

    if not crops:

        raise RuntimeError(
            "No person detected "
            "in query image."
        )

    # 가장 큰 crop 사용
    best_crop = None
    best_area = -1

    for crop in crops:

        image = cv2.imread(
            str(crop)
        )

        if image is None:
            continue

        area = (
            image.shape[0]
            * image.shape[1]
        )

        if area > best_area:

            best_area = area
            best_crop = crop

    if best_crop is None:

        raise RuntimeError(
            "No valid query crop."
        )

    print()
    print(
        f"Query crop: "
        f"{best_crop}"
    )

    (
        reid,
        face,
        semantic,
    ) = extract_embeddings(

        best_crop,

        reid_model,
        reid_device,

        face_model,

        semantic_model,
        semantic_processor,
        semantic_device,
    )

    print(
        f"ReID     : "
        f"{None if reid is None else reid.shape}"
    )

    print(
        f"Face     : "
        f"{None if face is None else face.shape}"
    )

    print(
        f"Semantic : "
        f"{None if semantic is None else semantic.shape}"
    )

    return (
        reid,
        face,
        semantic,
    )


# ============================================================
# Search
# ============================================================

def search_vector(
    client,
    vector_name,
    vector,
):

    if vector is None:

        return []

    response = (
        client.query_points(

            collection_name=
                COLLECTION_NAME,

            using=
                vector_name,

            query=
                vector.tolist(),

            limit=
                SEARCH_TOP_K,

            with_payload=True,

            with_vectors=False,
        )
    )

    return (
        response.points
    )


def image_to_video(
    query_image,
    top_k=10,
):

    models = (
        load_all_models()
    )

    client = get_client()

    (
        query_reid,
        query_face,
        query_semantic,
    ) = prepare_query(

        query_image,
        models,
    )

    reid_results = (
        search_vector(
            client,
            "reid",
            query_reid,
        )
    )

    face_results = (
        search_vector(
            client,
            "face",
            query_face,
        )
    )

    semantic_results = (
        search_vector(
            client,
            "semantic",
            query_semantic,
        )
    )

    candidates = {}

    def add_results(
        results,
        key_name,
    ):

        for point in results:

            pid = str(
                point.id
            )

            if pid not in candidates:

                candidates[
                    pid
                ] = {

                    "point":
                        point,

                    "reid":
                        None,

                    "face":
                        None,

                    "semantic":
                        None,
                }

            candidates[
                pid
            ][
                key_name
            ] = float(
                point.score
            )

    add_results(
        reid_results,
        "reid",
    )

    add_results(
        face_results,
        "face",
    )

    add_results(
        semantic_results,
        "semantic",
    )

    fused = []

    for item in candidates.values():

        scores = []
        weights = []

        if item["reid"] is not None:

            scores.append(
                item["reid"]
            )

            weights.append(
                REID_WEIGHT
            )

        if (
            query_face is not None
            and item["face"]
            is not None
        ):

            scores.append(
                item["face"]
            )

            weights.append(
                FACE_WEIGHT
            )

        if (
            query_semantic is not None
            and item["semantic"]
            is not None
        ):

            scores.append(
                item["semantic"]
            )

            weights.append(
                SEMANTIC_WEIGHT
            )

        if not weights:
            continue

        fusion = sum(
            score * weight
            for score, weight
            in zip(
                scores,
                weights,
            )
        ) / sum(weights)

        fused.append({

            **item,

            "fusion":
                fusion,
        })

    fused.sort(

        key=lambda x:
            x["fusion"],

        reverse=True,
    )

    print()
    print("=" * 80)
    print("PERSON IMAGE -> VIDEO")
    print("=" * 80)

    for rank, item in enumerate(
        fused[:top_k],
        start=1,
    ):

        payload = (
            item[
                "point"
            ].payload
            or {}
        )

        print()
        print(
            f"[Rank {rank}]"
        )

        print(
            f"Fusion   : "
            f"{item['fusion']:.4f}"
        )

        print(
            f"ReID     : "
            f"{item['reid']}"
        )

        print(
            f"Face     : "
            f"{item['face']}"
        )

        print(
            f"Semantic : "
            f"{item['semantic']}"
        )

        print(
            f"Video    : "
            f"{payload.get('video_name')}"
        )

        print(
            f"Time     : "
            f"{payload.get('timestamp_text')}"
        )

        print(
            f"Frame    : "
            f"{payload.get('frame_number')}"
        )

        print(
            f"Person   : "
            f"{payload.get('person_index')}"
        )

        print(
            f"Crop     : "
            f"{payload.get('crop_path')}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--build",
        action="store_true",
    )

    parser.add_argument(
        "--image",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if args.build:

        build_database()

        return

    if args.image:

        image_to_video(

            args.image,

            args.top_k,
        )

        return

    parser.print_help()


if __name__ == "__main__":
    main()