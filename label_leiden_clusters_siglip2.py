from __future__ import annotations

import argparse
import html
import json
import random
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests
from PIL import Image

import torch
from transformers import AutoModel, AutoProcessor


# ============================================================
# 기본 설정
# ============================================================

DEFAULT_ASSIGNMENTS = (
    r"outputs\clustering\leiden\person\person_leiden_assignments.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    r"outputs\clustering\leiden\person\labels_with_images"
)

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "forensic_person"

DEFAULT_MODEL = "google/siglip2-base-patch16-224"


# ============================================================
# SigLIP2 prompt
# 민감속성/신원 추론용이 아니라 외형 설명용
# ============================================================

PROMPT_BANKS = {

    "upper_color": [
        ("black", "a person wearing black upper clothing"),
        ("white", "a person wearing white upper clothing"),
        ("gray", "a person wearing gray upper clothing"),
        ("red", "a person wearing red upper clothing"),
        ("blue", "a person wearing blue upper clothing"),
        ("green", "a person wearing green upper clothing"),
        ("yellow", "a person wearing yellow upper clothing"),
        ("orange", "a person wearing orange upper clothing"),
        ("pink", "a person wearing pink upper clothing"),
        ("purple", "a person wearing purple upper clothing"),
        ("brown", "a person wearing brown upper clothing"),
        ("beige", "a person wearing beige upper clothing"),
    ],

    "upper_type": [
        ("t-shirt", "a person wearing a t-shirt"),
        ("shirt", "a person wearing a shirt"),
        ("jacket", "a person wearing a jacket"),
        ("hoodie", "a person wearing a hoodie"),
        ("sweater", "a person wearing a sweater"),
        ("coat", "a person wearing a coat"),
        ("sleeveless", "a person wearing a sleeveless top"),
    ],

    "lower_type": [
        ("pants", "a person wearing pants"),
        ("shorts", "a person wearing shorts"),
        ("skirt", "a person wearing a skirt"),
        ("dress", "a person wearing a dress"),
    ],

    "accessory": [
        ("backpack", "a person carrying a backpack"),
        ("shoulder bag", "a person carrying a shoulder bag"),
        ("handbag", "a person carrying a handbag"),
        ("hat/cap", "a person wearing a hat or cap"),
        ("glasses", "a person wearing glasses"),
        (
            "no obvious accessory",
            "a person with no obvious bag hat or glasses",
        ),
    ],

    "crop_quality": [
        ("full body", "a full body person crop"),
        ("upper body", "an upper body person crop"),
        (
            "partial body",
            "a partial body crop with only part of the person visible",
        ),
        (
            "poor crop",
            "a poor person crop dominated by background or body fragments",
        ),
    ],
}


# ============================================================
# Util
# ============================================================

def chunks(seq: Sequence[Any], size: int):

    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


# ============================================================
# Qdrant
# ============================================================

class QdrantHTTP:

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 180,
    ):

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()

        self.session.headers.update({
            "Content-Type": "application/json"
        })

        if api_key:
            self.session.headers["api-key"] = api_key

    def post(
        self,
        path: str,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:

        r = self.session.post(
            self.base_url + path,
            json=body,
            timeout=self.timeout,
        )

        if not r.ok:

            raise RuntimeError(
                f"Qdrant error\n"
                f"status={r.status_code}\n"
                f"path={path}\n"
                f"{r.text[:2000]}"
            )

        return r.json()

    def retrieve(
        self,
        collection: str,
        ids: Sequence[Any],
        batch_size: int = 256,
    ) -> Dict[str, Dict[str, Any]]:

        result = {}

        for batch in chunks(list(ids), batch_size):

            data = self.post(
                f"/collections/{collection}/points",
                {
                    "ids": list(batch),
                    "with_payload": True,
                    "with_vector": False,
                },
            )

            for row in data.get("result") or []:

                result[str(row["id"])] = (
                    row.get("payload") or {}
                )

        return result


# ============================================================
# Assignment
# ============================================================

def load_assignments(path: Path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if "point_id" not in row:

                raise RuntimeError(
                    f"point_id 없음: line={line_no}"
                )

            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"assignment 파일 비어 있음: {path}"
        )

    return rows


def split_groups(rows):

    groups = defaultdict(list)
    noise = []

    for row in rows:

        cid = row.get("cluster_id")

        is_noise = (
            bool(row.get("is_noise"))
            or cid is None
        )

        if is_noise:

            noise.append(row)

        else:

            groups[str(cid)].append(row)

    return dict(groups), noise


# ============================================================
# 대표 이미지 선택
# ============================================================

def choose_representatives(
    members,
    count: int,
    seed: int,
):

    if len(members) <= count:
        return list(members)

    rng = random.Random(seed)

    result = []

    # 앞/중간/뒤가 어느 정도 섞이도록
    result.append(members[0])

    if len(members) > 2:
        result.append(members[len(members) // 2])

    result.append(members[-1])

    used = {
        str(x["point_id"])
        for x in result
    }

    remain = [
        x for x in members
        if str(x["point_id"]) not in used
    ]

    need = count - len(result)

    if need > 0:

        if len(remain) > need:
            result.extend(
                rng.sample(remain, need)
            )
        else:
            result.extend(remain)

    return result[:count]


# ============================================================
# Payload path
# ============================================================

def get_crop_path(
    payload: Dict[str, Any]
) -> Optional[str]:

    keys = (
        "crop_path",
        "selected_crop_path",
        "image_path",
        "path",
    )

    for key in keys:

        value = payload.get(key)

        if value is not None:

            value = str(value).strip()

            if value:
                return value

    return None


def resolve_crop_path(
    raw_path: Optional[str],
    project_root: Path,
) -> Optional[Path]:

    if not raw_path:
        return None

    p = Path(raw_path)

    if p.is_file():
        return p

    if not p.is_absolute():

        p2 = (
            project_root / p
        ).resolve()

        if p2.is_file():
            return p2

    normalized = raw_path.replace("\\", "/")

    p3 = (
        project_root / normalized
    ).resolve()

    if p3.is_file():
        return p3

    return None


def load_image(path: Path):

    try:

        with Image.open(path) as im:
            return im.convert("RGB")

    except Exception:
        return None


# ============================================================
# SigLIP2
# ============================================================

class SigLIP2Labeler:

    def __init__(
        self,
        model_name: str,
        device: str,
    ):

        self.device = torch.device(device)

        print()
        print(
            f"[MODEL] loading {model_name}"
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                model_name
            )
        )

        dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        self.model = (
            AutoModel.from_pretrained(
                model_name,
                dtype=dtype,
            )
            .to(self.device)
            .eval()
        )

        self.prompt_features = {}

        self.prepare_prompts()

    def extract_tensor(
        self,
        output,
        kind: str,
    ):

        # 바로 tensor인 경우
        if torch.is_tensor(output):
            return output

        # transformers 버전에 따라
        # 다양한 ModelOutput 형태 가능
        candidates = [
            f"{kind}_embeds",
            "pooler_output",
            "pooled_output",
        ]

        for attr in candidates:

            value = getattr(
                output,
                attr,
                None,
            )

            if torch.is_tensor(value):
                return value

        # fallback
        last_hidden = getattr(
            output,
            "last_hidden_state",
            None,
        )

        if torch.is_tensor(last_hidden):

            if last_hidden.ndim == 3:
                return last_hidden[:, 0, :]

            return last_hidden

        keys = (
            list(output.keys())
            if hasattr(output, "keys")
            else "unknown"
        )

        raise RuntimeError(
            f"embedding tensor 추출 실패\n"
            f"type={type(output)}\n"
            f"keys={keys}"
        )

    @torch.inference_mode()
    def text_features(
        self,
        texts: List[str],
    ):

        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        if hasattr(
            self.model,
            "get_text_features",
        ):

            output = (
                self.model.get_text_features(
                    **inputs
                )
            )

        else:

            output = self.model(
                **inputs
            )

        features = self.extract_tensor(
            output,
            "text",
        )

        features = features.float()

        features = (
            features
            / features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        return features

    @torch.inference_mode()
    def image_features(
        self,
        images: List[Image.Image],
    ):

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        if hasattr(
            self.model,
            "get_image_features",
        ):

            output = (
                self.model.get_image_features(
                    **inputs
                )
            )

        else:

            output = self.model(
                **inputs
            )

        features = self.extract_tensor(
            output,
            "image",
        )

        features = features.float()

        features = (
            features
            / features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        return features

    def prepare_prompts(self):

        print(
            "[MODEL] preparing text prompts"
        )

        for bank_name, entries \
                in PROMPT_BANKS.items():

            labels = [
                x[0]
                for x in entries
            ]

            prompts = [
                x[1]
                for x in entries
            ]

            features = self.text_features(
                prompts
            )

            self.prompt_features[
                bank_name
            ] = (
                labels,
                features,
            )

    @torch.inference_mode()
    def classify(
        self,
        images: List[Image.Image],
    ):

        image_features = (
            self.image_features(images)
        )

        results = {}

        for (
            bank_name,
            (
                labels,
                text_features,
            )
        ) in self.prompt_features.items():

            similarity = (
                image_features
                @ text_features.T
            )

            # 상대 확률로 사용
            probabilities = torch.softmax(
                similarity * 10.0,
                dim=-1,
            )

            mean_prob = (
                probabilities.mean(
                    dim=0
                )
            )

            best_index = int(
                mean_prob.argmax().item()
            )

            winner_per_image = (
                probabilities.argmax(
                    dim=-1
                )
            )

            consensus = float(
                (
                    winner_per_image
                    == best_index
                )
                .float()
                .mean()
                .item()
            )

            values = torch.topk(
                mean_prob,
                k=min(
                    2,
                    len(labels),
                ),
            ).values

            if len(values) >= 2:

                margin = float(
                    (
                        values[0]
                        - values[1]
                    ).item()
                )

            else:

                margin = float(
                    values[0].item()
                )

            results[bank_name] = {

                "label":
                    labels[best_index],

                "confidence":
                    float(
                        mean_prob[
                            best_index
                        ].item()
                    ),

                "consensus":
                    consensus,

                "margin":
                    margin,
            }

        return results


# ============================================================
# 라벨 생성
# ============================================================

def build_cluster_label(
    scores: Dict[str, Dict[str, Any]]
):

    upper_color = (
        scores["upper_color"]
    )

    upper_type = (
        scores["upper_type"]
    )

    lower_type = (
        scores["lower_type"]
    )

    accessory = (
        scores["accessory"]
    )

    crop_quality = (
        scores["crop_quality"]
    )

    core_consensus = statistics.mean([
        upper_color["consensus"],
        upper_type["consensus"],
        crop_quality["consensus"],
    ])

    core_confidence = statistics.mean([
        upper_color["confidence"],
        upper_type["confidence"],
        crop_quality["confidence"],
    ])

    # crop 자체가 별로인 군집
    if (
        crop_quality["label"]
        == "poor crop"
        and
        crop_quality["consensus"] >= 0.50
    ):

        cluster_name = (
            "poor-crop / uncertain"
        )

    # 대표 crop들끼리 너무 제각각
    elif core_consensus < 0.45:

        cluster_name = (
            "mixed / uncertain"
        )

    else:

        parts = [

            f"{upper_color['label']} "
            f"{upper_type['label']}"
        ]

        if (
            lower_type["consensus"]
            >= 0.50
        ):

            parts.append(
                lower_type["label"]
            )

        if (
            accessory["label"]
            != "no obvious accessory"
            and
            accessory["consensus"]
            >= 0.50
        ):

            parts.append(
                accessory["label"]
            )

        cluster_name = (
            " · ".join(parts)
        )

    label_confidence = (

        0.5 * core_consensus
        +
        0.5 * core_confidence
    )

    description = (

        f"upper_color="
        f"{upper_color['label']} "
        f"(consensus="
        f"{upper_color['consensus']:.2f}), "

        f"upper_type="
        f"{upper_type['label']} "
        f"(consensus="
        f"{upper_type['consensus']:.2f}), "

        f"lower="
        f"{lower_type['label']} "
        f"(consensus="
        f"{lower_type['consensus']:.2f}), "

        f"accessory="
        f"{accessory['label']} "
        f"(consensus="
        f"{accessory['consensus']:.2f}), "

        f"crop="
        f"{crop_quality['label']} "
        f"(consensus="
        f"{crop_quality['consensus']:.2f})"
    )

    return (
        cluster_name,
        description,
        float(label_confidence),
    )


# ============================================================
# 이미지 복사
# ============================================================

def copy_representative_image(
    src: Path,
    output_dir: Path,
    cluster_index: int,
    image_index: int,
):

    dst_dir = (
        output_dir
        / "assets"
        / f"cluster_{cluster_index:04d}"
    )

    dst_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = src.suffix.lower()

    if suffix not in (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    ):
        suffix = ".jpg"

    dst = (
        dst_dir
        / f"rep_{image_index:02d}{suffix}"
    )

    shutil.copy2(
        src,
        dst,
    )

    return dst


# ============================================================
# HTML
# ============================================================

def write_html(
    records,
    output_dir: Path,
):

    cards = []

    for rank, record in enumerate(
        records,
        1,
    ):

        images_html = []

        for img in record[
            "representative_images"
        ]:

            relative = Path(
                img
            ).relative_to(
                output_dir
            ).as_posix()

            images_html.append(
                f"""
                <div class="thumb">
                    <a
                      href="{esc(relative)}"
                      target="_blank"
                    >
                      <img
                        src="{esc(relative)}"
                        loading="lazy"
                      >
                    </a>
                </div>
                """
            )

        score_rows = []

        for (
            category,
            result,
        ) in record[
            "scores"
        ].items():

            score_rows.append(
                f"""
                <tr>
                    <td>{esc(category)}</td>
                    <td>
                        {esc(result["label"])}
                    </td>
                    <td>
                        {result["confidence"]:.3f}
                    </td>
                    <td>
                        {result["consensus"]:.3f}
                    </td>
                </tr>
                """
            )

        cards.append(
            f"""
            <section class="cluster">

                <div class="header">

                    <div>

                        <h2>
                            Cluster {rank}
                        </h2>

                        <div class="id">
                            cluster_id:
                            {esc(record["cluster_id"])}
                        </div>

                    </div>

                    <div class="size">
                        {record["cluster_size"]:,}
                        points
                    </div>

                </div>


                <div class="label">

                    {esc(
                        record[
                            "cluster_name"
                        ]
                    )}

                </div>


                <div class="confidence">

                    Label confidence:
                    <b>{record["label_confidence"]:.3f}</b>

                </div>


                <div class="description">

                    {
                        esc(
                            record[
                                "cluster_description"
                            ]
                        )
                    }

                </div>


                <div class="gallery">

                    {''.join(images_html)}

                </div>


                <details>

                    <summary>
                        SigLIP2 상세 점수
                    </summary>

                    <table>

                        <thead>

                            <tr>
                                <th>Category</th>
                                <th>Label</th>
                                <th>Confidence</th>
                                <th>Consensus</th>
                            </tr>

                        </thead>

                        <tbody>

                            {''.join(score_rows)}

                        </tbody>

                    </table>

                </details>

            </section>
            """
        )

    html_text = f"""
<!doctype html>

<html lang="ko">

<head>

<meta charset="utf-8">

<meta
 name="viewport"
 content="width=device-width, initial-scale=1"
>

<title>
Leiden Cluster Labels + Images
</title>

<style>

:root {{

    --bg: #0b1020;
    --panel: #141b2d;
    --panel2: #19233b;
    --line: #2c395b;
    --text: #eef3ff;
    --muted: #9dadca;
    --accent: #7aa2ff;
    --green: #69d4aa;
}}

* {{
    box-sizing: border-box;
}}

body {{

    margin: 0;
    background: var(--bg);
    color: var(--text);

    font-family:
        Segoe UI,
        Arial,
        sans-serif;
}}

.container {{

    max-width: 1600px;
    margin: auto;
    padding: 26px;
}}

h1 {{
    margin-top: 0;
}}

.note {{

    background: var(--panel);
    border: 1px solid var(--line);
    border-left:
        4px solid
        var(--accent);

    padding: 14px;
    border-radius: 10px;

    color: var(--muted);

    margin-bottom: 22px;
}}

.cluster {{

    background: var(--panel);

    border:
        1px solid
        var(--line);

    border-radius: 14px;

    padding: 18px;

    margin-bottom: 22px;
}}

.header {{

    display: flex;
    justify-content: space-between;
    gap: 20px;
}}

.header h2 {{
    margin: 0;
}}

.id {{

    color: var(--muted);

    font-size: 12px;

    margin-top: 4px;

    word-break: break-all;
}}

.size {{

    color: var(--green);

    font-weight: 700;
}}

.label {{

    font-size: 23px;

    font-weight: 700;

    margin-top: 15px;
}}

.confidence {{

    margin-top: 6px;

    color: #d8e2fa;
}}

.description {{

    color: var(--muted);

    margin-top: 8px;

    font-size: 13px;
}}

.gallery {{

    display: grid;

    grid-template-columns:
        repeat(
            auto-fill,
            minmax(
                155px,
                1fr
            )
        );

    gap: 12px;

    margin-top: 18px;
}}

.thumb {{

    background:
        var(--panel2);

    border:
        1px solid
        var(--line);

    border-radius: 10px;

    padding: 7px;
}}

.thumb img {{

    width: 100%;

    height: 220px;

    object-fit: contain;

    background: #070b14;

    border-radius: 7px;

    display: block;
}}

details {{
    margin-top: 15px;
}}

summary {{
    cursor: pointer;
    color: #bcd0ff;
}}

table {{

    width: 100%;

    border-collapse: collapse;

    margin-top: 10px;

    font-size: 13px;
}}

th,
td {{

    border:
        1px solid
        var(--line);

    padding: 7px;

    text-align: left;
}}

th {{
    background:
        var(--panel2);
}}

</style>

</head>


<body>

<div class="container">

<h1>
Leiden Person Cluster
Auto Label + Image Gallery
</h1>


<div class="note">

이 이름은 사람의 신원을 뜻하지 않습니다.
SigLIP2가 대표 crop의 옷 색상,
의류 형태, 가방 등의
시각적 외형을 요약한
자동 메타데이터입니다.

</div>


{''.join(cards)}


</div>

</body>

</html>
"""

    output_html = (
        output_dir
        / "cluster_labels_with_images.html"
    )

    output_html.write_text(
        html_text,
        encoding="utf-8",
    )

    return output_html


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--assignments",
        default=DEFAULT_ASSIGNMENTS,
    )

    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
    )

    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
    )

    parser.add_argument(
        "--api-key",
        default=None,
    )

    parser.add_argument(
        "--project-root",
        default=".",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--representatives",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-clusters",
        type=int,
        default=100,
        help="0이면 전체 cluster",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--qdrant-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()


    assignments_path = Path(
        args.assignments
    ).resolve()

    project_root = Path(
        args.project_root
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    rows = load_assignments(
        assignments_path
    )

    groups, noise = split_groups(
        rows
    )


    ordered = sorted(

        groups.items(),

        key=lambda kv: (
            -len(kv[1]),
            kv[0],
        )
    )


    if args.max_clusters > 0:

        ordered = ordered[
            :args.max_clusters
        ]


    print("=" * 88)

    print(
        "LEIDEN AUTO LABEL + IMAGE GALLERY"
    )

    print("=" * 88)

    print(
        f"assignments       : "
        f"{len(rows):,}"
    )

    print(
        f"clusters total    : "
        f"{len(groups):,}"
    )

    print(
        f"clusters labeling : "
        f"{len(ordered):,}"
    )

    print(
        f"noise             : "
        f"{len(noise):,}"
    )

    print(
        f"reps / cluster    : "
        f"{args.representatives}"
    )

    print(
        f"collection        : "
        f"{args.collection}"
    )

    print(
        f"model             : "
        f"{args.model}"
    )

    print(
        f"device            : "
        f"{args.device}"
    )


    # --------------------------------------------------------
    # representative IDs
    # --------------------------------------------------------

    reps_by_cluster = {}

    all_rep_ids = []


    for index, (
        cid,
        members,
    ) in enumerate(
        ordered
    ):

        reps = choose_representatives(

            members,

            args.representatives,

            args.seed + index,
        )

        reps_by_cluster[
            cid
        ] = reps

        all_rep_ids.extend(
            [
                r["point_id"]
                for r in reps
            ]
        )


    qdrant = QdrantHTTP(

        args.qdrant_url,

        args.api_key,
    )


    payloads = qdrant.retrieve(

        args.collection,

        all_rep_ids,

        batch_size=
            args.qdrant_batch_size,
    )


    print(
        f"representative payloads fetched: "
        f"{len(payloads):,}"
    )


    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    labeler = SigLIP2Labeler(

        args.model,

        args.device,
    )


    records = []

    missing_images = 0


    # --------------------------------------------------------
    # each cluster
    # --------------------------------------------------------

    for cluster_index, (
        cid,
        members,
    ) in enumerate(
        ordered,
        1,
    ):

        representatives = (
            reps_by_cluster[
                cid
            ]
        )

        images = []

        valid_paths = []

        used_point_ids = []


        for rep in representatives:

            pid = rep["point_id"]

            payload = payloads.get(
                str(pid),
                {},
            )

            raw_path = get_crop_path(
                payload
            )

            path = resolve_crop_path(

                raw_path,

                project_root,
            )

            if path is None:

                missing_images += 1

                continue


            image = load_image(
                path
            )

            if image is None:

                missing_images += 1

                continue


            images.append(
                image
            )

            valid_paths.append(
                path
            )

            used_point_ids.append(
                pid
            )


        # 이미지 하나도 없음
        if not images:

            record = {

                "cluster_id":
                    cid,

                "cluster_size":
                    len(members),

                "cluster_name":
                    "unavailable",

                "cluster_description":
                    "representative image unavailable",

                "label_confidence":
                    0.0,

                "scores":
                    {},

                "representative_point_ids":
                    [],

                "representative_images":
                    [],
            }

            records.append(
                record
            )

            continue


        # ----------------------------------------------------
        # SigLIP2
        # ----------------------------------------------------

        scores = labeler.classify(
            images
        )


        (
            cluster_name,
            description,
            confidence,
        ) = build_cluster_label(
            scores
        )


        # ----------------------------------------------------
        # representative image copy
        # ----------------------------------------------------

        copied_paths = []

        for image_index, src_path \
                in enumerate(
                    valid_paths,
                    1,
                ):

            copied = (
                copy_representative_image(

                    src_path,

                    output_dir,

                    cluster_index,

                    image_index,
                )
            )

            copied_paths.append(
                str(copied)
            )


        record = {

            "cluster_id":
                cid,

            "cluster_size":
                len(members),

            "cluster_name":
                cluster_name,

            "cluster_description":
                description,

            "label_confidence":
                confidence,

            "scores":
                scores,

            "representative_point_ids":
                used_point_ids,

            "representative_images":
                copied_paths,
        }


        records.append(
            record
        )


        print(

            f"[{cluster_index:,}/"
            f"{len(ordered):,}] "

            f"size="
            f"{len(members):,} | "

            f"name="
            f"{cluster_name} | "

            f"conf="
            f"{confidence:.3f}"
        )


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_path = (
        output_dir
        / "cluster_labels.json"
    )


    json_path.write_text(

        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),

        encoding="utf-8",
    )


    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    jsonl_path = (
        output_dir
        / "cluster_labels.jsonl"
    )


    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(

                json.dumps(
                    record,
                    ensure_ascii=False,
                )

                + "\n"
            )


    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    html_path = write_html(

        records,

        output_dir,
    )


    print()

    print("=" * 88)

    print("DONE")

    print("=" * 88)

    print(
        f"JSON          : "
        f"{json_path}"
    )

    print(
        f"JSONL         : "
        f"{jsonl_path}"
    )

    print(
        f"HTML          : "
        f"{html_path}"
    )

    print(
        f"clusters      : "
        f"{len(records):,}"
    )

    print(
        f"missing images: "
        f"{missing_images:,}"
    )


if __name__ == "__main__":

    main()