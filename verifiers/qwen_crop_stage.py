from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from verifiers.qwen_stage import DEFAULT_MODEL, QwenVL, run as run_text_qwen

logger = logging.getLogger("qwen_crop_stage")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

QUERY_CROP_TO_TEXT_PROMPT = """Describe the single main subject in this crop image in one concise English sentence for retrieval verification. Focus only on clearly visible attributes. For a person, mention gender presentation if visible, clothing type and colors, carried items, accessories, hair or hat, and other distinctive appearance cues. For an object, mention object category, color, shape, material, and distinctive visual traits. Do not guess invisible details. Output only the sentence."""


def build_text_payload(
    payload: Dict[str, Any],
    *,
    model_id: str,
    dtype: str,
    device: str | None,
    max_pixels: int | None,
    qwen: QwenVL | None = None,
) -> Dict[str, Any]:
    """query crop 을 Qwen 으로 한 문장 캡션으로 바꿔 text 검증 payload 를 만든다.

    qwen 을 주면 그 인스턴스를 로드해 쓰고 release 하지 않는다 (호출자가 검증 단계까지 재사용).
    같은 프로세스에서 두 번 로드하면 Windows 에서 access violation 이 나는 경우가 있어 한 번만 로드한다.
    """
    crops = payload.get("crops") or []
    if not crops or not isinstance(crops[0], dict):
        raise ValueError("crop 검색 결과가 없습니다.")

    item = dict(crops[0])
    query_image = item.get("query_image") or payload.get("query_image")
    if not query_image:
        raise ValueError("query_image 경로가 없습니다.")

    query_path = Path(str(query_image))
    if not query_path.is_file():
        raise FileNotFoundError(f"query crop 파일을 찾을 수 없습니다: {query_path}")

    owns = qwen is None
    if qwen is None:
        qwen = QwenVL(model_id=model_id, dtype=dtype, device=device, max_pixels=max_pixels)
    try:
        qwen.load()
        query_text = qwen.generate(QUERY_CROP_TO_TEXT_PROMPT, image_path=str(query_path)).strip()
    finally:
        if owns:
            qwen.release()

    if not query_text:
        raise RuntimeError("Qwen이 query crop 설명을 생성하지 못했습니다.")

    return {
        "search_type": "text",
        "query": query_text,
        "query_en": query_text,
        "qwen": False,
        "media": payload.get("media"),
        "generated_from_crop": True,
        "query_image": str(query_path),
        "query_caption": query_text,
        "crops": [
            {
                "crop_index": item.get("crop_index", 1),
                "kind": "text",
                "query_text_original": query_text,
                "query_text": query_text,
                "scope": item.get("scope"),
                "collection": item.get("collection"),
                "media": item.get("media") or payload.get("media"),
                "results": item.get("results") or [],
            }
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Crop 기반 검색 결과를 Qwen으로 후처리한다.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--verify-mode", default="flag")
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-pixels", type=int, default=768 * 768)
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args()

    src = Path(args.inp)
    payload = json.loads(src.read_text(encoding="utf-8"))

    # 캡션 생성과 검증에 같은 인스턴스를 쓴다 (프로세스당 모델 로드 1회).
    qwen = QwenVL(model_id=args.model_id, dtype=args.dtype, device=args.device, max_pixels=args.max_pixels)
    try:
        text_payload = build_text_payload(
            payload,
            model_id=args.model_id,
            dtype=args.dtype,
            device=args.device,
            max_pixels=args.max_pixels,
            qwen=qwen,
        )
        out = run_text_qwen(
            text_payload,
            top_k=args.top_k,
            alpha=args.alpha,
            threshold=args.threshold,
            verify_mode=args.verify_mode,
            rescore_only=False,
            weights={},
            required=set(),
            soft=set(),
            model_id=args.model_id,
            dtype=args.dtype,
            device=args.device,
            max_pixels=args.max_pixels,
            qwen_instance=qwen,
        )
    finally:
        qwen.release()
    out["search_type"] = "crop"
    out["query_image"] = text_payload.get("query_image")
    out["query_caption"] = text_payload.get("query_caption")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("완료: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
