"""
main.py — ForensicSearch 진입점

이미지(전경 포함) 또는 텍스트로 유사 인물 검색.

사용법:
    # 텍스트 검색
    python src/RF-DETR/main.py --text "red jacket man" -k 10

    # 이미지 직접 검색 (crop 이미지)
    python src/RF-DETR/main.py --query 사진경로.jpg -k 10

    # RF-DETR 검출 후 검색 (배경 포함 전경 이미지)
    python src/RF-DETR/main.py --query 사진경로.jpg --detect -k 10

    # Qwen 리랭킹 포함
    python src/RF-DETR/main.py --text "red jacket man" --rerank -k 20
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

_RFDETR = Path(__file__).resolve().parent
_ROOT   = _RFDETR.parents[1]
sys.path.insert(0, str(_ROOT / "src" / "IRRA"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_RFDETR))

from forensic_video_search import ForensicSearcher, encode_image_irra, avg_normalize

RESULTS_OUTPUT_PATH = str(_ROOT / "data" / "results.json")


def build_result_json(query_type, query_value, results, search_time_ms):
    return {
        "query_type":     query_type,
        "query":          query_value,
        "result_count":   len(results),
        "search_time_ms": search_time_ms,
        "results": [
            {
                "rank":       r.get("rank", 0),
                "video":      r.get("video", ""),
                "track":      r.get("track", ""),
                "source":     r.get("source", ""),
                "similarity": r.get("similarity", 0.0),
                "best_path":  r.get("best_path") or r.get("path", ""),
                "n_frames":   r.get("n_frames", 1),
                **({"qwen_similarity": r["qwen_similarity"]} if "qwen_similarity" in r else {}),
            }
            for r in results
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="ForensicSearch 인물 검색")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--query",  "-q", help="이미지 파일 경로")
    grp.add_argument("--text",   "-t", help="텍스트 쿼리")
    ap.add_argument("--top-k",   "-k", type=int, default=10)
    ap.add_argument("--detect",        action="store_true",
                    help="RF-DETR 사람 검출 후 검색 (--query 와 함께 사용)")
    ap.add_argument("--rerank",        action="store_true",
                    help="Qwen3-VL 리랭킹 활성화")
    ap.add_argument("--output",        default=RESULTS_OUTPUT_PATH)
    ap.add_argument("--json-out",      action="store_true")
    args = ap.parse_args()

    searcher = ForensicSearcher()
    t0 = time.time()

    # ── 검색 ─────────────────────────────────────────────────────────
    if args.text:
        results     = searcher.search(args.text, top_k=args.top_k)
        query_type  = "text"
        query_value = args.text

    elif args.detect and args.query:
        # RF-DETR 검출 → 크롭 임베딩 평균
        from detect_rf import load_detect_model, detect_and_crop
        print("[*] RF-DETR 사람 검출 중 ...")
        det_model = load_detect_model()
        crops     = detect_and_crop(det_model, args.query)
        if not crops:
            print("[!] 사람 검출 실패")
            return
        print(f"[+] {len(crops)}명 검출")
        embs    = [encode_image_irra(searcher.model, c) for c in crops]
        q_emb   = avg_normalize(embs)
        results = searcher._search_by_emb(q_emb, top_k=args.top_k)
        query_type  = "image_detect"
        query_value = args.query

    else:
        results     = searcher.search_by_image(args.query, top_k=args.top_k)
        query_type  = "image"
        query_value = args.query

    # ── Qwen 리랭킹 ──────────────────────────────────────────────────
    if args.rerank and results:
        from qwen_vlm import load_embedding_model
        from qwen_reranker import rerank_with_qwen_text, rerank_with_qwen_image
        print("[*] Qwen3-VL 모델 로드 중 ...")
        qwen_model = load_embedding_model()
        if args.text:
            results = rerank_with_qwen_text(qwen_model, args.text, results, top_k=args.top_k)
        else:
            results = rerank_with_qwen_image(qwen_model, query_value, results, top_k=args.top_k)

    elapsed_ms = int((time.time() - t0) * 1000)

    # ── 출력 ─────────────────────────────────────────────────────────
    if args.json_out:
        print(json.dumps(build_result_json(query_type, query_value, results, elapsed_ms),
                         ensure_ascii=False, indent=2))
    else:
        searcher.print_results(results, query_value)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(build_result_json(query_type, query_value, results, elapsed_ms),
                  f, ensure_ascii=False, indent=2)
    print(f"[+] 결과 저장: {args.output} ({elapsed_ms}ms)")
    searcher.close()


if __name__ == "__main__":
    main()
