"""
search_and_save.py — 자연어 검색 결과를 이미지 폴더로 저장

사용법:
  python src/RF-DETR/search_and_save.py --query "red jacket man"
  python src/RF-DETR/search_and_save.py --query "white shirt dark pants" --top-k 20
  python src/RF-DETR/search_and_save.py --query "red jacket man" --out search_results/my_run

결과 폴더 구조:
  search_results/
    red_jacket_man_k10/
      01_0.5449_Normal_Videos_935_track_0869.jpg
      02_0.5293_Normal_Videos_935_track_0941.jpg
      ...
      _query.txt   ← 사용한 쿼리 + 점수 목록 기록
"""

import sys
import shutil
import argparse
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from forensic_video_search import ForensicSearcher

OUTPUT_BASE = "search_results"


def slugify(text: str) -> str:
    """쿼리 문자열 → 폴더명용 안전한 문자열."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s가-힣-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:60]


def save_results(query: str, top_k: int = 10, out_dir: str = None):
    searcher = ForensicSearcher()

    # 단일 + multi 병합 옵션 대신 단순하게 단일 쿼리 사용
    results = searcher.search(query, top_k=top_k, by_track=True)

    if not results:
        print("[!] 결과 없음")
        return

    # 출력 폴더 결정
    if out_dir:
        out = Path(out_dir)
    else:
        slug = slugify(query)
        out = Path(OUTPUT_BASE) / f"{slug}_k{top_k}"

    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[+] 저장 폴더: {out.resolve()}")

    # 이미지 복사
    copied = 0
    log_lines = [f"쿼리: {query}", f"top_k: {top_k}", ""]
    for r in results:
        src = Path(r.get("best_path") or r.get("path", ""))
        if not src.exists():
            print(f"  [!] 파일 없음: {src}")
            log_lines.append(f"{r['rank']:02d}위  [{r['similarity']:.4f}]  파일없음  {src}")
            continue

        rank  = r["rank"]
        score = r["similarity"]
        video = r["video"].replace(" ", "_")
        track = r["track"]

        dst_name = f"{rank:02d}_{score:.4f}_{video}_{track}{src.suffix}"
        dst = out / dst_name
        shutil.copy2(src, dst)
        copied += 1

        log_lines.append(f"{rank:02d}위  [{score:.4f}]  {video} / {track}")
        log_lines.append(f"       → {dst_name}")
        print(f"  {rank:02d}위  [{score:.4f}]  {dst_name}")

    # 쿼리 + 점수 로그 저장
    (out / "_query.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[+] 총 {copied}개 이미지 저장 완료 → {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="검색 결과를 이미지 폴더로 저장")
    ap.add_argument("--query", "-q", required=True, help='검색 쿼리 (예: "red jacket man")')
    ap.add_argument("--top-k", "-k", type=int, default=10, help="상위 결과 수 (기본: 10)")
    ap.add_argument("--out", default=None, help="저장 폴더 경로 (기본: search_results/<쿼리>_k<k>)")
    ap.add_argument("--index", default="data/irra_index/irra.faiss", help="FAISS 인덱스 경로")
    ap.add_argument("--meta",  default="data/irra_index/metadata.db",  help="메타데이터 DB 경로")
    ap.add_argument("--weight", default="weights/IRRA/irra_cuhk_pedes_download", help="모델 가중치")
    args = ap.parse_args()
    save_results(args.query, args.top_k, args.out)
