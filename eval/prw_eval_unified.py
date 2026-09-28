"""
PRW GT 기반 **통합검색(2단계) 파이프라인** 평가
==============================================
eval/prw_eval.py 가 임베딩 하나의 단독 retrieval 을 재는 것이라면, 이 스크립트는
GUI/운영 통합검색(unified_search_4mode.py, scope=person, image query)의 순위 논리를
같은 PRW GT 프로토콜 위에서 오프라인으로 재현해 mAP / Rank-k 를 잰다.

운영 경로 (unified_search_4mode.py + search.py + qdrant_store.py):
  STEP 1  SigLIP2 와 IRRA 가 각각 prefetch_limit 개 후보를 내고
          가중 RRF(siglip2 1.0, irra 1.5, Qdrant 기본 k) 로 합쳐 상위 solider_pool(200) 개를 남긴다.
          (search.py _fetch: prefetch_limit(100) < 필요 후보 수(200) 이면 200 으로 올린다)
  STEP 2  후보 200개를 SOLIDER cosine 으로 재정렬해 Top-K(20) 를 보여 준다.

여기서는 gallery = PRW test 프레임 GT crop(19,127), query = query_box(2,057) 로
세 임베딩의 전체 유사도 행렬을 만든 뒤 위 순위 논리를 numpy 로 재현한다.
Qdrant HNSW/양자화 근사, 운영 DB 의 다른 source(COCO/영상) 방해물은 포함되지 않으므로
운영 수치의 **상한** 으로 해석해야 한다.

변형(진단용):
  single:<model>                각 임베딩 단독 (prw_eval.py 와 같아야 함)
  stage1_rrf                    STEP 1 만 (재정렬 없이 RRF 순위 그대로)
  unified                       STEP 1 → STEP 2 (운영 경로)
  unified_pool<P>               후보 수 P 를 바꿔 본 것
  irra_prefetch_solider         STEP 1 에서 SigLIP2 를 빼고 IRRA 만 → SOLIDER 재정렬
  rrf3                          세 임베딩 모두 RRF (search.py SearchEngine.search_image 기본 경로)
  stage1_rrf_k60                RRF k=60 (관례값) 로 바꾼 STEP 1
그리고 후보군 recall(정답이 후보 200개 안에 들어온 비율)을 함께 기록한다.

임베딩은 eval/results/cache/prw_gt_<model>.npz 에 캐시된다 (재실행 시 GT crop 임베딩 생략).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import prw_eval  # noqa: E402  (기존 스크립트 재사용, 수정 없음)

MODELS = ("siglip2", "irra", "solider")
DEFAULT_WEIGHTS = {"siglip2": 1.0, "irra": 1.5, "solider": 1.5}   # pipeline.yaml retrievers.weight
QDRANT_RRF_K_DEFAULT = 2   # Qdrant 서버 Rrf.k 기본값 (client 는 None 으로 넘긴다)


# ---------------------------------------------------------------------------
# 순위 논리 (순수 numpy; 테스트 가능)
# ---------------------------------------------------------------------------
def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """점수 내림차순 상위 k 개 인덱스 (k >= len 이면 전체 정렬)."""
    if k >= scores.shape[0]:
        return np.argsort(-scores, kind="stable")
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


def rrf_fuse(rank_lists: Dict[str, np.ndarray], weights: Dict[str, float], k: float) -> Tuple[np.ndarray, np.ndarray]:
    """rank_lists[name] = 상위 후보 인덱스 배열(순위순). 반환: (융합 순위의 인덱스, 융합 점수) 내림차순.

    RRF: score(d) = Σ_m w_m / (k + rank_m(d)),  rank 는 1부터. 동점은 인덱스 오름차순(결정적).
    """
    scores: Dict[int, float] = {}
    for name, idx in rank_lists.items():
        w = float(weights.get(name, 1.0))
        for r, gi in enumerate(idx.tolist(), start=1):
            scores[gi] = scores.get(gi, 0.0) + w / (k + r)
    if not scores:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return (np.asarray([i for i, _ in items], dtype=np.int64),
            np.asarray([s for _, s in items], dtype=np.float32))


def rerank_by(pool: np.ndarray, scores_row: np.ndarray) -> np.ndarray:
    """후보 pool 을 다른 점수(예: SOLIDER cosine) 로 재정렬."""
    if pool.size == 0:
        return pool
    order = np.argsort(-scores_row[pool], kind="stable")
    return pool[order]


def evaluate_ranked(ranked_lists: Sequence[np.ndarray], q_pids, q_frames, g_pids, g_frames, max_rank: int = 10) -> dict:
    """부분 순위(후보 밖 정답은 '못 찾음')에 대한 mAP / Rank-k. junk(같은 frame 같은 pid) 규칙은 prw_eval 과 동일."""
    g_pids_arr = np.asarray(g_pids)
    g_frames_arr = np.asarray(g_frames)
    cmc_sum = np.zeros(max_rank, dtype=np.float64)
    ap_sum = 0.0
    valid = 0
    pool_recall_sum = 0.0
    all_in_pool = 0
    any_in_pool = 0
    for qi in range(len(q_pids)):
        qpid, qframe = q_pids[qi], q_frames[qi]
        junk = (g_pids_arr == qpid) & (g_frames_arr == qframe)
        n_pos = int(((g_pids_arr == qpid) & (g_frames_arr != qframe)).sum())
        if n_pos == 0:
            continue
        valid += 1
        ranked = ranked_lists[qi]
        ranked = ranked[~junk[ranked]]
        is_pos = g_pids_arr[ranked] == qpid
        hit_positions = np.nonzero(is_pos)[0]
        if hit_positions.size and hit_positions[0] < max_rank:
            cmc_sum[hit_positions[0]:] += 1.0
        ap = 0.0
        for h, pos in enumerate(hit_positions.tolist(), start=1):
            ap += h / (pos + 1)
        ap_sum += ap / n_pos
        found = int(is_pos.sum())
        pool_recall_sum += found / n_pos
        all_in_pool += int(found == n_pos)
        any_in_pool += int(found > 0)
    if valid == 0:
        raise RuntimeError("유효한 query 가 없습니다.")
    cmc = cmc_sum / valid
    return {
        "mAP": round(100.0 * ap_sum / valid, 4),
        "Rank-1": round(100.0 * float(cmc[0]), 4),
        "Rank-5": round(100.0 * float(cmc[min(5, max_rank) - 1]), 4),
        "Rank-10": round(100.0 * float(cmc[min(10, max_rank) - 1]), 4),
        "valid_queries": valid,
        "pool_recall(%)": round(100.0 * pool_recall_sum / valid, 4),
        "queries_all_positives_in_pool(%)": round(100.0 * all_in_pool / valid, 4),
        "queries_any_positive_in_pool(%)": round(100.0 * any_in_pool / valid, 4),
    }


# ---------------------------------------------------------------------------
# 변형 실행
# ---------------------------------------------------------------------------
def run_variants(sims: Dict[str, np.ndarray], q_pids, q_frames, g_pids, g_frames, weights: Dict[str, float],
                 rrf_k: float, prefetch: int, pool: int, pools: Sequence[int], log=print) -> Dict[str, dict]:
    Q = next(iter(sims.values())).shape[0]
    results: Dict[str, dict] = {}

    def evaluate(name: str, make_ranked, note: str):
        t0 = time.time()
        ranked = [make_ranked(qi) for qi in range(Q)]
        res = evaluate_ranked(ranked, q_pids, q_frames, g_pids, g_frames)
        res["note"] = note
        res["sec"] = round(time.time() - t0, 1)
        results[name] = res
        log(f"  {name:<28} mAP={res['mAP']:6.2f}  R1={res['Rank-1']:6.2f}  R5={res['Rank-5']:6.2f}  "
            f"R10={res['Rank-10']:6.2f}  pool_recall={res['pool_recall(%)']:6.2f}")

    # 단독 (전체 순위)
    for m in MODELS:
        evaluate(f"single:{m}", lambda qi, m=m: np.argsort(-sims[m][qi], kind="stable"), f"{m} 단독, 전체 순위")

    def stage1(qi, names=("siglip2", "irra"), P=prefetch, k=rrf_k):
        lists = {m: topk_indices(sims[m][qi], P) for m in names}
        fused, _ = rrf_fuse(lists, weights, k)
        return fused

    # STEP 1 만
    evaluate("stage1_rrf", lambda qi: stage1(qi)[:pool], f"SigLIP2+IRRA 가중RRF(k={rrf_k}, prefetch={prefetch}) 상위 {pool}, 재정렬 없음")
    # 운영 통합검색
    evaluate("unified", lambda qi: rerank_by(stage1(qi)[:pool], sims["solider"][qi]),
             f"STEP1 상위 {pool} → SOLIDER 재정렬 (GUI 경로)")
    # 후보 수 변화
    for P in pools:
        if P == pool:
            continue
        evaluate(f"unified_pool{P}", lambda qi, P=P: rerank_by(stage1(qi, P=max(P, prefetch))[:P], sims["solider"][qi]),
                 f"후보 {P} (prefetch {max(P, prefetch)}) → SOLIDER 재정렬")
    # SigLIP2 제외
    evaluate("irra_prefetch_solider", lambda qi: rerank_by(topk_indices(sims["irra"][qi], pool), sims["solider"][qi]),
             f"IRRA 단독 후보 {pool} → SOLIDER 재정렬")
    evaluate("siglip2_prefetch_solider", lambda qi: rerank_by(topk_indices(sims["siglip2"][qi], pool), sims["solider"][qi]),
             f"SigLIP2 단독 후보 {pool} → SOLIDER 재정렬")
    # 세 임베딩 RRF (search.py 기본 경로)
    evaluate("rrf3", lambda qi: stage1(qi, names=MODELS)[:pool], f"SigLIP2+IRRA+SOLIDER 가중RRF(k={rrf_k}) 상위 {pool}")
    evaluate("rrf3_solider_rerank", lambda qi: rerank_by(stage1(qi, names=MODELS)[:pool], sims["solider"][qi]),
             "세 임베딩 RRF 후보 → SOLIDER 재정렬")
    # RRF k 민감도
    evaluate("stage1_rrf_k60", lambda qi: stage1(qi, k=60)[:pool], "STEP1 을 k=60 으로")
    evaluate("unified_k60", lambda qi: rerank_by(stage1(qi, k=60)[:pool], sims["solider"][qi]), "k=60 STEP1 → SOLIDER 재정렬")
    # 균등 가중치
    evaluate("stage1_rrf_equal", lambda qi: rrf_fuse({m: topk_indices(sims[m][qi], prefetch) for m in ("siglip2", "irra")},
                                                     {"siglip2": 1.0, "irra": 1.0}, rrf_k)[0][:pool],
             "STEP1 균등 가중치")
    return results


# ---------------------------------------------------------------------------
def retriever_fingerprint(config_path: str, model: str, log=print) -> str:
    """임베더 선언 지문 (module/class/params + 가중치 파일 내용 sha256) 의 sha1. yaml 의 다른 줄이 바뀌어도 그대로,
    같은 경로의 checkpoint 를 바꿔 끼우면 달라진다 (ingest.build_db.declared_retriever_fingerprint 재사용)."""
    from bench.ledger import retriever_fingerprint_sha
    return retriever_fingerprint_sha(config_path, model, log)


def load_or_embed(model: str, cache_dir: Path, g_crops, q_crops, g_meta, q_meta, args, log=print):
    cache = cache_dir / f"prw_gt_{model}.npz"
    fp = retriever_fingerprint(args.config, model, log)
    if cache.is_file() and not args.no_cache:
        d = np.load(cache, allow_pickle=False)
        # 호환성: 임베더 지문이 같아야 캐시를 쓴다. 지문 계산에 실패하면(fp == "") 재사용하지 않는다 —
        # 지문 없는 옛 캐시(retriever_fp 키 없음)만 config sha 동일 조건으로 한 번 더 허용하고, 새로 저장할 때 지문을 붙인다.
        if not fp:
            same_model = False
            log(f"  [{model}] 임베더 지문을 못 구해 캐시를 재사용하지 않음")
        elif "retriever_fp" in d.files:
            same_model = str(d["retriever_fp"]) == fp
        else:
            same_model = str(d["config_sha256"]) == args.config_sha256
        if d["g_vecs"].shape[0] == len(g_crops) and d["q_vecs"].shape[0] == len(q_crops) and same_model:
            log(f"  [{model}] 캐시 사용: {cache}" + (f" (지문 {fp[:8]})" if fp else ""))
            return d["g_vecs"], d["q_vecs"]
        log(f"  [{model}] 캐시 불일치(임베더 지문/크기) → 다시 임베딩")
    log(f"  [{model}] 임베더 로드...")
    emb = prw_eval.load_embedder(model, args)
    t0 = time.time()
    g_vecs = prw_eval.embed_pil(emb, g_crops, args.batch_size)
    q_vecs = prw_eval.embed_pil(emb, q_crops, args.batch_size)
    log(f"  [{model}] 임베딩 완료 gallery={g_vecs.shape} query={q_vecs.shape} ({time.time() - t0:.0f}s)")
    del emb
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, g_vecs=g_vecs.astype(np.float32), q_vecs=q_vecs.astype(np.float32),
             g_pids=g_meta[0], g_frames=np.array(g_meta[1]), q_pids=q_meta[0], q_frames=np.array(q_meta[1]),
             config_sha256=np.array(args.config_sha256), retriever_fp=np.array(fp))
    return g_vecs, q_vecs


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pipeline.yaml")
    ap.add_argument("--data-root", default="./data/PRW")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cache-dir", default="eval/results/cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--only-embed", default=None,
                    help="쉼표 구분 모델명. 해당 모델 임베딩만 캐시에 만들고 종료 (메모리 압박 시 모델별로 나눠 실행)")
    ap.add_argument("--prefetch", type=int, default=200, help="STEP1 retriever 별 후보 수 (운영: max(100, solider_pool)=200)")
    ap.add_argument("--pool", type=int, default=200, help="SOLIDER 재정렬 후보 수 (GUI SOLIDER_POOL_DEFAULT=200)")
    ap.add_argument("--pools", default="50,100,500,1000", help="후보 수 변형 목록")
    ap.add_argument("--rrf-k", type=float, default=QDRANT_RRF_K_DEFAULT, help="RRF k (Qdrant 기본 2)")
    ap.add_argument("--weights", default=None, help="예: siglip2=1.0,irra=1.5,solider=1.5 (기본 pipeline.yaml 값)")
    ap.add_argument("--out-json", default="eval/results/unified_eval.json")
    ap.add_argument("--out-csv", default="eval/results/unified_eval.csv")
    ap.add_argument("--ledger", default=None, help="실행 원장 JSONL (기본 bench/ledger.jsonl; '' 또는 none 이면 기록 안 함)")
    ap.add_argument("--no-ledger", action="store_true", help="원장(bench/ledger.jsonl)에 기록하지 않음")
    # prw_eval.load_embedder 가 읽는 인자들 (None 이면 pipeline.yaml 값)
    for name in ("irra-root", "irra-ckpt", "irra-cfg", "clip-pt", "solider-root", "solider-ckpt"):
        ap.add_argument(f"--{name}", default=None)
    ap.add_argument("--backbone", default="swin_base")
    ap.add_argument("--semantic-weight", type=float, default=0.2)
    ap.add_argument("--neck-feat", default="before")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.command_line = list(argv) if argv is not None else sys.argv[1:]
    import hashlib
    args.config_sha256 = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        for part in args.weights.split(","):
            k, v = part.split("=")
            weights[k.strip()] = float(v)
    else:
        try:
            from config import PipelineConfig
            cfg = PipelineConfig.load(args.config)
            weights = {m: float(cfg.retrievers[m].weight) for m in MODELS if m in cfg.retrievers}
        except Exception as exc:
            print(f"[경고] pipeline.yaml 가중치 읽기 실패, 기본값 사용: {exc}")
    pools = [int(x) for x in args.pools.split(",") if x.strip()]

    data_root = Path(args.data_root).expanduser().resolve()
    frames_dir, ann_dir = data_root / "frames", data_root / "annotations"
    qbox_dir, qinfo_path, ft_path = data_root / "query_box", data_root / "query_info.txt", data_root / "frame_test.mat"
    for p in (frames_dir, ann_dir, qbox_dir, qinfo_path, ft_path):
        if not p.exists():
            print(f"[ERROR] 경로가 없습니다: {p}")
            sys.exit(1)

    print("=" * 70)
    print(f" PRW 통합검색 평가 | weights={weights} rrf_k={args.rrf_k} prefetch={args.prefetch} pool={args.pool}")
    print("=" * 70)
    t0 = time.time()
    print("[1/3] gallery (GT crop) 로드...")
    test_frames = prw_eval.load_frame_list(ft_path)
    g_crops, g_pids, g_camids, g_frames = prw_eval.build_gallery(test_frames, frames_dir, ann_dir)
    print(f"  gallery={len(g_crops)} persons={len(set(g_pids.tolist()))} ({time.time() - t0:.0f}s)")
    print("[2/3] query 로드...")
    q_crops, q_pids, q_camids, q_frames = prw_eval.build_query(qbox_dir, qinfo_path)
    print(f"  queries={len(q_crops)} persons={len(set(q_pids.tolist()))}")

    try:
        import torch
        dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
        print(f"  device={dev} (cuda_available={torch.cuda.is_available()}, gpu0={name})")
    except Exception as exc:
        print(f"  device 확인 실패: {exc}")

    cache_dir = Path(args.cache_dir)
    if args.only_embed:
        for m in [x.strip() for x in args.only_embed.split(",") if x.strip()]:
            if m not in MODELS:
                raise SystemExit(f"알 수 없는 모델: {m} (선택: {MODELS})")
            print(f"[embed-only] {m}")
            load_or_embed(m, cache_dir, g_crops, q_crops, (g_pids, g_frames), (q_pids, q_frames), args)
        print("embed-only 완료")
        return None

    print("[3/3] 임베딩 (캐시 있으면 생략)...")
    sims: Dict[str, np.ndarray] = {}
    for m in MODELS:
        g_vecs, q_vecs = load_or_embed(m, cache_dir, g_crops, q_crops, (g_pids, g_frames), (q_pids, q_frames), args)
        sims[m] = l2n(q_vecs) @ l2n(g_vecs).T
    del g_crops, q_crops

    print("\n=== 변형별 결과 (mAP / Rank-k, %) ===")
    results = run_variants(sims, q_pids, q_frames, g_pids, g_frames, weights, args.rrf_k, args.prefetch, args.pool, pools)

    out = dict(protocol="PRW GT crops (gallery=test frames GT bbox, query=query_box), junk=same frame same pid",
               gallery_size=int(len(g_pids)), query_total=int(len(q_pids)),
               weights=weights, rrf_k=args.rrf_k, prefetch=args.prefetch, pool=args.pool, pools=pools,
               config=str(Path(args.config).resolve()), config_sha256=args.config_sha256,
               generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), results=results)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.out_csv).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["variant", "mAP", "Rank-1", "Rank-5", "Rank-10", "pool_recall(%)",
                    "queries_all_positives_in_pool(%)", "queries_any_positive_in_pool(%)", "note"])
        for name, r in results.items():
            w.writerow([name, r["mAP"], r["Rank-1"], r["Rank-5"], r["Rank-10"], r["pool_recall(%)"],
                        r["queries_all_positives_in_pool(%)"], r["queries_any_positive_in_pool(%)"], r["note"]])
    print(f"\njson: {args.out_json}\ncsv : {args.out_csv}")
    from bench import ledger
    ledger.record(lambda: ledger.entries_from_unified_result(out, report=Path(args.out_json).resolve(), command=args.command_line,
                                                             versions=ledger.versions_info()),
                  args.ledger, args.no_ledger)
    return out


if __name__ == "__main__":
    main()
