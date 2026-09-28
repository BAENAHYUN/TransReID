"""bench/splits.py — 인물(pid) 단위 데이터 분할 (탐색용 tune / 최종 비교용 holdout).

Optuna(P3)는 tune 부분으로 조합·하이퍼파라미터를 고르고, 최종 비교는 holdout 으로 한다.
분할은 seed 로 결정적이며 파일에 고정된다 (같은 파일을 쓰면 누가 돌려도 같은 인물 집합).

  python bench/splits.py make [--data-root ./data/PRW] [--seed 42] [--part tune=0.3 --part holdout=0.7]
                              [--out bench/splits/prw_pids_seed42.json]
  python bench/splits.py show bench/splits/prw_pids_seed42.json

평가 스크립트 인자: --pid-split <파일>:<부분>  (prw_cluster_gt_eval.py, prw_e2e_search_eval.py)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "bench" / "splits"


def split_ids(ids: Sequence[int], parts: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    """ids 를 비율대로 나눈다 (seed 결정적, 비율 합은 1 이어야 하며 마지막 부분이 나머지를 받는다)."""
    if not parts:
        raise ValueError("parts 가 비었습니다")
    for name, frac in parts.items():
        if not (0.0 < float(frac) <= 1.0):
            raise ValueError(f"비율은 0 초과 1 이하: {name}={frac}")
    total = sum(parts.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"비율 합이 1 이어야 합니다: {total}")
    order = sorted(set(int(i) for i in ids))
    rng = random.Random(seed)
    rng.shuffle(order)
    out: Dict[str, List[int]] = {}
    start = 0
    names = list(parts)
    for i, name in enumerate(names):
        n = len(order) - start if i == len(names) - 1 else int(round(len(order) * parts[name]))
        out[name] = sorted(order[start:start + n])
        start += n
    return out


def prw_query_pids(data_root: Path) -> Tuple[List[int], Dict[int, int]]:
    """query_info.txt 의 인물 id 와 인물별 쿼리 수."""
    counts: Dict[int, int] = {}
    with (data_root / "query_info.txt").open() as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                pid = int(parts[0])
                counts[pid] = counts.get(pid, 0) + 1
    return sorted(counts), counts


def make_split(data_root: Path, parts: Dict[str, float], seed: int, dataset: str = "prw") -> Dict[str, Any]:
    ids, counts = prw_query_pids(data_root)
    split = split_ids(ids, parts, seed)
    digest = hashlib.sha1(json.dumps(split, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return {
        "dataset": dataset, "unit": "pid (PRW query_info 인물)", "seed": seed, "fractions": parts,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "digest": digest,
        "counts": {name: {"pids": len(pids), "queries": sum(counts[p] for p in pids)} for name, pids in split.items()},
        "parts": split,
    }


def load_split(path: Path) -> Dict[str, Any]:
    """분할 파일 로드 + 무결성 검사: parts 존재, 부분끼리 겹치지 않음, digest(있으면) 일치."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict) or not isinstance(d.get("parts"), dict) or not d["parts"]:
        raise ValueError(f"분할 파일 형식이 아닙니다: {path}")
    seen: Dict[int, str] = {}
    for name, ids in d["parts"].items():
        for x in ids:
            x = int(x)
            if x in seen:
                raise ValueError(f"분할 {path}: pid {x} 가 '{seen[x]}' 와 '{name}' 에 모두 있음")
            seen[x] = name
    if d.get("digest"):
        digest = hashlib.sha1(json.dumps({k: sorted(int(x) for x in v) for k, v in d["parts"].items()}, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        if digest != d["digest"]:
            raise ValueError(f"분할 {path}: digest 불일치 (파일이 손으로 바뀜?) {d['digest']} != {digest}")
    return d


def parse_split_arg(text: str) -> Tuple[Path, str]:
    """'파일:부분' → (Path, 부분). 윈도 드라이브 문자(C:\\…)와 구분하기 위해 마지막 ':' 로 나눈다."""
    if ":" not in text:
        raise ValueError(f"--pid-split 은 <파일>:<부분> 형식: {text!r}")
    path, part = text.rsplit(":", 1)
    # 부분 이름은 단순 식별자여야 한다 — 'C:\x\s.json' 처럼 드라이브 문자만 있는 경로를 부분으로 오인하지 않게
    if not path or not part or not all(c.isalnum() or c in "_-" for c in part) or len(path) < 2:
        raise ValueError(f"--pid-split 은 <파일>:<부분> 형식 (부분 = 영숫자/_/-): {text!r}")
    return Path(path), part


def load_split_arg(text: str) -> Set[int]:
    path, part = parse_split_arg(text)
    d = load_split(path)
    if part not in d["parts"]:
        raise ValueError(f"분할 {path} 에 부분 '{part}' 이 없습니다 (있는 것: {list(d['parts'])})")
    return set(int(x) for x in d["parts"][part])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="인물 단위 분할 파일 만들기/보기")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("make")
    s.add_argument("--data-root", default="./data/PRW")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--part", action="append", default=[], help="이름=비율 (기본 tune=0.3 holdout=0.7)")
    s.add_argument("--out", default=None, help=f"기본 {SPLIT_DIR}/prw_pids_seed<seed>.json")
    s = sub.add_parser("show")
    s.add_argument("path")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "make":
        parts: Dict[str, float] = {}
        for item in args.part or ["tune=0.3", "holdout=0.7"]:
            k, v = item.split("=", 1)
            parts[k.strip()] = float(v)
        d = make_split(Path(args.data_root).expanduser().resolve(), parts, args.seed)
        out = Path(args.out) if args.out else SPLIT_DIR / f"prw_pids_seed{args.seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"[split] {out} · {d['counts']} · digest {d['digest']}")
        return 0
    d = load_split(Path(args.path))
    print(json.dumps({k: v for k, v in d.items() if k != "parts"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
