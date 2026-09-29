#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/check_install.py — 설치 점검. 새 PC 에서 setup.ps1 뒤에 돌린다 (check_install.ps1 이 부른다).

항목마다 [OK] / [없음] / [경고] 를 찍고, 필수 항목이 하나라도 없으면 종료 코드 1.
  1. Python 3.11 · PyTorch CUDA · GPU 이름
  2. 핵심 패키지 import (GUI · 검출 · 임베딩 · 클러스터링 · DB)
  3. pipeline.yaml 이 가리키는 가중치 · 서드파티 폴더
  4. Qdrant 서버 응답 (pipeline.yaml 의 qdrant.url)
  5. GUI 파이프라인 정의(gui_pipelines.json) 로드
  --tests 를 주면 단위 테스트(Qdrant·GPU 불필요, 1~2분)까지.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

PACKAGES = [("PySide6", "GUI"), ("torch", "모델 실행"), ("transformers", "SigLIP2·Qwen"), ("qdrant_client", "벡터 DB"),
            ("ultralytics", "YOLO26 검출"), ("rfdetr", "RF-DETR 검출"), ("boxmot", "BoT-SORT 추적"),
            ("leidenalg", "Leiden 클러스터링"), ("igraph", "Leiden 그래프"), ("sklearn", "DBSCAN"),
            ("torch_geometric", "SUSHI 스티칭 (영상)"), ("sentencepiece", "한→영 번역"), ("yaml", "설정")]
# (경로, 설명, 필수 여부) — pipeline.yaml 의 retrievers / stitcher 설정과 같다
WEIGHTS = [("weights/IRRA/cuhk_pedes/best.pth", "IRRA 가중치", True),
           ("weights/IRRA/cuhk_pedes/configs.yaml", "IRRA 설정", True),
           ("weights/IRRA/ViT-B-16.pt", "IRRA 의 CLIP 백본", True),
           ("weights/SOLIDER/solider_market_swin_base.pth", "SOLIDER 가중치", True),
           ("IRRA", "IRRA 코드", True), ("third_party/SOLIDER", "SOLIDER 코드", True),
           ("third_party/SUSHI", "SUSHI 코드 (영상 처리만)", False),
           ("third_party/SUSHI/pretrained_models/mot17private.pth", "SUSHI 스티칭 모델 (영상 처리만)", False),
           ("third_party/SUSHI/fastreid-models/model_weights/msmt_bot_R50-ibn.pth", "SUSHI 용 fast-reid (영상 처리만)", False)]

failed = 0


def report(ok: bool, what: str, detail: str = "", required: bool = True) -> None:
    global failed
    tag = "[OK]  " if ok else ("[없음]" if required else "[경고]")
    if not ok and required:
        failed += 1
    print(f"{tag} {what}" + (f" — {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tests", action="store_true", help="단위 테스트까지 돌린다")
    args = ap.parse_args()

    print("== 1. Python · GPU")
    report(sys.version_info[:2] == (3, 11), f"Python {sys.version.split()[0]}", "3.11 이 필요합니다" if sys.version_info[:2] != (3, 11) else "")
    try:
        import torch
        cuda = torch.cuda.is_available()
        report(cuda, f"PyTorch {torch.__version__} CUDA", torch.cuda.get_device_name(0) if cuda else "GPU 를 못 찾음 — 검색·라벨은 매우 느리거나 실패합니다",
               required=False)
    except Exception as exc:  # noqa: BLE001
        report(False, "PyTorch", f"{type(exc).__name__}: {exc}")

    print("== 2. 패키지")
    for mod, why in PACKAGES:
        try:
            m = importlib.import_module(mod)
            report(True, f"{mod} {getattr(m, '__version__', '')}".rstrip(), why)
        except Exception as exc:  # noqa: BLE001
            report(False, mod, f"{why} · {type(exc).__name__}: {exc}", required=mod not in ("torch_geometric",))

    print("== 3. 가중치 · 서드파티 (README 2.3 · 2.4)")
    for rel, why, required in WEIGHTS:
        p = ROOT / rel
        size = f"{p.stat().st_size / 1048576:,.0f} MB" if p.is_file() else ("폴더" if p.is_dir() else "")
        report(p.exists(), rel, f"{why} {size}".strip(), required)

    print("== 4. Qdrant")
    url = "http://localhost:6333"
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8")) or {}
        url = str((cfg.get("qdrant") or {}).get("url") or url)
    except Exception:  # noqa: BLE001
        pass
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/collections", timeout=5) as r:
            names = [c["name"] for c in json.loads(r.read().decode("utf-8"))["result"]["collections"]]
        report(True, f"Qdrant {url}", f"컬렉션 {len(names)}개" + ("" if names else " (비어 있음 — 사진 처리/영상 처리로 DB 를 만드세요)"))
    except Exception as exc:  # noqa: BLE001
        report(False, f"Qdrant {url}", f"응답 없음 ({type(exc).__name__}) — docker compose -f docker-compose.qdrant.yml up -d")

    print("== 5. GUI 파이프라인 정의")
    try:
        from gui import pipeline_page as pp
        groups = pp.load_registry()
        report(True, "gui_pipelines.json", ", ".join(f"{g['id']} {len(g['stages'])}단계" for g in groups))
    except Exception as exc:  # noqa: BLE001
        report(False, "gui_pipelines.json", f"{type(exc).__name__}: {exc}")

    if args.tests:
        print("== 6. 단위 테스트")
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        code = subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=str(ROOT), env=env)
        report(code == 0, "unittest", "통과" if code == 0 else f"종료 코드 {code}")

    print()
    print("점검 결과: " + ("모두 준비됨" if not failed else f"필수 항목 {failed}개 없음 — 위 [없음] 을 먼저 해결하세요"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
