#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/reports_page.py — '결과 보기'(사진 처리·영상 처리가 만든 산출물만) 와 '정답 라벨링'(평가용 시트) 페이지.

결과 보기: outputs/image_review(결과창 인덱스) · outputs/image_db_html(DB 리포트) · outputs/clustering/<실행>/<대상>/gallery*/(묶음 갤러리)
  를 사진/영상 구분·실행 이름·수정 시각으로 나열한다. 개발 문서(outputs/audit)나 평가 시트는 여기 넣지 않는다.
정답 라벨링: eval/gt 의 시트 3종과 labels.json 진행 상태 — 사람이 할 일이라 '평가' 섹션에 둔다.
열기는 GUI 안 뷰어(HtmlView), '브라우저' 는 외부 브라우저. 파일을 읽기만 한다.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.html_view import HtmlView

ROOT = Path(__file__).resolve().parents[1]

SHEET_KINDS = {"track_labels": "추적 (구간 → 사람 id)", "object_pair_labels": "객체 재출현 (같은 개체?)", "qwen_labels": "Qwen 판정 (설명에 맞는 사람?)"}


# ---------------------------------------------------------------- 공용
def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _open(path: Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))


# ---------------------------------------------------------------- 결과 보기: 사진/영상 처리 산출물
def _run_media(run_dir: Path, target_dir: Path) -> str:
    """클러스터 실행이 사진인지 영상인지: 실행 이름(video/track) → report.json 설정(media_type video / video_tracks) → 기본 사진."""
    name = run_dir.name.lower()
    if "video" in name or "track" in name:
        return "영상"
    for rp in sorted(target_dir.glob("*_report.json")):
        d = _load_json(rp) or {}
        cfg = d.get("config") if isinstance(d.get("config"), dict) else {}
        blob = json.dumps(cfg, ensure_ascii=False).lower()
        if '"media_type": "video"' in blob or "video_tracks" in blob or "track_centroid" in blob:
            return "영상"
    return "사진"


def scan_reports(root: Path, limit: int = 80) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(path: Path, media: str, kind: str, name: str) -> None:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        rows.append({"media": media, "kind": kind, "name": name, "path": path, "folder": path.parent, "mtime": mtime,
                     "rel": path.relative_to(root).as_posix()})

    for p in (root / "outputs" / "image_review").glob("*.html"):
        # 결과창은 DB 리포트(image_db_*.html)도 같은 폴더에 만든다
        add(p, "사진", "DB 리포트" if p.stem.startswith("image_db_") else "결과창 인덱스", p.stem)
    for p in (root / "outputs" / "image_db_html").glob("*.html"):
        add(p, "사진", "DB 리포트", p.stem)
    cl = root / "outputs" / "clustering"
    if cl.is_dir():
        for run in cl.iterdir():
            if not run.is_dir():
                continue
            for target in ("person", "object"):
                tdir = run / target
                if not tdir.is_dir():
                    continue
                htmls = [p for g in tdir.glob("gallery*") if g.is_dir() for p in g.glob("*.html")]
                if not htmls:
                    continue
                media = _run_media(run, tdir)
                for p in htmls:
                    add(p, media, f"{'사람' if target == 'person' else '물건'} 묶음 갤러리", f"{run.name} · {p.stem}")
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows[:limit]


class _TablePage(QWidget):
    """표 + 내장 뷰어 스택 공용 뼈대."""

    def __init__(self, root: Optional[Path], title: str, intro: str, headers: List[str], stretch_col: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.root = Path(root) if root is not None else ROOT
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack)
        list_page = QWidget()
        self.stack.addWidget(list_page)
        self.viewer = HtmlView(back_label="← 목록")
        self.viewer.back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        viewer_page = QWidget()
        vl = QVBoxLayout(viewer_page)
        vl.setContentsMargins(16, 12, 16, 12)
        vl.addWidget(self.viewer)
        self.stack.addWidget(viewer_page)

        layout = QVBoxLayout(list_page)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)
        head = QHBoxLayout()
        t = QLabel(title)
        t.setObjectName("resultsTitle")
        head.addWidget(t)
        head.addStretch(1)
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.refresh)
        head.addWidget(self.refresh_btn)
        layout.addLayout(head)
        lab = QLabel(intro)
        lab.setObjectName("subtleLabel")
        lab.setWordWrap(True)
        layout.addWidget(lab)
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.stretch_col = stretch_col
        layout.addWidget(self.table, 1)

    def _fill(self, rows: List[List[Any]], buttons: List[List[Any]]) -> None:
        table = self.table
        table.setRowCount(0)
        for r, (cells, btns) in enumerate(zip(rows, buttons)):
            table.insertRow(r)
            for c, text in enumerate(cells):
                item = QTableWidgetItem("" if text is None else str(text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(r, c, item)
            for c, (label, path) in enumerate(btns, start=len(cells)):
                b = QPushButton(label)
                b.setFixedHeight(26)
                if label in ("열기", "시트 열기"):
                    b.clicked.connect(lambda _=False, p=path: self.show(p))          # GUI 안에서
                else:
                    b.clicked.connect(lambda _=False, p=path: _open(p))              # 브라우저 / 폴더
                table.setCellWidget(r, c, b)
        table.resizeColumnsToContents()
        for c in range(table.columnCount()):
            table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch if c == self.stretch_col else QHeaderView.ResizeToContents)

    def show(self, path: Path) -> None:
        self.viewer.load(path)
        self.stack.setCurrentIndex(1)

    def refresh(self) -> None:  # pragma: no cover - 하위 클래스가 채운다
        raise NotImplementedError


class ReportsPage(_TablePage):
    def __init__(self, root: Optional[Path] = None, parent: Optional[QWidget] = None):
        super().__init__(root, "결과 보기",
                         "사진 처리·영상 처리가 만든 결과 (최근 수정순): 결과창 인덱스 · DB 리포트 · 사람/물건 묶음 갤러리. '열기' 는 여기서, '브라우저' 는 밖에서 봅니다.",
                         ["구분", "종류", "실행 · 파일", "수정", "", "", ""], stretch_col=2, parent=parent)
        self.refresh()

    def refresh(self) -> None:
        self.report_rows = scan_reports(self.root)
        self._fill([[r["media"], r["kind"], r["name"], datetime.fromtimestamp(r["mtime"]).strftime("%m-%d %H:%M")] for r in self.report_rows],
                   [[("열기", r["path"]), ("브라우저", r["path"]), ("폴더", r["folder"])] for r in self.report_rows])
        for r, row in enumerate(self.report_rows):
            self.table.item(r, 2).setToolTip(row["rel"])

    # 호환: 예전 이름
    @property
    def reports(self) -> QTableWidget:
        return self.table


# ---------------------------------------------------------------- 정답 라벨링 (평가)
def sheet_items(proposals: Dict[str, Any]) -> Optional[int]:
    kind = proposals.get("kind")
    if kind == "track_labels":
        n = proposals.get("n_segments")
        return int(n) if n is not None else (len(proposals.get("short_tracks") or []) or None)
    if kind == "object_pair_labels":
        return len(proposals.get("pairs") or []) or None
    if kind == "qwen_labels":
        q = proposals.get("queries") or []
        k = int(proposals.get("top_k") or 0)
        return (len(q) * k) if q and k else None
    return None


def label_status(folder: Path, proposals: Dict[str, Any]) -> str:
    lp = folder / "labels.json"
    if not lp.is_file():
        return "라벨 없음 — 시트를 열어 판정하고 labels.json 을 이 폴더에 저장"
    d = _load_json(lp)
    if d is None:
        return "labels.json 을 읽을 수 없음"
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    manifest = d.get("manifest") or meta.get("manifest")
    reviewed, items = d.get("reviewed"), d.get("items")
    if reviewed is None:
        labels = d.get("labels") or {}
        reviewed = sum(1 for v in labels.values() if isinstance(v, dict) and v.get("reviewed") is True)
        items = items or len(labels)
    s = f"labels.json · 검토 {reviewed}/{items}"
    if proposals.get("manifest") and manifest and manifest != proposals.get("manifest"):
        s += " · manifest 불일치 (시트가 다시 만들어짐 — 평가가 거부함)"
    return s


def scan_sheets(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    gt = root / "eval" / "gt"
    folders = sorted(p.parent for p in gt.glob("tracks/*/sheet.html")) + [gt / "object_pairs", gt / "qwen"]
    for folder in folders:
        sheet = folder / "sheet.html"
        if not sheet.is_file():
            continue
        proposals = _load_json(folder / "proposals.json") or {}
        kind = str(proposals.get("kind") or "")
        name = folder.name if kind == "track_labels" else SHEET_KINDS.get(kind, folder.name)
        out.append({"kind": SHEET_KINDS.get(kind, kind or "시트"), "name": name, "items": sheet_items(proposals), "status": label_status(folder, proposals),
                    "path": sheet, "folder": folder})
    return out


class LabelingPage(_TablePage):
    def __init__(self, root: Optional[Path] = None, parent: Optional[QWidget] = None):
        super().__init__(root, "정답 라벨링",
                         "사람이 할 일: 시트를 열어 판정하고 내려받은 labels.json 을 같은 폴더에 두면 평가(평가 / 비교 12·13·14)가 읽습니다. 순서는 추적 → 객체 → Qwen.",
                         ["종류", "이름", "항목", "상태", "", "", ""], stretch_col=3, parent=parent)
        self.refresh()

    def refresh(self) -> None:
        self.sheet_rows = scan_sheets(self.root)
        self._fill([[s["kind"], s["name"], s["items"], s["status"]] for s in self.sheet_rows],
                   [[("시트 열기", s["path"]), ("브라우저", s["path"]), ("폴더", s["folder"])] for s in self.sheet_rows])

    @property
    def sheets(self) -> QTableWidget:
        return self.table
