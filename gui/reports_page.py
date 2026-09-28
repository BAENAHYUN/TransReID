#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/reports_page.py — '결과 보기': 사람이 할 라벨링 시트와 생성된 리포트 HTML 을 한 곳에서 연다 (Immich 의 앨범 격).

- 라벨링 시트: eval/gt/tracks/<영상>/sheet.html · eval/gt/object_pairs/sheet.html · eval/gt/qwen/sheet.html
  각 시트의 항목 수(proposals.json)와 labels.json 유무·검토 수·manifest 일치 여부를 보여 준다.
- 리포트: outputs/image_review, outputs/image_db_html, outputs/clustering/**/gallery, outputs/audit 의 HTML (수정 시각순).
열기는 기본 브라우저(QDesktopServices). 파일을 읽기만 하고 아무것도 바꾸지 않는다.
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parents[1]

SHEET_KINDS = {"track_labels": "추적 (구간 → 사람 id)", "object_pair_labels": "객체 재출현 (같은 개체?)", "qwen_labels": "Qwen 판정 (설명에 맞는 사람?)"}
REPORT_GLOBS = [
    ("결과 인덱스", "outputs/image_review/*.html"),
    ("DB 리포트", "outputs/image_db_html/*.html"),
    ("클러스터 갤러리", "outputs/clustering/*/*/gallery/*.html"),
    ("클러스터 갤러리", "outputs/clustering/*/gallery/*.html"),
    ("문서 (기준표·로드맵·요약)", "outputs/audit/*.html"),
]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


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


def scan_reports(root: Path, limit: int = 60) -> List[Dict[str, Any]]:
    seen: set = set()
    rows: List[Dict[str, Any]] = []
    for kind, pattern in REPORT_GLOBS:
        for p in root.glob(pattern):
            if not p.is_file() or p in seen:
                continue
            seen.add(p)
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            rows.append({"kind": kind, "name": p.relative_to(root).as_posix(), "mtime": mtime, "path": p, "folder": p.parent})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows[:limit]


def _open(path: Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))


class ReportsPage(QWidget):
    def __init__(self, root: Optional[Path] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.root = Path(root) if root is not None else ROOT
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        head = QHBoxLayout()
        title = QLabel("결과 보기")
        title.setObjectName("resultsTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.refresh)
        head.addWidget(self.refresh_btn)
        layout.addLayout(head)

        lab1 = QLabel("정답 라벨링 시트 — 사람이 할 일. 시트를 열어 판정하고 내려받은 labels.json 을 같은 폴더에 두면 평가(평가 / 비교 12·13·14)가 읽습니다.")
        lab1.setObjectName("subtleLabel")
        lab1.setWordWrap(True)
        layout.addWidget(lab1)
        self.sheets = self._table(["종류", "이름", "항목", "상태", "", ""], stretch_col=3)
        layout.addWidget(self.sheets, 1)

        lab2 = QLabel("생성된 리포트 — 결과창·갤러리·DB 리포트·문서 (최근 수정순)")
        lab2.setObjectName("subtleLabel")
        layout.addWidget(lab2)
        self.reports = self._table(["종류", "파일", "수정", "", ""], stretch_col=1)
        layout.addWidget(self.reports, 2)
        self.refresh()

    @staticmethod
    def _table(headers: List[str], stretch_col: int) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setShowGrid(False)
        t.horizontalHeader().setStretchLastSection(False)
        t.setProperty("stretchCol", stretch_col)
        return t

    def _fill(self, table: QTableWidget, rows: List[List[Any]], buttons: List[List[Any]]) -> None:
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
                b.clicked.connect(lambda _=False, p=path: _open(p))
                table.setCellWidget(r, c, b)
        # 이름/파일은 내용 폭대로, 긴 설명 열(상태 / 파일) 하나만 늘어난다
        table.resizeColumnsToContents()
        stretch = int(table.property("stretchCol") or 1)
        for c in range(table.columnCount()):
            table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch if c == stretch else QHeaderView.ResizeToContents)

    def refresh(self) -> None:
        self.sheet_rows = scan_sheets(self.root)
        self._fill(self.sheets, [[s["kind"], s["name"], s["items"], s["status"]] for s in self.sheet_rows],
                   [[("시트 열기", s["path"]), ("폴더", s["folder"])] for s in self.sheet_rows])
        self.report_rows = scan_reports(self.root)
        self._fill(self.reports, [[r["kind"], r["name"], datetime.fromtimestamp(r["mtime"]).strftime("%m-%d %H:%M")] for r in self.report_rows],
                   [[("열기", r["path"]), ("폴더", r["folder"])] for r in self.report_rows])
