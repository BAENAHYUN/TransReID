#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/people_page.py — '인물 분류': Immich 의 People 처럼, 묶음 결과를 사람별 카드와 파일별 목록으로 본다.

- 위: 묶음 결과(클러스터 실행) 선택 · 폴더 필터 · [사람별 | 파일별] · 불러오기(새로 만들기)
- 사람별: 카드 격자(대표 crop, 이름 또는 #id, 장수·파일 수) → 오른쪽에 그 사람이 나온 파일 목록과 crop
- 파일별: 파일 목록(사람 n명: 이름…) → 오른쪽에 그 파일의 crop 들과 누구인지
- 이름 붙이기(person_names.json), '이 사람으로 검색'(사진에서 찾기로 넘김)
DB 조회는 첫 불러오기 한 번(페이로드만) 이고 결과는 people_index_<method>.json 에 캐시된다.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui import people_index as PI
from gui.thumbs import square_thumb

ROOT = Path(__file__).resolve().parents[1]
CARD = 140
CROP = 96


class _LoadWorker(QThread):
    progress = Signal(int, int)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, run: Dict[str, Any], rebuild: bool, fetch: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        # 주의: 속성 이름을 run 으로 두면 QThread.run() 을 덮어써 스레드가 시작하자마자 죽는다
        self.info, self.rebuild, self.fetch = run, rebuild, fetch

    def run(self) -> None:  # noqa: D401
        try:
            idx = PI.load_or_build(self.info, fetch=self.fetch, progress=lambda a, b: self.progress.emit(a, b), rebuild=self.rebuild)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.done.emit(idx)


class PeoplePage(QWidget):
    searchRequested = Signal(str)     # crop 경로 → '사진에서 찾기' 의 query 로

    def __init__(self, root: Optional[Path] = None, parent: Optional[QWidget] = None, *, fetch: Optional[Callable] = None):
        super().__init__(parent)
        self.root = Path(root) if root is not None else ROOT
        self._fetch = fetch                    # 테스트용: Qdrant 대신 페이로드를 주는 함수
        self.index: Dict[str, Any] = {}
        self.names: Dict[str, str] = {}
        self.run: Optional[Dict[str, Any]] = None
        self.worker: Optional[_LoadWorker] = None
        self.view_mode = "people"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # ---- 상단 카드 ----
        head = QFrame()
        head.setObjectName("searchHeader")
        hv = QVBoxLayout(head)
        hv.setContentsMargins(16, 12, 16, 10)
        row = QHBoxLayout()
        title = QLabel("인물 분류")
        title.setObjectName("resultsTitle")
        row.addWidget(title)
        row.addSpacing(12)
        lab = QLabel("묶음 결과")
        lab.setObjectName("advLabel")
        row.addWidget(lab)
        # 콤보가 가장 긴 항목 폭을 창의 최소 폭으로 요구하지 않게 한다 (긴 실행 이름 → 창이 옆으로 늘어나던 문제)
        self.run_combo = QComboBox()
        self.run_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.run_combo.setMinimumContentsLength(12)
        self.run_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.run_combo, 1)
        lab2 = QLabel("폴더")
        lab2.setObjectName("advLabel")
        row.addWidget(lab2)
        self.folder_combo = QComboBox()
        self.folder_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.folder_combo.setMinimumContentsLength(6)
        self.folder_combo.addItem("전체", "")
        self.folder_combo.currentIndexChanged.connect(lambda *_: self._render())
        row.addWidget(self.folder_combo)
        self.load_btn = QPushButton("불러오기")
        self.load_btn.setObjectName("primaryButton")
        self.load_btn.clicked.connect(lambda: self.load(rebuild=False))
        row.addWidget(self.load_btn)
        self.rebuild_btn = QPushButton("새로 읽기")
        self.rebuild_btn.setToolTip("캐시(people_index)를 버리고 DB 페이로드를 다시 읽는다")
        self.rebuild_btn.clicked.connect(lambda: self.load(rebuild=True))
        row.addWidget(self.rebuild_btn)
        hv.addLayout(row)

        row2 = QHBoxLayout()
        self.mode_people = QToolButton()
        self.mode_people.setObjectName("modeButton")
        self.mode_people.setText("사람별")
        self.mode_people.setCheckable(True)
        self.mode_people.setChecked(True)
        self.mode_files = QToolButton()
        self.mode_files.setObjectName("modeButton")
        self.mode_files.setText("파일별")
        self.mode_files.setCheckable(True)
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self.mode_people)
        grp.addButton(self.mode_files)
        self.mode_people.toggled.connect(lambda on: self._set_mode("people") if on else None)
        self.mode_files.toggled.connect(lambda on: self._set_mode("files") if on else None)
        row2.addWidget(self.mode_people)
        row2.addWidget(self.mode_files)
        row2.addSpacing(12)
        self.summary = QLabel("묶음 결과를 고르고 '불러오기' 를 누르세요 (3. 클러스터 단계의 출력)")
        self.summary.setObjectName("statusLabel")
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # 긴 문장이 최소 폭을 키우지 않게
        row2.addWidget(self.summary, 1)
        self.progress = QProgressBar()
        self.progress.setObjectName("busyBar")
        self.progress.setFixedWidth(160)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        row2.addWidget(self.progress)
        hv.addLayout(row2)
        layout.addWidget(head)

        # ---- 본문: 왼쪽 목록/격자, 오른쪽 상세 ----
        split = QSplitter(Qt.Horizontal)
        self.grid = QListWidget()
        self.grid.setObjectName("resultGrid")
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setIconSize(QSize(CARD, CARD))
        self.grid.setGridSize(QSize(CARD + 20, CARD + 50))
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setWrapping(True)
        self.grid.setSpacing(4)
        self.grid.setUniformItemSizes(True)
        self.grid.setWordWrap(True)
        self.grid.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grid.currentItemChanged.connect(lambda cur, _prev: self._show_person(cur))
        self.grid.itemDoubleClicked.connect(self._rename)

        self.file_list = QListWidget()
        self.file_list.setObjectName("resultGrid")
        self.file_list.currentItemChanged.connect(lambda cur, _prev: self._show_file(cur))

        self.left = QStackedWidget()
        self.left.addWidget(self.grid)
        self.left.addWidget(self.file_list)
        split.addWidget(self.left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        self.detail_title = QLabel("선택하면 상세가 나옵니다")
        self.detail_title.setObjectName("detailTitle")
        self.detail_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # 긴 제목이 최소 폭을 키우지 않게
        rv.addWidget(self.detail_title)
        self.detail_sub = QLabel("")
        self.detail_sub.setObjectName("subtleLabel")
        self.detail_sub.setWordWrap(True)
        self.detail_sub.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        rv.addWidget(self.detail_sub)
        self.detail_list = QListWidget()
        self.detail_list.setObjectName("resultGrid")
        self.detail_list.setViewMode(QListView.IconMode)
        self.detail_list.setIconSize(QSize(CROP, CROP))
        self.detail_list.setGridSize(QSize(CROP + 16, CROP + 46))
        self.detail_list.setResizeMode(QListView.Adjust)
        self.detail_list.setMovement(QListView.Static)
        self.detail_list.setWrapping(True)
        self.detail_list.setWordWrap(True)
        self.detail_list.setUniformItemSizes(True)
        rv.addWidget(self.detail_list, 1)
        btns = QHBoxLayout()
        self.name_btn = QPushButton("이름 붙이기")
        self.name_btn.setEnabled(False)
        self.name_btn.clicked.connect(lambda: self._rename(self.grid.currentItem()))
        btns.addWidget(self.name_btn)
        self.search_btn = QPushButton("이 사람으로 검색")
        self.search_btn.setEnabled(False)
        self.search_btn.clicked.connect(self._search_person)
        btns.addWidget(self.search_btn)
        btns.addStretch(1)
        rv.addLayout(btns)
        right.setMinimumWidth(240)
        split.addWidget(right)
        split.setSizes([700, 400])
        layout.addWidget(split, 1)

        self.refresh_runs()

    # ---- 실행 목록 ----
    def refresh_runs(self) -> None:
        self.runs = PI.find_runs(self.root)
        self.run_combo.clear()
        for r in self.runs:
            when = datetime.fromtimestamp(r["mtime"]).strftime("%m-%d %H:%M")
            self.run_combo.addItem(f"{r['run']} · {r['method']} · {when}" + (" · 캐시" if r["cached"] else ""), r)
        if not self.runs:
            self.summary.setText("묶음 결과가 없습니다 — 사진 처리 3. 클러스터 단계를 먼저 실행하세요")

    def current_run(self) -> Optional[Dict[str, Any]]:
        return self.run_combo.currentData() if self.run_combo.count() else None

    # ---- 불러오기 ----
    def load(self, rebuild: bool = False) -> None:
        run = self.current_run()
        if run is None:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.run = run
        self.names = PI.load_names(run)
        self.load_btn.setEnabled(False)
        self.rebuild_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.summary.setText("불러오는 중… (처음은 DB 페이로드를 읽어 캐시합니다)")
        self.worker = _LoadWorker(run, rebuild, self._fetch, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_loaded)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: (self.load_btn.setEnabled(True), self.rebuild_btn.setEnabled(True), self.progress.setVisible(False)))
        self.worker.start()

    def load_sync(self, rebuild: bool = False) -> None:
        """테스트/스크립트용: 워커 없이 바로."""
        run = self.current_run()
        if run is None:
            return
        self.run = run
        self.names = PI.load_names(run)
        self._on_loaded(PI.load_or_build(run, fetch=self._fetch, rebuild=rebuild))

    def _on_progress(self, a: int, b: int) -> None:
        self.progress.setRange(0, max(1, b))
        self.progress.setValue(a)
        self.summary.setText(f"DB 페이로드 읽는 중 {a:,} / {b:,}")

    def _on_failed(self, msg: str) -> None:
        self.summary.setText("불러오기 실패: " + msg)
        QMessageBox.warning(self, "불러오기 실패", msg)

    def _on_loaded(self, index: Dict[str, Any]) -> None:
        self.index = index
        folders: Dict[str, int] = {}
        for f in index.get("files", {}).values():
            folders[f["folder"]] = folders.get(f["folder"], 0) + 1
        self.folder_combo.blockSignals(True)
        self.folder_combo.clear()
        self.folder_combo.addItem("전체", "")
        for name, n in sorted(folders.items(), key=lambda kv: -kv[1]):
            self.folder_combo.addItem(f"{name} ({n:,}개 파일)", name)
        self.folder_combo.blockSignals(False)
        self._render()

    # ---- 표시 ----
    def _set_mode(self, mode: str) -> None:
        self.view_mode = mode
        self.left.setCurrentIndex(0 if mode == "people" else 1)
        # 보기가 바뀌면 오른쪽 상세는 비운다 (이전 보기의 선택이 남아 있지 않게)
        self.detail_list.clear()
        self.detail_title.setText("선택하면 상세가 나옵니다" if mode == "people" else "파일을 고르면 그 파일에 나온 사람이 나옵니다")
        self.detail_sub.setText("")
        self.name_btn.setEnabled(False)
        self.search_btn.setEnabled(False)
        self._render()

    def _folder(self) -> str:
        return str(self.folder_combo.currentData() or "")

    def visible_clusters(self) -> List[str]:
        folder = self._folder()
        cl = self.index.get("clusters", {})
        ids = [cid for cid, c in cl.items() if cid != PI.NOISE and (not folder or folder in c.get("folders", {}))]
        ids.sort(key=lambda cid: -cl[cid]["size"])
        return ids

    def visible_files(self) -> List[str]:
        folder = self._folder()
        fl = self.index.get("files", {})
        return sorted(f for f, d in fl.items() if not folder or d["folder"] == folder)

    def _render(self) -> None:
        if not self.index:
            return
        cl = self.index["clusters"]
        n_noise = cl.get(PI.NOISE, {}).get("size", 0)
        if self.view_mode == "people":
            self.grid.clear()
            ids = self.visible_clusters()
            for cid in ids:
                c = cl[cid]
                rep = self.index["points"].get(c.get("rep") or "", {})
                pix = square_thumb(PI.resolve_crop(rep.get("crop", ""), self.root), CARD)
                item = QListWidgetItem(QIcon(pix) if pix else QIcon(), f"{PI.display_name(cid, self.names)}\n{c['size']:,}장 · 파일 {c['n_files']:,}")
                item.setData(Qt.UserRole, cid)
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
                item.setToolTip(f"{cid}\n폴더: " + ", ".join(f"{k} {v}" for k, v in c.get("folders", {}).items()))
                self.grid.addItem(item)
            self.summary.setText(f"사람 {len(ids):,}명 · 사진/track {self.index.get('n_points', 0):,}개 · 미분류 {n_noise:,}"
                                 + (f" · 폴더 {self._folder()}" if self._folder() else ""))
        else:
            self.file_list.clear()
            files = self.visible_files()
            fl = self.index["files"]
            for f in files:
                d = fl[f]
                people = [cid for cid in d["clusters"] if cid != PI.NOISE]
                people.sort(key=lambda cid: -len(d["clusters"][cid]))
                names = ", ".join(PI.display_name(cid, self.names) for cid in people[:6]) + (" …" if len(people) > 6 else "")
                extra = f" · 미분류 {len(d['clusters'][PI.NOISE])}" if PI.NOISE in d["clusters"] else ""
                item = QListWidgetItem(f"{Path(f).name}    사람 {len(people)}명: {names or '-'}{extra}")
                item.setData(Qt.UserRole, f)
                item.setToolTip(f)
                self.file_list.addItem(item)
            self.summary.setText(f"파일 {len(files):,}개" + (f" · 폴더 {self._folder()}" if self._folder() else ""))

    def _show_person(self, item: Optional[QListWidgetItem]) -> None:
        self.detail_list.clear()
        if item is None or not self.index:
            self.name_btn.setEnabled(False)
            self.search_btn.setEnabled(False)
            return
        cid = str(item.data(Qt.UserRole))
        c = self.index["clusters"].get(cid)
        if not c:
            return
        self.detail_title.setText(f"{PI.display_name(cid, self.names)}  ·  {c['size']:,}장 · 파일 {c['n_files']:,}개")
        self.detail_sub.setText("나온 파일 (파일마다 crop 하나씩). 두 번 클릭하면 이름을 붙일 수 있습니다.")
        pts = self.index["points"]
        for fname, pids in sorted(c["files"].items(), key=lambda kv: -len(kv[1]))[:200]:
            best = max(pids, key=lambda p: pts.get(p, {}).get("score", 0.0))
            pt = pts.get(best, {})
            pix = square_thumb(PI.resolve_crop(pt.get("crop", ""), self.root), CROP)
            when = f" · {pt['time']}" if pt.get("time") else ""
            it = QListWidgetItem(QIcon(pix) if pix else QIcon(), f"{Path(fname).name}{when}\n{len(pids)}장")
            it.setToolTip(fname)
            it.setData(Qt.UserRole, best)
            it.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            self.detail_list.addItem(it)
        self.name_btn.setEnabled(True)
        self.search_btn.setEnabled(PI.resolve_crop(pts.get(c.get("rep") or "", {}).get("crop", ""), self.root) is not None)

    def _show_file(self, item: Optional[QListWidgetItem]) -> None:
        self.detail_list.clear()
        self.name_btn.setEnabled(False)
        self.search_btn.setEnabled(False)
        if item is None or not self.index:
            return
        fname = str(item.data(Qt.UserRole))
        d = self.index["files"].get(fname)
        if not d:
            return
        pts = self.index["points"]
        n_people = sum(1 for cid in d["clusters"] if cid != PI.NOISE)
        self.detail_title.setText(f"{Path(fname).name}  ·  사람 {n_people}명")
        self.detail_sub.setText(fname)
        for cid, pids in sorted(d["clusters"].items(), key=lambda kv: (kv[0] == PI.NOISE, -len(kv[1]))):
            for pid in pids[:50]:
                pt = pts.get(pid, {})
                pix = square_thumb(PI.resolve_crop(pt.get("crop", ""), self.root), CROP)
                it = QListWidgetItem(QIcon(pix) if pix else QIcon(), f"{PI.display_name(cid, self.names)}\n{pt.get('score', 0.0):.2f}")
                it.setData(Qt.UserRole, pid)
                it.setToolTip(cid)
                it.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
                self.detail_list.addItem(it)

    # ---- 동작 ----
    def _rename(self, item: Optional[QListWidgetItem]) -> None:
        if item is None or self.run is None or self.view_mode != "people":
            return
        cid = str(item.data(Qt.UserRole))
        cur = self.names.get(cid, "")
        name, ok = QInputDialog.getText(self, "이름 붙이기", f"{PI.short_id(cid)} 의 이름 (비우면 지움):", text=cur)
        if not ok:
            return
        self.set_name(cid, name)

    def set_name(self, cid: str, name: str) -> None:
        if self.run is None:
            return
        self.names = PI.save_name(self.run, cid, name)
        cur = self.grid.currentRow()
        self._render()
        if 0 <= cur < self.grid.count():
            self.grid.setCurrentRow(cur)

    def _search_person(self) -> None:
        item = self.grid.currentItem()
        if item is None or not self.index:
            return
        c = self.index["clusters"].get(str(item.data(Qt.UserRole)), {})
        crop = PI.resolve_crop(self.index["points"].get(c.get("rep") or "", {}).get("crop", ""), self.root)
        if crop is not None:
            self.searchRequested.emit(str(crop))
