#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/people_page.py — '인물 분류': Immich 의 People 처럼, 묶음 결과를 사람(또는 물건)별 카드와 파일별 목록으로 본다.

- 위: 묶음 결과(클러스터 실행) 선택 · 폴더 필터 · 불러오기 / 새로 읽기
- 둘째 줄: 대상(사람/물건) · [사람별 | 파일별] · 요약 · 라벨 도구 · 없으면 자동 · 자동 라벨 붙이기
- 사람별: 카드 격자(대표 crop, 이름, 장수·파일 수) → 오른쪽에 그 사람이 나온 파일 목록과 crop
- 파일별: 파일 목록(사람 n명: 이름…) → 오른쪽에 그 파일의 crop 들과 누구인지
- 카드 이름 = 이 결과에서 붙인 이름 > 다른 결과에서 이어받은 이름(구성원 과반 공유) > 자동 라벨 > #id
- 자동 라벨: 같은 폴더의 labels_qwen*/labels_vec*/cluster_labels.jsonl (사진 처리 8·8b / 영상 처리 7·7b 출력).
  라벨 파일이 없는 결과를 불러오면 색상 라벨러가 저절로 돈다('없으면 자동'). 라벨러 로그는 결과 폴더의 auto_label_<도구>.log.
- '이 사람으로 검색': 대표 crop 으로 바로 검색 — 사진 결과는 '사진에서 찾기', 영상 결과는 '영상에서 찾기'
  (searchRequested(crop, media, target) 를 search_gui 가 받아 화면을 고른다).
DB 조회는 첫 불러오기 한 번(페이로드만) 이고 결과는 people_index_<method>.json 에 캐시된다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
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
# 대상별 낱말: (명사, 단위)
NOUNS = {"person": ("사람", "명"), "object": ("물건", "개")}


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
    # (대표 crop 경로, 'image'|'video', 'person'|'object') → search_gui 가 사진/영상 검색 화면을 골라 바로 검색한다
    searchRequested = Signal(str, str, str)

    def __init__(self, root: Optional[Path] = None, parent: Optional[QWidget] = None, *, fetch: Optional[Callable] = None,
                 auto_label_on_load: bool = True):
        super().__init__(parent)
        self.root = Path(root) if root is not None else ROOT
        self._fetch = fetch                    # 테스트용: Qdrant 대신 페이로드를 주는 함수
        self.target = "person"
        self.index: Dict[str, Any] = {}
        self.names: Dict[str, str] = {}
        self.inherited: Dict[str, Dict[str, Any]] = {}    # cluster_id → 다른 결과에서 이어받은 이름 (people_index.inherited_names)
        self.labels: Dict[str, Dict[str, Any]] = {}       # cluster_id → 자동 라벨 (people_index.load_labels)
        self.run: Optional[Dict[str, Any]] = None
        self.runs: List[Dict[str, Any]] = []
        self.worker: Optional[_LoadWorker] = None
        self.label_worker = None                           # gui.pipeline_page.ProcessWorker (자동 라벨 붙이기)
        self._label_log: List[str] = []
        self._label_log_file: Optional[TextIO] = None
        self.label_log_path: Optional[Path] = None
        self._label_run: Optional[Dict[str, Any]] = None  # 라벨러가 돌고 있는 실행 (도는 사이 다른 실행을 불러와도 섞이지 않게)
        self._label_auto = False                           # 불러오기 때 저절로 시작했으면 실패해도 대화상자를 띄우지 않는다
        self._auto_tried: set = set()                      # 이 세션에서 자동 라벨을 이미 시도한 assignments (실패 반복 방지)
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
        self.run_combo.setToolTip("클러스터 실행 결과 (outputs/clustering/<실행>/<대상>/).\n"
                                  "항목 = 실행 · 방법 · 만든 시각 · 캐시 여부 · 라벨 종류. 마지막으로 연 결과가 기본으로 골라집니다.")
        row.addWidget(self.run_combo, 1)
        lab2 = QLabel("폴더")
        lab2.setObjectName("advLabel")
        row.addWidget(lab2)
        self.folder_combo = QComboBox()
        self.folder_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.folder_combo.setMinimumContentsLength(6)
        self.folder_combo.addItem("전체", "")
        self.folder_combo.setToolTip("사진은 image_id 의 앞 경로(PRW, coco 등), 영상은 videos 로 거릅니다. 괄호 안은 파일 수")
        self.folder_combo.currentIndexChanged.connect(lambda *_: self._render())
        row.addWidget(self.folder_combo)
        self.load_btn = QPushButton("불러오기")
        self.load_btn.setObjectName("primaryButton")
        self.load_btn.setToolTip("고른 결과를 엽니다. 처음은 DB 페이로드를 읽어 캐시(people_index_<방법>.json)하고, 다음부터는 캐시로 바로 엽니다")
        self.load_btn.clicked.connect(lambda: self.load(rebuild=False))
        row.addWidget(self.load_btn)
        self.rebuild_btn = QPushButton("새로 읽기")
        self.rebuild_btn.setToolTip("캐시(people_index)를 버리고 DB 페이로드를 다시 읽는다")
        self.rebuild_btn.clicked.connect(lambda: self.load(rebuild=True))
        row.addWidget(self.rebuild_btn)
        hv.addLayout(row)

        row2 = QHBoxLayout()
        # 대상은 첫 줄이 아니라 여기 — 첫 줄에 콤보를 더하면 창 최소 폭이 늘어난다
        self.target_combo = QComboBox()
        for key, (noun, _unit) in NOUNS.items():
            self.target_combo.addItem(noun, key)
        self.target_combo.setToolTip("사람 군집(person_*_assignments) 과 물건 군집(object_*_assignments) 중 무엇을 볼지")
        self.target_combo.currentIndexChanged.connect(lambda *_: self.set_target(str(self.target_combo.currentData())))
        row2.addWidget(self.target_combo)
        self.mode_people = QToolButton()
        self.mode_people.setObjectName("modeButton")
        self.mode_people.setText("사람별")
        self.mode_people.setCheckable(True)
        self.mode_people.setChecked(True)
        self.mode_people.setToolTip("카드 격자 — 군집 하나가 카드 하나")
        self.mode_files = QToolButton()
        self.mode_files.setObjectName("modeButton")
        self.mode_files.setText("파일별")
        self.mode_files.setCheckable(True)
        self.mode_files.setToolTip("파일 목록 — 파일마다 나온 사람과 이름")
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self.mode_people)
        grp.addButton(self.mode_files)
        self.mode_people.toggled.connect(lambda on: self._set_mode("people") if on else None)
        self.mode_files.toggled.connect(lambda on: self._set_mode("files") if on else None)
        row2.addWidget(self.mode_people)
        row2.addWidget(self.mode_files)
        row2.addStretch(1)
        # 요약 줄은 셋째 줄에 전체 폭으로 — 둘째 줄에 두면 960px 창에서 라벨 도구에 밀려 한두 글자만 보였다
        self.summary = QLabel("")
        self.summary.setObjectName("statusLabel")
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # 긴 문장이 최소 폭을 키우지 않게
        self._status("묶음 결과를 고르고 '불러오기' 를 누르세요 (3. 클러스터 단계의 출력)")
        # 진행 표시는 이 줄이 아니라 헤더 아래 전체 폭의 얇은 막대로 둔다 (아래 hv.addWidget). 폭 160 고정으로 이 줄에 두면
        # 켜지는 순간 라벨 도구와 합쳐 창 최소 폭이 957 → 1027 로 늘고, Qt 는 창을 다시 줄이지 않는다.
        self.progress = QProgressBar()
        self.progress.setObjectName("busyBar")
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        # 자동 라벨 붙이기: 라벨러를 골라 이 실행의 assignments 에 돌린다 (결과는 같은 폴더의 labels_vec/ 또는 labels_qwen/)
        self.label_tool = QComboBox()
        self.label_tool.addItem("색상 라벨 (SigLIP2 벡터 · 빠름)", "vec")
        self.label_tool.addItem("문장 라벨 (Qwen3-VL · GPU · 느림)", "qwen")
        self.label_tool.setToolTip("색상: DB 에 있는 SigLIP2 벡터만 써서 '노란색 상의' · '빨간색 자동차' 같은 색 라벨을 붙인다 (30초~3분).\n"
                                   "문장: Qwen3-VL 이 대표 crop 몽타주를 보고 '노란 반팔에 검은 바지' 같은 문장을 만든다\n"
                                   "(GPU, 군집당 약 0.6초 + 시작 1~2분, 결과가 있으면 빠진 군집만 채움).")
        self.label_tool.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.label_tool.setMinimumContentsLength(10)
        row2.addWidget(self.label_tool)
        self.auto_check = QCheckBox("없으면 자동")
        self.auto_check.setChecked(auto_label_on_load)
        self.auto_check.setToolTip("라벨 파일이 없는 묶음 결과(사진·영상, 사람·물건 모두)를 불러오면 색상 라벨을 저절로 만든다\n"
                                   "(30초~3분, 끝나면 카드가 바뀜). GUI 를 켠 동안 결과마다 한 번만 시도합니다.")
        row2.addWidget(self.auto_check)
        self.label_btn = QPushButton("자동 라벨 붙이기")
        self.label_btn.setEnabled(False)
        self.label_btn.setToolTip("이름 없는 카드에 옷차림·색 라벨을 자동으로 붙인다. 도는 동안 '중단' 으로 바뀝니다.\n"
                                  "로그는 결과 폴더의 auto_label_<도구>.log. 먼저 묶음 결과를 불러오세요.")
        self.label_btn.clicked.connect(self.auto_label)
        row2.addWidget(self.label_btn)
        hv.addLayout(row2)
        hv.addWidget(self.summary)
        hv.addWidget(self.progress)
        layout.addWidget(head)

        # ---- 본문: 왼쪽 목록/격자, 오른쪽 상세 ----
        split = QSplitter(Qt.Horizontal)
        self.grid = QListWidget()
        self.grid.setObjectName("resultGrid")
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setIconSize(QSize(CARD, CARD))
        # 글자 세 줄(두 줄로 접힌 라벨 + '652장 · 파일 581') 이 들어가게. 카드마다 setSizeHint(gridSize) 도 준다 —
        # 안 주면 uniform 크기가 첫 카드(두 줄)에서 정해져 긴 라벨의 셋째 줄이 '…' 로 잘린다
        self.grid.setGridSize(QSize(CARD + 20, CARD + 3 * self.grid.fontMetrics().lineSpacing() + 20))
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
        self.name_btn.setToolTip("고른 카드의 이름을 정합니다 (카드 두 번 클릭과 같음). 입력칸을 비우고 확인하면 이름을 지우고 자동 라벨로 돌아갑니다")
        self.name_btn.clicked.connect(lambda: self._rename(self.grid.currentItem()))
        btns.addWidget(self.name_btn)
        self.search_btn = QPushButton("이 사람으로 검색")
        self.search_btn.setEnabled(False)
        self.search_btn.setToolTip("대표 crop 으로 바로 검색합니다. 사진 결과는 '사진에서 찾기', 영상 결과는 '영상에서 찾기' 로 갑니다")
        self.search_btn.clicked.connect(self._search_person)
        btns.addWidget(self.search_btn)
        btns.addStretch(1)
        rv.addLayout(btns)
        right.setMinimumWidth(240)
        split.addWidget(right)
        split.setSizes([700, 400])
        layout.addWidget(split, 1)

        self.refresh_runs()

    # ---- 낱말 ----
    def noun(self) -> str:
        return NOUNS.get(self.target, NOUNS["person"])[0]

    def count_word(self, n: int) -> str:
        noun, unit = NOUNS.get(self.target, NOUNS["person"])
        return f"{noun} {n:,}{unit}"

    def _status(self, text: str) -> None:
        """요약 줄. 창이 좁으면 끝이 잘리므로 툴팁에 전문을 둔다."""
        self.summary.setText(text)
        self.summary.setToolTip(text)

    # ---- 대상 · 실행 목록 ----
    def set_target(self, target: str) -> None:
        if target not in NOUNS or target == self.target and self.runs:
            return
        self.target = target
        i = self.target_combo.findData(target)
        if i >= 0 and i != self.target_combo.currentIndex():
            self.target_combo.blockSignals(True)
            self.target_combo.setCurrentIndex(i)
            self.target_combo.blockSignals(False)
        noun = self.noun()
        self.mode_people.setText(f"{noun}별")
        self.search_btn.setText(f"이 {noun}으로 검색" if target == "object" else "이 사람으로 검색")
        # 다른 대상의 화면이 남지 않게 비운다
        self.index, self.run, self.names, self.inherited, self.labels = {}, None, {}, {}, {}
        self.grid.clear()
        self.file_list.clear()
        self.detail_list.clear()
        self.detail_title.setText("선택하면 상세가 나옵니다")
        self.detail_sub.setText("")
        self.name_btn.setEnabled(False)
        self.search_btn.setEnabled(False)
        self.label_btn.setEnabled(False)
        self.refresh_runs()

    def refresh_runs(self) -> None:
        self.runs = PI.find_runs(self.root, self.target)
        self.run_combo.clear()
        for r in self.runs:
            self.run_combo.addItem(PI.run_title(r), r)
        last = PI.load_last_run(self.root, self.target)
        for i, r in enumerate(self.runs):
            if last and str(r["assignments"]) == last:
                self.run_combo.setCurrentIndex(i)
                break
        if not self.runs:
            where = "사진 처리 · 영상 처리 3. 클러스터" if self.target == "person" else "사진 처리 4. 클러스터 · 영상 처리 4. Object 클러스터"
            self._status(f"{self.noun()} 묶음 결과가 없습니다 — {where} 단계를 먼저 실행하세요")
        else:
            self._status("묶음 결과를 고르고 '불러오기' 를 누르세요 (3. 클러스터 단계의 출력)")

    def current_run(self) -> Optional[Dict[str, Any]]:
        return self.run_combo.currentData() if self.run_combo.count() else None

    # ---- 불러오기 ----
    def _prepare(self, run: Dict[str, Any]) -> None:
        self.run = run
        self.names = PI.load_names(run)
        self.labels = PI.load_labels(run)
        self.inherited = {}

    def load(self, rebuild: bool = False) -> None:
        run = self.current_run()
        if run is None:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self._prepare(run)
        self.load_btn.setEnabled(False)
        self.rebuild_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self._status("불러오는 중… (처음은 DB 페이로드를 읽어 캐시합니다)")
        self.worker = _LoadWorker(run, rebuild, self._fetch, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_loaded)
        self.worker.failed.connect(self._on_failed)
        # 불러오기가 끝나면 자동 라벨이 곧바로 시작될 수 있다 — 그때는 진행 표시를 끄지 않는다
        self.worker.finished.connect(lambda: (self.load_btn.setEnabled(True), self.rebuild_btn.setEnabled(True),
                                              self.progress.setVisible(self._label_running())))
        self.worker.start()

    def load_sync(self, rebuild: bool = False) -> None:
        """테스트/스크립트용: 워커 없이 바로."""
        run = self.current_run()
        if run is None:
            return
        self._prepare(run)
        self._on_loaded(PI.load_or_build(run, fetch=self._fetch, rebuild=rebuild))

    def _on_progress(self, a: int, b: int) -> None:
        self.progress.setRange(0, max(1, b))
        self.progress.setValue(a)
        self._status(f"DB 페이로드 읽는 중 {a:,} / {b:,}")

    def _on_failed(self, msg: str) -> None:
        self._status("불러오기 실패: " + msg)
        QMessageBox.warning(self, "불러오기 실패", msg)

    def _on_loaded(self, index: Dict[str, Any]) -> None:
        self.index = index
        if self.run is not None:
            PI.save_last_run(self.run, self.root)
            try:
                self.inherited = PI.inherited_names(self.run, index.get("clusters", {}), self.root)
            except Exception:  # noqa: BLE001 — 이어받기는 부가 기능이라 실패해도 화면은 연다
                self.inherited = {}
        folders: Dict[str, int] = {}
        for f in index.get("files", {}).values():
            folders[f["folder"]] = folders.get(f["folder"], 0) + 1
        self.folder_combo.blockSignals(True)
        self.folder_combo.clear()
        self.folder_combo.addItem("전체", "")
        for name, n in sorted(folders.items(), key=lambda kv: -kv[1]):
            self.folder_combo.addItem(f"{name} ({n:,}개 파일)", name)
        self.folder_combo.blockSignals(False)
        self.label_btn.setEnabled(self.run is not None and not self._label_running())
        self._render()
        self._maybe_auto_label()

    # ---- 표시 ----
    def display(self, cid: str) -> str:
        return PI.display_name(cid, self.names, self.labels, self.inherited)

    def _set_mode(self, mode: str) -> None:
        self.view_mode = mode
        self.left.setCurrentIndex(0 if mode == "people" else 1)
        # 보기가 바뀌면 오른쪽 상세는 비운다 (이전 보기의 선택이 남아 있지 않게)
        self.detail_list.clear()
        self.detail_title.setText("선택하면 상세가 나옵니다" if mode == "people" else f"파일을 고르면 그 파일에 나온 {self.noun()}이 나옵니다")
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

    def _tooltip(self, cid: str, c: Dict[str, Any]) -> str:
        auto = PI.label_line(self.labels.get(cid))
        inh = self.inherited.get(cid)
        return (f"{cid}\n폴더: " + ", ".join(f"{k} {v}" for k, v in c.get("folders", {}).items())
                + (f"\n자동 라벨: {auto}" if auto else "\n자동 라벨 없음")
                + (f"\n이어받은 이름: {inh['name']} ({inh['from_run']} 에서, 구성원 공유 {inh['share']:.0%})" if inh else "")
                + (f"\n이름: {self.names[cid]}" if self.names.get(cid) else ""))

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
                item = QListWidgetItem(QIcon(pix) if pix else QIcon(), f"{self.display(cid)}\n{c['size']:,}장 · 파일 {c['n_files']:,}")
                item.setData(Qt.UserRole, cid)
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
                item.setSizeHint(self.grid.gridSize())
                item.setToolTip(self._tooltip(cid, c))
                self.grid.addItem(item)
            n_auto = self.labeled_count(ids)
            n_inh = sum(1 for cid in ids if cid in self.inherited and not self.names.get(cid))
            unit = NOUNS.get(self.target, NOUNS["person"])[1]
            self._status(f"{self.count_word(len(ids))} · 사진/track {self.index.get('n_points', 0):,}개 · 미분류 {n_noise:,}"
                         + (f" · 자동 라벨 {n_auto:,}{unit}" if n_auto else " · 자동 라벨 없음 → '자동 라벨 붙이기'")
                         + (f" · 이어받은 이름 {n_inh:,}{unit}" if n_inh else "")
                         + (f" · 폴더 {self._folder()}" if self._folder() else ""))
        else:
            self.file_list.clear()
            files = self.visible_files()
            fl = self.index["files"]
            for f in files:
                d = fl[f]
                people = [cid for cid in d["clusters"] if cid != PI.NOISE]
                people.sort(key=lambda cid: -len(d["clusters"][cid]))
                names = ", ".join(self.display(cid) for cid in people[:6]) + (" …" if len(people) > 6 else "")
                extra = f" · 미분류 {len(d['clusters'][PI.NOISE])}" if PI.NOISE in d["clusters"] else ""
                item = QListWidgetItem(f"{Path(f).name}    {self.count_word(len(people))}: {names or '-'}{extra}")
                item.setData(Qt.UserRole, f)
                item.setToolTip(f)
                self.file_list.addItem(item)
            self._status(f"파일 {len(files):,}개" + (f" · 폴더 {self._folder()}" if self._folder() else ""))

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
        self.detail_title.setText(f"{self.display(cid)}  ·  {c['size']:,}장 · 파일 {c['n_files']:,}개")
        auto = PI.label_line(self.labels.get(cid))
        inh = self.inherited.get(cid)
        self.detail_sub.setText(f"{PI.short_id(cid)} · " + (f"자동 라벨: {auto}" if auto else "자동 라벨 없음")
                                + (f" · 이어받은 이름: {inh['name']} ({inh['from_run']})" if inh else "")
                                + ". 나온 파일 (파일마다 crop 하나씩). 두 번 클릭하면 이름을 붙일 수 있습니다.")
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
        self.detail_title.setText(f"{Path(fname).name}  ·  {self.count_word(n_people)}")
        self.detail_sub.setText(fname)
        for cid, pids in sorted(d["clusters"].items(), key=lambda kv: (kv[0] == PI.NOISE, -len(kv[1]))):
            for pid in pids[:50]:
                pt = pts.get(pid, {})
                pix = square_thumb(PI.resolve_crop(pt.get("crop", ""), self.root), CROP)
                it = QListWidgetItem(QIcon(pix) if pix else QIcon(), f"{self.display(cid)}\n{pt.get('score', 0.0):.2f}")
                it.setData(Qt.UserRole, pid)
                it.setToolTip(cid)
                it.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
                self.detail_list.addItem(it)

    # ---- 동작 ----
    def _rename(self, item: Optional[QListWidgetItem]) -> None:
        if item is None or self.run is None or self.view_mode != "people":
            return
        cid = str(item.data(Qt.UserRole))
        # 이름이 없으면 이어받은 이름 → 자동 라벨 순으로 미리 채워 두어 그대로 확정하거나 고칠 수 있게 한다
        cur = (self.names.get(cid, "") or str(self.inherited.get(cid, {}).get("name") or "")
               or str(self.labels.get(cid, {}).get("name") or ""))
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

    def search_request(self) -> Optional[tuple]:
        """'이 사람으로 검색' 이 보낼 (crop, media, target). 대표 crop 이 없으면 None."""
        item = self.grid.currentItem()
        if item is None or not self.index:
            return None
        c = self.index["clusters"].get(str(item.data(Qt.UserRole)), {})
        rep = self.index["points"].get(c.get("rep") or "", {})
        crop = PI.resolve_crop(rep.get("crop", ""), self.root)
        if crop is None:
            return None
        media = "video" if rep.get("folder") == "videos" else "image"
        return str(crop), media, self.target

    def _search_person(self) -> None:
        req = self.search_request()
        if req is not None:
            self.searchRequested.emit(*req)

    # ---- 자동 라벨 붙이기 ----
    def labeled_count(self, cluster_ids: Optional[List[str]] = None) -> int:
        ids = self.visible_clusters() if cluster_ids is None else cluster_ids
        return sum(1 for cid in ids if self.labels.get(cid, {}).get("name"))

    def _label_running(self) -> bool:
        return self.label_worker is not None and self.label_worker.isRunning()

    def label_tool_key(self) -> str:
        return str(self.label_tool.currentData() or "vec")

    def auto_label_command(self) -> List[str]:
        """현재 실행에 라벨러를 돌리는 명령 (people_index.auto_label_command)."""
        if self.run is None:
            raise RuntimeError("먼저 묶음 결과를 불러오세요")
        return PI.auto_label_command(self.run, self.label_tool_key(), root=self.root)

    def auto_label(self) -> None:
        """버튼: 돌고 있으면 중단, 아니면 고른 라벨러를 subprocess 로 시작한다."""
        if self._label_running():
            self.label_worker.stop()
            self._status("자동 라벨 중단 요청…")
            return
        if self.run is None:
            return
        self._start_label_worker(self.auto_label_command())

    def _maybe_auto_label(self) -> bool:
        """불러온 실행에 라벨 파일이 하나도 없으면(사진·영상, 사람·물건 무관) 색상 라벨러를 저절로 시작한다. 세션당 실행마다 한 번."""
        if not self.auto_check.isChecked() or self.run is None or self._label_running():
            return False
        key = str(self.run["assignments"])
        if key in self._auto_tried or PI.find_labels(self.run):
            return False
        self._auto_tried.add(key)
        self._start_label_worker(PI.auto_label_command(self.run, "vec", root=self.root), auto=True)
        return True

    @staticmethod
    def _tool_of(cmd: List[str]) -> str:
        script = " ".join(cmd[:4])
        return "qwen" if "label_clusters_qwen" in script else "vec" if "label_clusters_from_vectors" in script else "custom"

    def _start_label_worker(self, cmd: List[str], auto: bool = False) -> None:
        from gui.pipeline_page import ProcessWorker

        self._label_log = []
        self._label_run = self.run
        self._label_auto = auto
        # 로그를 결과 폴더에 남긴다 — GUI 메모리에만 두면 창이 닫히거나 도중에 죽었을 때 원인을 알 수 없다
        self._close_label_log()
        self.label_log_path = None
        if self.run is not None:
            try:
                self.label_log_path = Path(self.run["folder"]) / f"auto_label_{self._tool_of(cmd)}.log"
                self._label_log_file = self.label_log_path.open("w", encoding="utf-8")
                self._label_log_file.write("$ " + " ".join(cmd) + "\n")
                self._label_log_file.flush()
            except OSError:
                self._label_log_file, self.label_log_path = None, None
        self.label_btn.setText("중단")
        self.label_btn.setEnabled(True)
        self.label_tool.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self._status("라벨 파일이 없어 색상 라벨을 자동으로 만드는 중… (30초~3분, 끝나면 카드가 바뀝니다)" if auto else
                     "자동 라벨 실행 중… (색상: 30초~3분 · 문장: 군집당 약 0.6초 + 시작 1~2분)")
        self.label_worker = ProcessWorker(cmd, self.root, self)
        self.label_worker.line.connect(self._on_label_line)
        self.label_worker.finished_with.connect(self._on_label_finished)
        self.label_worker.start()

    def _close_label_log(self) -> None:
        if self._label_log_file is not None:
            try:
                self._label_log_file.close()
            except OSError:
                pass
            self._label_log_file = None

    def _on_label_line(self, line: str) -> None:
        self._label_log.append(line)
        if len(self._label_log) > 400:
            del self._label_log[:200]
        if self._label_log_file is not None:
            try:
                self._label_log_file.write(line + "\n")
                self._label_log_file.flush()
            except OSError:
                pass
        text = line.strip()
        if text and not text.startswith("=") and not text.startswith("RESULT_"):
            self._status("자동 라벨 실행 중… " + text[:100])

    @staticmethod
    def _same_run(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
        return a is not None and b is not None and str(a.get("assignments")) == str(b.get("assignments"))

    def _on_label_finished(self, code: int) -> None:
        if self._label_log_file is not None:
            try:
                self._label_log_file.write(f"[exit {code}]\n")
            except OSError:
                pass
        self._close_label_log()
        self.label_btn.setText("자동 라벨 붙이기")
        self.label_btn.setEnabled(self.run is not None)
        self.label_tool.setEnabled(True)
        self.progress.setVisible(False)
        done_run, auto = self._label_run, self._label_auto
        self._label_run, self._label_auto = None, False
        log_note = f" · 로그: {self.label_log_path}" if self.label_log_path else ""
        if not self._same_run(done_run, self.run):
            # 라벨러가 도는 사이 다른 실행을 불러왔다 — 지금 화면은 건드리지 않고 알린 뒤, 지금 실행에도 필요하면 시작한다
            name = done_run["run"] if done_run else "?"
            self._status(f"{name} 자동 라벨 " + ("완료" if code == 0 else f"실패 (종료 코드 {code})")
                         + " — 그 결과를 다시 불러오면 보입니다" + ("" if code == 0 else log_note))
            self._maybe_auto_label()
            return
        self.labels = PI.load_labels(self.run)
        cur = self.grid.currentRow()
        self._render()
        if 0 <= cur < self.grid.count():
            self.grid.setCurrentRow(cur)
        unit = NOUNS.get(self.target, NOUNS["person"])[1]
        if code == 0:
            self._status(f"자동 라벨 완료 — {self.noun()} {self.labeled_count():,}{unit}에 라벨이 붙었습니다 ({self.label_tool.currentText()}). "
                         "이름을 붙인 카드는 이름이 그대로 우선합니다.")
        else:
            tail = "\n".join(self._label_log[-12:])
            self._status(f"자동 라벨 실패 (종료 코드 {code}) — 마지막 로그: {(self._label_log or [''])[-1][:80]}{log_note}")
            if not auto:          # 저절로 시작한 실행은 대화상자로 막지 않는다 (버튼으로 다시 돌리면 로그를 보여 준다)
                QMessageBox.warning(self, "자동 라벨 실패", f"종료 코드 {code}\n\n{tail}" + (f"\n\n전체 로그: {self.label_log_path}" if self.label_log_path else ""))
