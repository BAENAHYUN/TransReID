"""gui/bench_page.py — 벤치마크 탭 (P5): 원장(bench/ledger.jsonl) 리더보드 + 채택 기준 색 + 상세 + verify 실행 + 채택→yaml + 그래프.

- 단계(detect/embed/search/cluster/e2e)·이름 필터·이름별 최근만·통과만 으로 원장 행을 고른다. 열은 bench.ledger.METRIC_KEYS.
- 행 색: bench.criteria.ADOPTION_RULES 대조 — 초록(전부 통과) / 노랑(일부) / 빨강(미달) / 없음(기준 없음).
- 선택한 행: 상세(JSON), 재현 명령(show-cmd), verify 실행(bench/run.py verify, 로그 탭), 채택 → yaml (bench.criteria.adopt_yaml → 드롭다운 자동 등장),
  결과 폴더 열기. 그래프 탭: 단계의 (제약 지표, 목적 지표) 산점도 + 검출 행이면 PR 곡선 (matplotlib 있을 때).
새 실행은 평가 탭 5(러너)·9(탐색)·10(조합) 에서 돌리고, 여기서 새로고침한다.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import criteria, ledger  # noqa: E402

PY = sys.executable
STAGES = list(ledger.STAGES)
STAGE_LABEL = {"detect": "검출", "embed": "임베딩(단독)", "search": "검색 조합(단독)", "cluster": "클러스터링", "e2e": "전체 파이프라인",
               "track": "추적·스티칭", "object": "객체 재출현", "qwen": "Qwen 후처리"}
STATUS_COLOR = {"pass": QColor("#e6f4ea"), "partial": QColor("#fff8e1"), "fail": QColor("#fdecea")}
STATUS_MARK = {"pass": "✓", "partial": "△", "fail": "✗", "n/a": "—", "incomplete": "?"}


class NumItem(QTableWidgetItem):
    """숫자 정렬용 셀 (표시는 ledger.fmt, 정렬은 실제 값)."""

    def __init__(self, value: Any):
        super().__init__(ledger.fmt(value))
        self.setData(Qt.ItemDataRole.UserRole, value if isinstance(value, (int, float)) and not isinstance(value, bool) else None)
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other):  # type: ignore[override]
        a, b = self.data(Qt.ItemDataRole.UserRole), other.data(Qt.ItemDataRole.UserRole)
        if a is None or b is None:
            return (a is None) and (b is not None)
        return a < b


class BenchPage(QWidget):
    def __init__(self, ledger_path: Optional[Path] = None, adopt_root: Optional[Path] = None, charts: bool = True,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ledger_path = Path(ledger_path) if ledger_path else ledger.DEFAULT_LEDGER
        self.adopt_root = Path(adopt_root) if adopt_root else ROOT
        self.entries: List[Dict[str, Any]] = []
        self.rows: List[Dict[str, Any]] = []
        self.worker = None
        self._charts = charts
        self._build()
        self.refresh()

    # ---------------------------------------------------------------- UI
    def _build(self) -> None:
        root = QVBoxLayout(self)
        intro = QLabel("원장(bench/ledger.jsonl)의 모든 평가 실행을 한 표로 비교합니다. 행 색 = 채택 기준(기준표) 통과 여부. "
                       "새 실행은 평가 탭 5(러너)·9(탐색)·10(조합)에서, 여기서는 verify·재현 명령·채택→yaml.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#5a6673;")
        root.addWidget(intro)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("단계"))
        self.stage_combo = QComboBox()
        for s in STAGES:
            self.stage_combo.addItem(f"{s} · {STAGE_LABEL[s]}", s)
        self.stage_combo.currentIndexChanged.connect(self.refresh)
        bar.addWidget(self.stage_combo)
        bar.addWidget(QLabel("이름 필터"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("부분 문자열 (예: yolo, combo:, study:)")
        self.name_edit.textChanged.connect(self.refresh)
        bar.addWidget(self.name_edit, 1)
        self.latest_check = QCheckBox("이름별 최근만")
        self.latest_check.setChecked(True)
        self.latest_check.toggled.connect(self.refresh)
        bar.addWidget(self.latest_check)
        self.pass_check = QCheckBox("통과만")
        self.pass_check.toggled.connect(self.refresh)
        bar.addWidget(self.pass_check)
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.refresh)
        bar.addWidget(self.refresh_btn)
        self.count_label = QLabel("")
        bar.addWidget(self.count_label)
        root.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.verticalHeader().setVisible(False)
        split.addWidget(self.table)

        self.right = QTabWidget()
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setFont(QFont("Consolas", 9))
        self.right.addTab(self.detail, "상세")
        self.chart_holder = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_holder)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = None
        self.right.addTab(self.chart_holder, "그래프")
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 9))
        self.console.setStyleSheet("background:#0f1419; color:#d6deeb;")
        self.right.addTab(self.console, "실행 로그")
        split.addWidget(self.right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        btns = QHBoxLayout()
        self.verify_btn = QPushButton("verify 실행 (재현 검증)")
        self.verify_btn.clicked.connect(self._verify)
        self.cmd_btn = QPushButton("재현 명령")
        self.cmd_btn.clicked.connect(self._show_cmd)
        self.adopt_btn = QPushButton("채택 → yaml")
        self.adopt_btn.clicked.connect(self._adopt)
        self.overwrite_check = QCheckBox("같은 이름 덮어쓰기")
        self.open_btn = QPushButton("결과 폴더 열기")
        self.open_btn.clicked.connect(self._open_report)
        self.stop_btn = QPushButton("중단")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        for b in (self.verify_btn, self.cmd_btn, self.adopt_btn, self.overwrite_check, self.open_btn, self.stop_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        root.addLayout(btns)
        self._set_selection_enabled(False)

    def _set_selection_enabled(self, on: bool) -> None:
        for b in (self.verify_btn, self.cmd_btn, self.adopt_btn, self.open_btn):
            b.setEnabled(on)

    # ---------------------------------------------------------------- 데이터
    def current_stage(self) -> str:
        return str(self.stage_combo.currentData() or STAGES[0])

    def filtered(self) -> List[Dict[str, Any]]:
        stage = self.current_stage()
        rows = ledger.filter_entries(self.entries, stage=stage, name=self.name_edit.text().strip() or None)
        if self.latest_check.isChecked():
            rows = list(ledger.latest_by_name(rows).values())
        for r in rows:
            r["_adopt"] = criteria.evaluate(r)
        if self.pass_check.isChecked():
            rows = [r for r in rows if r["_adopt"]["status"] == "pass"]
        return rows

    def refresh(self) -> None:
        errors: List[str] = []
        self.entries = ledger.read_entries(self.ledger_path, errors)
        self.rows = self.filtered()
        stage = self.current_stage()
        keys = ledger.METRIC_KEYS.get(stage, [])
        headers = ["채택", "name", "created", *keys, "verify", "run_id"]
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self.rows))
        for i, e in enumerate(self.rows):
            st = e["_adopt"]["status"]
            cells: List[QTableWidgetItem] = [QTableWidgetItem(f"{STATUS_MARK[st]} {e['_adopt']['passed']}/{e['_adopt']['applicable']}"),
                                             QTableWidgetItem(str(e.get("name", ""))), QTableWidgetItem(str(e.get("created_at", ""))[:16])]
            m = e.get("metrics") or {}
            cells += [NumItem(m.get(k)) for k in keys]
            v = e.get("verify") or {}
            cells.append(QTableWidgetItem(v.get("status", "PASS" if v.get("passed") else "FAIL") if v else ""))
            cells.append(QTableWidgetItem(str(e.get("run_id", ""))))
            color = STATUS_COLOR.get(st)
            for j, c in enumerate(cells):
                if color is not None:
                    c.setBackground(color)
                c.setData(Qt.ItemDataRole.UserRole + 1, i)          # 정렬 후에도 원래 행 인덱스를 찾기 위해
                self.table.setItem(i, j, c)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        n_pass = sum(1 for r in self.rows if r["_adopt"]["status"] == "pass")
        self.count_label.setText(f"{len(self.rows)} 행 · 통과 {n_pass}" + (f" · 손상 줄 {len(errors)}" if errors else "")
                                 + ("" if self.ledger_path.is_file() else " · 원장 없음"))
        self._set_selection_enabled(False)
        self.detail.setPlainText("행을 선택하면 상세가 나옵니다.")
        self._draw_chart(None)

    def selected_entry(self) -> Optional[Dict[str, Any]]:
        items = self.table.selectedItems()
        if not items:
            return None
        idx = items[0].data(Qt.ItemDataRole.UserRole + 1)
        return self.rows[idx] if isinstance(idx, int) and 0 <= idx < len(self.rows) else None

    def _on_select(self) -> None:
        e = self.selected_entry()
        self._set_selection_enabled(e is not None)
        if e is None:
            return
        ad = e["_adopt"]
        lines = [f"{criteria.STATUS_LABEL[ad['status']]}  ({ad['passed']}/{ad['applicable']})"]
        for c in ad["checks"]:
            lines.append(f"  {'ok ' if c['ok'] else ('NG ' if c['ok'] is not None else '-- ')}{c['label']}: {ledger.fmt(c['value'])}")
        show = {k: e.get(k) for k in ("run_id", "stage", "name", "created_at", "producer", "component", "params", "gt", "metrics", "timing",
                                      "versions", "env", "hardware", "weights", "config", "report", "command", "note", "verify", "bench", "study", "combo")
                if e.get(k) not in (None, {}, [])}
        lines.append("")
        lines.append(json.dumps(show, ensure_ascii=False, indent=1, default=str))
        self.detail.setPlainText("\n".join(lines))
        self._draw_chart(e)

    # ---------------------------------------------------------------- 그래프
    def _draw_chart(self, selected: Optional[Dict[str, Any]]) -> None:
        if not self._charts:
            return
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as exc:  # noqa: BLE001
            if self.canvas is None:
                lbl = QLabel(f"matplotlib 을 불러오지 못해 그래프를 그릴 수 없습니다: {exc}")
                lbl.setWordWrap(True)
                self.chart_layout.addWidget(lbl)
                self.canvas = lbl
            return
        if self.canvas is None or not isinstance(self.canvas, FigureCanvasQTAgg):
            self.canvas = FigureCanvasQTAgg(Figure(figsize=(5, 6), dpi=96))
            self.chart_layout.addWidget(self.canvas)
        fig = self.canvas.figure
        fig.clear()
        stage = self.current_stage()
        xk, yk = criteria.CHART_AXES.get(stage, ("", ""))
        curve = _pr_curve_for(selected) if selected and selected.get("stage") == "detect" else None
        ax = fig.add_subplot(2 if curve else 1, 1, 1)
        pts = [(r["metrics"].get(xk), r["metrics"].get(yk), r) for r in self.rows
               if isinstance(r["metrics"].get(xk), (int, float)) and isinstance(r["metrics"].get(yk), (int, float))]
        colors = {"pass": "#2e7d32", "partial": "#f9a825", "fail": "#c62828", "n/a": "#9aa5b1"}
        for x, y, r in pts:
            ax.scatter([x], [y], s=60 if r is selected else 28, c=colors[r["_adopt"]["status"]], edgecolors="#1f2933" if r is selected else "none", zorder=3)
            if r is selected or len(pts) <= 12:
                ax.annotate(str(r.get("name", ""))[:22], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        for metric, op, target, _ in criteria.ADOPTION_RULES.get(stage, []):
            if metric == xk:
                ax.axvline(target, color="#c3cad2", linestyle="--", linewidth=1)
            if metric == yk:
                ax.axhline(target, color="#c3cad2", linestyle="--", linewidth=1)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title(f"{stage}: {yk} vs {xk} (점선 = 채택 기준)", fontsize=9)
        ax.grid(True, alpha=0.3)
        if curve:
            ax2 = fig.add_subplot(2, 1, 2)
            rec, prec = zip(*curve)
            ax2.plot(rec, prec, color="#2a78d6")
            ax2.set_xlabel("recall")
            ax2.set_ylabel("precision")
            ax2.set_title(f"PR 곡선: {selected.get('name')} (AP@0.5 {ledger.fmt(selected['metrics'].get('ap50'))})", fontsize=9)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1.02)
            ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- 동작
    def _append(self, text: str) -> None:
        self.console.appendPlainText(text)

    def _verify(self) -> None:
        e = self.selected_entry()
        if e is None:
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "실행 중", "이미 verify 가 실행 중입니다.")
            return
        from gui.pipeline_page import ProcessWorker
        cmd = [PY, "bench/run.py", "verify", str(e["run_id"]), "--ledger", str(self.ledger_path)]
        self.right.setCurrentWidget(self.console)
        self._append("=" * 60)
        self._append(" ".join(cmd))
        self.worker = ProcessWorker(cmd, ROOT, self)
        self.worker.line.connect(self._append)
        self.worker.finished_with.connect(self._verify_done)
        self.verify_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def _verify_done(self, code: int) -> None:
        self._append(f"[종료 코드 {code}] {'PASS' if code == 0 else 'FAIL/UNVERIFIED — 로그 확인'}")
        self.stop_btn.setEnabled(False)
        self.refresh()

    def _stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._append("[중단 요청]")
            self.worker.stop()

    def _show_cmd(self) -> None:
        e = self.selected_entry()
        if e is None:
            return
        try:
            cmd = criteria.reproduce_command(e)
        except Exception as exc:  # noqa: BLE001
            cmd = f"재현 명령을 만들 수 없음: {exc}"
        self.detail.setPlainText(cmd + "\n\n(클립보드에 복사됨)\n\n" + self.detail.toPlainText())
        try:
            QGuiApplication.clipboard().setText(cmd)
        except Exception:  # noqa: BLE001
            pass

    def _adopt(self) -> None:
        e = self.selected_entry()
        if e is None:
            return
        try:
            res = criteria.adopt_yaml(e, root=self.adopt_root, overwrite=self.overwrite_check.isChecked(),
                                      tracking_template=ROOT / "pipeline_tracking.yaml", pipeline_path=ROOT / "pipeline.yaml")
        except FileExistsError as exc:
            QMessageBox.warning(self, "채택", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "채택 실패", f"{type(exc).__name__}: {exc}")
            return
        files = "\n".join(str(p) for p in res["files"]) or "(만든 파일 없음)"
        self._append(f"[채택] {e['run_id']} → {files}\n{res['note']}")
        QMessageBox.information(self, "채택 → yaml", f"{files}\n\n{res['note']}")

    def _open_report(self) -> None:
        e = self.selected_entry()
        if e is None:
            return
        target = e.get("report") or (e.get("bench") or {}).get("run_dir")
        if not target:
            QMessageBox.information(self, "결과", "이 행에는 결과 경로가 없습니다.")
            return
        p = Path(str(target))
        folder = p.parent if p.suffix else p
        if not folder.exists():
            QMessageBox.information(self, "결과", f"경로가 없습니다: {folder}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))


def _pr_curve_for(entry: Dict[str, Any]) -> Optional[List[List[float]]]:
    """검출 행의 사이드카(detect_eval_report.json) 에서 그 method 의 PR 곡선 [[recall, precision], …]."""
    report = entry.get("report")
    if not report:
        return None
    try:
        d = json.loads(Path(str(report)).read_text(encoding="utf-8"))
        res = (d.get("results") or {}).get(str(entry.get("name"))) or {}
        curve = res.get("curve")
        return curve if isinstance(curve, list) and curve else None
    except (OSError, ValueError):
        return None
