"""
pipeline_page.py — DB 구축 / 클러스터링 / 평가 단계를 GUI 에서 실행한다.

search_gui.py 의 MainWindow 가 이 모듈의 PipelinePage 를 탭으로 붙인다.
검색 쪽 코드는 건드리지 않는다.

설계
----
단계 정의는 이 파일이 아니라 gui_pipelines.json 에 있다. 단계를 추가하거나
빼거나 다른 스크립트로 바꾸려면 그 JSON 만 고치면 되고, 이 파일은 그대로 둔다.
폼 위젯은 arg 의 type 을 보고 런타임에 만들어진다.

실행 방식
--------
모든 단계는 subprocess 로 돈다. GUI 프로세스 안에서 import 하지 않는다.
  * 임베더/모델이 GUI 메모리에 눌러앉지 않는다
  * 단계가 죽어도 GUI 는 살아있다
  * 중단 버튼으로 프로세스 트리를 죽일 수 있다
stdout 은 줄 단위로 읽어 로그 창에 흘린다 (-u 로 버퍼링 해제).

안전장치
-------
DESTRUCTIVE_FLAGS 에 있는 인자는 JSON 에 적혀 있어도 실행 직전에 걸러내고
경고를 남긴다. GUI 로 DB 를 날릴 수 있는 경로를 만들지 않는다.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import signal
from html import escape as html_escape
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "gui_pipelines.json"

# 단계 스크립트가 성공 시 stdout 마지막에 찍는 산출물 마커. GUI 는 마지막 마커의 경로를
# 기억해 "결과 열기" 버튼을 켠다. 경로에 공백/한글이 있어도 줄 끝까지 통째로 받는다.
#   RESULT_HTML: C:/.../outputs/image_review/index_PRW.html
RESULT_MARKER_RE = re.compile(r"^\s*RESULT_HTML:\s*(?P<path>.+?)\s*$")
# 산출물 옆 sidecar JSON (warnings 배열) 마커. 있으면 "완료 · 경고 N" 으로 판정을 세분화한다.
#   RESULT_SUMMARY: C:/.../index_PRW.summary.json
SUMMARY_MARKER_RE = re.compile(r"^\s*RESULT_SUMMARY:\s*(?P<path>.+?)\s*$")

# GUI 에서 절대 실행하지 않는 인자. JSON 에 적어도 무시된다.
DESTRUCTIVE_FLAGS = {
    # 컬렉션/체크포인트를 지우고 다시 만드는 인자
    "--recreate",
    "--recreate-person",
    "--recreate-object",
    "--recreate-collection",
    "--fresh",
    "--reset",
    "--reset-checkpoint",
    # 기존 산출물을 지우거나 덮어쓰는 인자
    "--force",
    "--overwrite",
    "--clean",
    "--clear",
    # 삭제 계열
    "--drop",
    "--drop-collection",
    "--delete",
    "--delete-all",
    "--purge",
    "--prune",
    "--wipe",
    "--truncate",
    "--remove",
}


# =============================================================================
# 단계 정의 로딩
# =============================================================================

class RegistryError(RuntimeError):
    pass


# 최상위 키 -> 탭 제목. JSON 의 "_groups" 로 덮어쓸 수 있다.
BUILTIN_GROUP_TITLES = {
    "video_pipeline": "영상 파이프라인",
    "image_pipeline": "이미지 파이프라인",
    "evaluation": "평가 / 비교",
}

VALID_ARG_TYPES = {"str", "int", "float", "bool", "choice", "path", "dir", "list"}


def _prettify(key: str) -> str:
    return key.replace("_", " ").strip().title()


def _check_stages(where: str, group_id: str, stages: Any) -> None:
    if not isinstance(stages, list) or not stages:
        raise RegistryError(f"{where}: '{group_id}' 의 단계 목록이 비어 있습니다.")
    seen = set()
    for st in stages:
        if not isinstance(st, dict):
            raise RegistryError(f"{where}: '{group_id}' 의 단계가 객체가 아닙니다.")
        for key in ("id", "title", "script"):
            if key not in st:
                raise RegistryError(
                    f"{where}: 단계에 '{key}' 가 없습니다 (group={group_id})"
                )
        if st["id"] in seen:
            raise RegistryError(
                f"{where}: 단계 id 가 중복입니다: '{st['id']}' (group={group_id})"
            )
        seen.add(st["id"])
        for a in st.get("args", []):
            t = a.get("type", "str")
            if t not in VALID_ARG_TYPES:
                raise RegistryError(
                    f"{where}: 알 수 없는 arg type '{t}' "
                    f"(stage={st['id']}, label={a.get('label')})\n"
                    f"  허용={sorted(VALID_ARG_TYPES)}"
                )
            if t == "choice" and not (a.get("choices") or a.get("choices_glob")):
                raise RegistryError(
                    f"{where}: type=choice 인데 choices / choices_glob 가 없습니다 "
                    f"(stage={st['id']}, label={a.get('label')})"
                )
    ids = {str(st["id"]) for st in stages}
    for st in stages:
        for t in st.get("tools") or []:
            if not isinstance(t, dict) or str(t.get("stage") or "") not in ids:
                raise RegistryError(
                    f"{where}: 단계 '{st['id']}' 의 tools 항목은 같은 그룹의 단계 id 를 가리켜야 합니다: {t}"
                )


def load_registry(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    gui_pipelines.json 을 읽어 내부 표현으로 정규화한다.

    받는 형식 두 가지:

      1) 평평한 형식 (권장)
         { "video_pipeline": [ {...}, ... ], "image_pipeline": [ ... ] }
         최상위 키 하나가 탭 하나. '_' 로 시작하는 키는 메타데이터라 무시한다.
         탭 제목은 "_groups" -> BUILTIN_GROUP_TITLES -> 키 이름 순으로 정한다.

      2) 예전 형식
         { "groups": [ {"id":..., "title":..., "stages":[...]} ] }

    반환: [{"id", "title", "description", "stages"}]
    """
    p = Path(path or REGISTRY_PATH)
    if not p.is_file():
        raise RegistryError(
            f"단계 정의 파일이 없습니다: {p}\n"
            f"gui_pipelines.json 을 프로젝트 루트에 두세요."
        )

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RegistryError(f"{p.name} JSON 문법 오류: {e}") from e

    if not isinstance(data, dict):
        raise RegistryError(f"{p.name}: 최상위가 객체여야 합니다.")

    meta = data.get("_groups") or {}
    out: List[Dict[str, Any]] = []

    # ---- 예전 형식 ----
    if isinstance(data.get("groups"), list):
        for g in data["groups"]:
            gid = g.get("id")
            if not gid:
                raise RegistryError(f"{p.name}: group 에 'id' 가 없습니다.")
            _check_stages(p.name, gid, g.get("stages"))
            out.append({
                "id": gid,
                "title": g.get("title") or _prettify(gid),
                "description": g.get("description", ""),
                "stages": g["stages"],
            })
        if not out:
            raise RegistryError(f"{p.name}: groups 가 비어 있습니다.")
        return out

    # ---- 평평한 형식 ----
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, list):
            # version 같은 스칼라는 조용히 건너뛴다.
            continue
        _check_stages(p.name, key, value)
        info = meta.get(key) or {}
        out.append({
            "id": key,
            "title": info.get("title") or BUILTIN_GROUP_TITLES.get(key) or _prettify(key),
            "description": info.get("description", ""),
            "stages": value,
        })

    if not out:
        raise RegistryError(
            f"{p.name}: 단계 그룹을 하나도 찾지 못했습니다.\n"
            f"  최상위에 \"video_pipeline\": [ ... ] 형태의 키가 필요합니다."
        )
    return out


# =============================================================================
# 프로세스 실행 워커
# =============================================================================

class ProcessWorker(QThread):
    """스크립트를 subprocess 로 돌리고 stdout 을 줄 단위로 흘린다."""

    line = Signal(str)
    finished_with = Signal(int)

    def __init__(self, cmd: List[str], cwd: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.cmd = cmd
        self.cwd = cwd
        self._proc: Optional[subprocess.Popen] = None
        self._stopped = False

    def run(self) -> None:
        env = os.environ.copy()
        # 워커는 자식 출력을 UTF-8 로 디코드한다. 상속 환경이 다른 인코딩을 지정해도
        # 자식(파이썬)과 디코더가 어긋나지 않게 고정한다 (setdefault 아님).
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        creationflags = 0
        preexec = None
        if os.name == "nt":
            # 중단 시 자식까지 정리할 수 있도록 별도 프로세스 그룹으로 띄운다.
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            preexec = os.setsid

        try:
            self._proc = subprocess.Popen(
                self.cmd,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                preexec_fn=preexec,
            )
        except FileNotFoundError as e:
            self.line.emit(f"[실행 실패] {e}")
            self.finished_with.emit(-1)
            return
        except Exception as e:  # noqa: BLE001
            self.line.emit(f"[실행 실패] {type(e).__name__}: {e}")
            self.finished_with.emit(-1)
            return

        # Popen 직전에 중단 버튼이 눌렸으면 (_stopped 만 세워지고 proc 이 없어 신호를 못 보냄)
        # 여기서 다시 보낸다. 그렇지 않으면 그 요청은 유실되고 자식은 끝까지 돈다.
        if self._stopped:
            self.stop()

        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            self.line.emit(raw.rstrip("\n"))

        code = self._proc.wait()
        # Windows 는 CTRL_BREAK 등으로 죽은 자식에 0xC000013A 처럼 32bit unsigned 코드를 준다.
        # Signal(int) 는 signed 32bit 라 그대로 emit 하면 OverflowError 로 신호가 사라지고
        # 실행/중단 버튼이 잠긴다. signed 로 접어서 보낸다 (0xC000013A -> -1073741510).
        if code is not None and code > 0x7FFFFFFF:
            code -= 0x100000000
        if self._stopped:
            self.line.emit("[중단됨]")
        self.finished_with.emit(code)

    # CTRL_BREAK 뒤 이만큼 기다려도 살아 있으면 프로세스 트리를 강제 종료한다.
    KILL_GRACE_SEC = 3.0

    def stop(self) -> None:
        self._stopped = True
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            proc.terminate()
        # CTRL_BREAK 는 콘솔이 없는 환경(pythonw, 분리 실행, 일부 자동화)에서는 아무 효과가
        # 없고 예외도 나지 않는다. 그러면 자식이 끝까지 돌고 GUI 는 "중단됨" 을 못 낸다.
        # 유예 뒤에도 살아 있으면 트리째 강제 종료한다 (그때 종료 코드는 1/-9 계열).
        timer = threading.Timer(self.KILL_GRACE_SEC, self._force_kill)
        timer.daemon = True
        timer.start()

    def _force_kill(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self.line.emit(f"[중단] {self.KILL_GRACE_SEC:.0f}초 안에 끝나지 않아 강제 종료합니다")
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
                )
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


# =============================================================================
# arg -> 위젯
# =============================================================================

def _yaml_block(path: Path, key: str) -> Optional[Any]:
    """yaml 최상위 key 의 값. 파일이 깨졌거나 키가 없으면 None."""
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    if not isinstance(raw, dict) or key not in raw:
        return None
    return raw[key]


def _block_identity(block: Any, unique_by: Any) -> str:
    """중복 판정 키. unique_by="*" 면 블록 전체, 목록이면 그 하위 키들만."""
    if unique_by == "*" or not isinstance(block, dict):
        return json.dumps(block, sort_keys=True, ensure_ascii=False, default=str)
    keys = [unique_by] if isinstance(unique_by, str) else list(unique_by)
    return json.dumps({k: block.get(k) for k in keys}, sort_keys=True, ensure_ascii=False, default=str)


def _block_label(block: Any, label_key: Optional[str], value: str) -> str:
    """드롭다운에 보여 줄 이름: label_key 값(예 detector.class) 또는 하위 항목 이름들 + 파일명."""
    name = None
    if isinstance(block, dict):
        if label_key and block.get(label_key) not in (None, ""):
            name = str(block[label_key])
        elif all(isinstance(v, dict) for v in block.values()) and block:
            name = ", ".join(str(k) for k in block.keys())     # retrievers: {siglip2: {...}, irra: {...}}
    return f"{name}  ({value})" if name else value


def _glob_dirs(base: Path, pattern: str, exclude: List[str]) -> List[str]:
    """pattern(예 data/*/*) 에 맞는 폴더의 상대경로 목록. 제외 패턴(폴더 이름 또는 상대경로 fnmatch)에 걸린 폴더는
    그 아래를 아예 훑지 않는다 — crops 처럼 파일이 수십만 개인 폴더를 glob 으로 나열하면 GUI 기동이 멈춘다.
    단계마다 os.scandir 로 폴더만 보고, 부모 순서 → 이름(대소문자 무시) 순으로 정렬한다."""
    parts = [x for x in pattern.replace("\\", "/").split("/") if x]
    level: List[Path] = [base]
    for part in parts:
        nxt: List[Path] = []
        for d in level:
            try:
                with os.scandir(d) as it:
                    ents = [e for e in it if e.is_dir(follow_symlinks=False) and fnmatch.fnmatch(e.name, part)]
            except OSError:
                continue
            for e in sorted(ents, key=lambda e: e.name.lower()):
                rel = Path(e.path).relative_to(base).as_posix()
                if any(fnmatch.fnmatch(e.name, x) or fnmatch.fnmatch(rel, x) for x in exclude):
                    continue
                nxt.append(Path(e.path))
        level = nxt
    return [p.relative_to(base).as_posix() for p in level]


def _choice_values(spec: Dict[str, Any], root: Optional[Path] = None) -> List[Tuple[str, str]]:
    """드롭다운 항목 [(값, 표시 이름)].
    choices 에 적힌 값 + choices_glob(프로젝트 루트 기준 glob, 문자열 또는 목록) 으로 찾은 파일의 상대경로.
      choices_yaml_key   : 그 최상위 키를 가진 yaml 만 (예 detector → 검출기 설정 파일만)
      choices_unique_by  : 블록의 하위 키들(예 ["module","class"]) 또는 "*"(블록 전체) 가 같은 파일은 하나만 —
                           같은 검출기를 가리키는 사본 yaml 이 여러 개여도 항목은 검출기 수만큼만 보인다. default 가 우선.
      choices_label_key  : 표시 이름으로 쓸 블록 하위 키 (예 class → 'YOLO26Detector  (pipeline_tracking_yolo26.yaml)')
    파일을 하나 추가하면 GUI 를 고치지 않아도 항목이 생긴다. default 가 목록에 없으면 앞에 넣는다."""
    base = Path(root) if root is not None else ROOT
    default = spec.get("default")
    items: List[Tuple[str, str]] = [(str(c), str(c)) for c in (spec.get("choices") or [])]
    globs = spec.get("choices_glob")
    if globs:
        key = spec.get("choices_yaml_key")
        unique_by = spec.get("choices_unique_by")
        label_key = spec.get("choices_label_key")
        found: List[Tuple[str, Any]] = []
        for pattern in ([globs] if isinstance(globs, str) else list(globs)):
            for p in sorted(base.glob(str(pattern))):
                if not p.is_file():
                    continue
                block = _yaml_block(p, str(key)) if key else None
                if key and block is None:
                    continue
                found.append((p.relative_to(base).as_posix(), block))
        # default 파일이 있으면 맨 앞으로 — 중복 제거 때 default 쪽이 남는다
        found.sort(key=lambda it: 0 if str(default) == it[0] else 1)
        seen: set = set()
        for value, block in found:
            if key and unique_by is not None:
                ident = _block_identity(block, unique_by)
                if ident in seen:
                    continue
                seen.add(ident)
            items.append((value, _block_label(block, label_key, value) if key else value))
    dirs = spec.get("choices_dirs")
    if dirs:
        # 폴더 드롭다운 (프로젝트 루트 기준 glob; choices_exclude 는 폴더 이름/상대경로 fnmatch 패턴)
        excl = [str(x) for x in (spec.get("choices_exclude") or [])]
        for pattern in ([dirs] if isinstance(dirs, str) else list(dirs)):
            for rel in _glob_dirs(base, str(pattern), excl):
                items.append((rel, rel))
    if default not in (None, "") and str(default) not in {v for v, _ in items}:
        items.insert(0, (str(default), str(default)))
    out: List[Tuple[str, str]] = []
    known: set = set()
    for value, label in items:
        if value not in known:
            known.add(value)
            out.append((value, label))
    return out


def resolve_default(spec: Dict[str, Any], root: Optional[Path] = None) -> Any:
    """필드 기본값. default 가 비어 있으면 default_prefer(존재하는 첫 경로) → default_first_choice(드롭다운 첫 항목) 순으로 채운다.
    빈 칸을 보고 사용자가 뭘 넣어야 할지 몰라 멈추지 않게 하려는 것이다."""
    base = Path(root) if root is not None else ROOT
    default = spec.get("default")
    if default not in (None, ""):
        return default
    for cand in (spec.get("default_prefer") or []):
        if (base / str(cand)).exists():
            return str(cand)
    if spec.get("default_first_choice"):
        items = _choice_values(spec, base)
        if items:
            return items[0][0]
    return default


class ArgField:
    """arg 정의 하나에 대응하는 위젯 + 값 읽기."""

    def __init__(self, spec: Dict[str, Any], parent: QWidget):
        self.spec = spec
        self.flag = str(spec.get("flag", "") or "")
        self.type = str(spec.get("type", "str"))
        self.required = bool(spec.get("required", False))
        self.optional = False          # int/float 에서 "지정" 체크박스 사용 여부
        default = spec.get("default")

        self.container = QWidget(parent)
        row = QHBoxLayout(self.container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        if self.type == "bool":
            self.widget = QCheckBox()
            self.widget.setChecked(bool(default))
            row.addWidget(self.widget)
            row.addStretch(1)

        elif self.type in ("int", "float"):
            # "optional": true 이면 체크박스 "지정" 이 앞에 붙는다. 해제 상태면 인자를
            # 넘기지 않아 스크립트/yaml 기본값이 쓰인다. 0 은 유효한 명시값이다.
            self.optional = bool(spec.get("optional", False))
            if self.type == "int":
                self.widget = QSpinBox()
                lo, hi = spec.get("min"), spec.get("max")
                self.widget.setRange(
                    int(lo) if lo is not None else -1_000_000,
                    int(hi) if hi is not None else 100_000_000,
                )
            else:
                self.widget = QDoubleSpinBox()
                lo, hi = spec.get("min"), spec.get("max")
                self.widget.setRange(
                    float(lo) if lo is not None else -1e6,
                    float(hi) if hi is not None else 1e6,
                )
                self.widget.setDecimals(4)
                self.widget.setSingleStep(0.01)
            # 해제 상태에서 보여 줄 참고값(yaml 기본 등). 전달되지는 않는다.
            shown = default if default is not None else spec.get("placeholder")
            if shown is not None:
                self.widget.setValue(int(shown) if self.type == "int" else float(shown))
            if self.optional:
                self.opt_check = QCheckBox("지정")
                self.opt_check.setChecked(default is not None)
                self.opt_check.setToolTip("해제하면 인자를 넘기지 않아 스크립트/pipeline.yaml 기본값이 쓰인다")
                self.opt_check.toggled.connect(self.widget.setEnabled)
                self.widget.setEnabled(default is not None)
                row.addWidget(self.opt_check)
            row.addWidget(self.widget)
            row.addStretch(1)

        elif self.type == "choice":
            # 표시 이름과 실제 값(명령에 넘기는 문자열)이 다를 수 있다 — 값은 itemData 에 둔다.
            self.widget = QComboBox()
            # 가장 긴 항목 폭을 최소 폭으로 잡지 않게 — 항목이 길면(yaml 경로 등) 폼이 옆으로 넘쳐 '찾기' 버튼이 잘렸다
            self.widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self.widget.setMinimumContentsLength(18)
            for value, label in _choice_values(spec):
                self.widget.addItem(str(label), str(value))
            idx = self.widget.findData(str(default))
            if idx >= 0:
                self.widget.setCurrentIndex(idx)
            row.addWidget(self.widget)
            row.addStretch(1)

        else:  # str / list / path / dir
            default = resolve_default(spec)
            if spec.get("choices") or spec.get("choices_glob") or spec.get("choices_dirs"):
                # 고를 수도 직접 칠 수도 있는 콤보 — 값은 표시 문자열 그대로 (currentText)
                self.widget = QComboBox()
                self.widget.setEditable(True)
                self.widget.setInsertPolicy(QComboBox.NoInsert)
                self.widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                self.widget.setMinimumContentsLength(18)
                for value, _label in _choice_values(spec):
                    self.widget.addItem(str(value), str(value))
                self.widget.setEditText(str(default if default is not None else ""))
                line = self.widget.lineEdit()
            else:
                self.widget = QLineEdit(str(default if default is not None else ""))
                line = self.widget
            if spec.get("placeholder") is not None and line is not None:
                line.setPlaceholderText(str(spec.get("placeholder")))
            row.addWidget(self.widget, 1)
            if self.type in ("path", "dir"):
                btn = QPushButton("찾기")
                btn.setFixedWidth(56)
                btn.clicked.connect(self._browse)
                row.addWidget(btn)

        help_text = spec.get("help")
        if help_text:
            self.container.setToolTip(str(help_text))

    def _browse(self) -> None:
        if self.type == "dir":
            got = QFileDialog.getExistingDirectory(
                self.container, "폴더 선택", str(ROOT)
            )
        else:
            got, _ = QFileDialog.getOpenFileName(
                self.container, "파일 선택", str(ROOT)
            )
        if got:
            self._set_text(got)

    def _text(self) -> str:
        return self.widget.currentText() if isinstance(self.widget, QComboBox) else self.widget.text()

    def _set_text(self, s: str) -> None:
        if isinstance(self.widget, QComboBox):
            self.widget.setEditText(s)
        else:
            self.widget.setText(s)

    def set_value(self, v: Any) -> None:
        """기억해 둔 값을 위젯에 되돌린다. 타입이 안 맞으면 조용히 건너뛴다."""
        try:
            if self.type == "bool":
                self.widget.setChecked(bool(v))
            elif self.type in ("int", "float"):
                if getattr(self, "optional", False):
                    self.opt_check.setChecked(v is not None)
                if v is not None:
                    self.widget.setValue(int(v) if self.type == "int" else float(v))
            elif self.type == "choice":
                idx = self.widget.findData(str(v))
                if idx >= 0:
                    self.widget.setCurrentIndex(idx)
            else:
                self._set_text(str(v))
        except (TypeError, ValueError):
            pass

    def value(self) -> Any:
        if self.type == "bool":
            return self.widget.isChecked()
        if self.type in ("int", "float"):
            if getattr(self, "optional", False) and not self.opt_check.isChecked():
                return None
            return self.widget.value()
        if self.type == "choice":
            data = self.widget.currentData()
            return str(data) if data is not None else self.widget.currentText()
        return self._text().strip()

    def to_argv(self) -> List[str]:
        """이 필드를 커맨드라인 조각으로 바꾼다. 빈 값은 생략한다."""
        v = self.value()

        if self.type == "bool":
            # 플래그는 True 일 때만 붙인다.
            return [self.flag] if (v and self.flag) else []

        if self.type in ("int", "float"):
            # optional 이고 "지정" 해제 → 생략 (스크립트/yaml 기본값 사용)
            if v is None:
                return []
            # 0 을 "생략" 으로 쓰는 인자는 JSON 에서 "omit_if_zero": true 로 선언한다.
            # 플래그 이름을 여기 하드코딩하지 않는다.
            if self.spec.get("omit_if_zero") and not v:
                return []
            return [self.flag, str(v)] if self.flag else [str(v)]

        text = str(v)
        if not text:
            if self.required:
                raise ValueError(
                    f"'{self.spec.get('label') or self.flag}' 은(는) 필수입니다."
                )
            return []

        if self.type == "list":
            parts = text.split()
            return ([self.flag] + parts) if self.flag else parts

        return [self.flag, text] if self.flag else [text]


# =============================================================================
# 단계 실행 폼
# =============================================================================

class StagePanel(QWidget):
    """선택된 단계 하나의 설명 + 폼 + 실행 버튼 + 로그."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.stage: Optional[Dict[str, Any]] = None
        self.fields: List[ArgField] = []
        self.worker: Optional[ProcessWorker] = None
        # 같은 세션 안에서 단계를 오갈 때 사용자가 바꾼 입력값을 stage id 별로 기억한다.
        # (프로세스를 다시 켜면 JSON 기본값으로 돌아간다 — 의도된 동작)
        self._saved_values: Dict[str, Dict[str, Any]] = {}
        # 마지막 실행이 stdout 에 찍은 RESULT_HTML / RESULT_SUMMARY 마커 경로
        self._result_path: Optional[Path] = None
        self._summary_path: Optional[Path] = None
        self._marker_count = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        self.title = QLabel("단계를 선택하세요")
        f = self.title.font()
        # 앱 폰트에서 파생시킨다. QFont() 를 새로 만들어 px 기반 앱 폰트를 만나면
        # pointSize() 가 -1 이라 setPointSize 경고가 났다.
        f.setPointSizeF(max(f.pointSizeF(), 10.0) + 3.0)
        f.setBold(True)
        self.title.setFont(f)
        outer.addWidget(self.title)

        self.desc = QLabel("")
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color:#5a6673;")
        outer.addWidget(self.desc)

        self.script_label = QLabel("")
        self.script_label.setStyleSheet(
            "color:#7b8794; font-family:Consolas,monospace;"
        )
        outer.addWidget(self.script_label)

        # 핵심 단계의 '도구' 선택 (JSON stage.tools: 같은 단계를 다른 스크립트/프리셋으로) — 도구가 없으면 숨김
        self.tool_row = QWidget()
        tr = QHBoxLayout(self.tool_row)
        tr.setContentsMargins(0, 4, 0, 4)
        tool_label = QLabel("도구")
        tool_label.setObjectName("toolLabel")
        tr.addWidget(tool_label)
        self.tool_combo = QComboBox()
        self.tool_combo.setMinimumWidth(340)
        self.tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        tr.addWidget(self.tool_combo)
        tr.addStretch(1)
        self.tool_row.setVisible(False)
        outer.addWidget(self.tool_row)
        self.core_stage: Optional[Dict[str, Any]] = None
        self.stages_by_id: Dict[str, Dict[str, Any]] = {}
        self._tool_choice: Dict[str, int] = {}

        split = QSplitter(Qt.Vertical)

        # ---- 폼 ----
        form_host = QWidget()
        self.form = QFormLayout(form_host)
        self.form.setLabelAlignment(Qt.AlignRight)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_host)
        box = QGroupBox("실행 옵션")
        bl = QVBoxLayout(box)
        # 핵심 단계는 "basic" 으로 표시된 필드만 먼저 보이고 나머지는 이 체크박스로 편다 (JSON 의 basic 플래그)
        self.adv_check = QCheckBox("고급 옵션 보기")
        self.adv_check.setVisible(False)
        self.adv_check.toggled.connect(self._apply_advanced)
        bl.addWidget(self.adv_check)
        bl.addWidget(scroll)
        split.addWidget(box)

        # ---- 로그 ----
        log_box = QGroupBox("실행 로그")
        ll = QVBoxLayout(log_box)
        self.cmd_preview = QLineEdit()
        self.cmd_preview.setReadOnly(True)
        self.cmd_preview.setStyleSheet(
            "font-family:Consolas,monospace; color:#42505e; background:#f4f6f8;"
        )
        ll.addWidget(self.cmd_preview)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setStyleSheet(
            "font-family:Consolas,monospace; font-size:11px; "
            "background:#1e242b; color:#d6dde5;"
        )
        ll.addWidget(self.log)
        split.addWidget(log_box)
        split.setSizes([360, 420])
        outer.addWidget(split, 1)

        # ---- 버튼 ----
        btns = QHBoxLayout()
        self.preview_btn = QPushButton("명령 미리보기")
        self.preview_btn.clicked.connect(self._preview)
        self.run_btn = QPushButton("실행")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("중단")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.clear_btn = QPushButton("로그 지우기")
        self.clear_btn.clicked.connect(self.log.clear)
        self.open_btn = QPushButton("결과 열기")
        self.open_btn.setEnabled(False)
        self.open_btn.setToolTip("마지막 실행이 만든 HTML 을 기본 브라우저로 연다 (RESULT_HTML 마커)")
        self.open_btn.clicked.connect(self._open_result)
        btns.addWidget(self.preview_btn)
        # 실행 상태: 진행 표시(범위 0,0) + 결과 라벨(완료/실패/중단 · 소요 시간)
        self.progress = QProgressBar()
        self.progress.setObjectName("busyBar")
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(140)
        self.progress.setVisible(False)
        btns.addWidget(self.progress)
        self.state_label = QLabel("")
        self.state_label.setObjectName("runState")
        btns.addWidget(self.state_label)
        btns.addStretch(1)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.clear_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.run_btn)
        outer.addLayout(btns)

        self.set_stage(None)

    # ------------------------------------------------------------------
    def set_stage(self, stage: Optional[Dict[str, Any]]) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self, "실행 중",
                "현재 단계가 실행 중입니다. 중단하거나 끝난 뒤에 바꾸세요."
            )
            return

        self._remember_values()
        self.core_stage = stage
        tools = list((stage or {}).get("tools") or [])
        self.tool_row.setVisible(bool(tools))
        if tools:
            self.tool_combo.blockSignals(True)
            self.tool_combo.clear()
            for t in tools:
                self.tool_combo.addItem(str(t.get("label") or t.get("stage")), t)
            idx = min(self._tool_choice.get(str(stage.get("id")), 0), len(tools) - 1)
            self.tool_combo.setCurrentIndex(idx)
            self.tool_combo.blockSignals(False)
            self._build_form(self.resolve_tool(stage, tools[idx]))
        else:
            self._build_form(stage)

    def _on_tool_changed(self, idx: int) -> None:
        if self.core_stage is None or idx < 0:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self._remember_values()
        self._tool_choice[str(self.core_stage.get("id"))] = idx
        tool = self.tool_combo.itemData(idx) or {}
        self._build_form(self.resolve_tool(self.core_stage, tool))

    def resolve_tool(self, core: Dict[str, Any], tool: Dict[str, Any]) -> Dict[str, Any]:
        """도구 항목 → 실제 실행할 단계 정의. 대상 단계(tool.stage)의 스크립트·인자를 쓰되 제목은 핵심 단계 것을 쓰고,
        tool.set 의 값은 해당 인자의 기본값으로 넣어 basic 으로 올린다 (사용자가 바로 보고 고칠 수 있게)."""
        target = self.stages_by_id.get(str(tool.get("stage") or ""), core)
        eff = json.loads(json.dumps(target, ensure_ascii=False))
        eff["core_title"] = core.get("core_title") or core.get("title")
        if tool.get("description"):
            eff["description"] = str(tool["description"])
        presets = dict(tool.get("set") or {})
        for a in eff.get("args", []):
            if a.get("flag") in presets:
                a["default"] = presets[a["flag"]]
                a["basic"] = True
        eff["tool_label"] = tool.get("label")
        return eff

    def _build_form(self, stage: Optional[Dict[str, Any]]) -> None:
        self.stage = stage
        while self.form.rowCount():
            self.form.removeRow(0)
        self.fields = []
        self._result_path = None
        self._summary_path = None
        self._marker_count = 0
        self.open_btn.setEnabled(False)

        if stage is None:
            self.title.setText("단계를 선택하세요")
            self.desc.setText("")
            self.script_label.setText("")
            self.run_btn.setEnabled(False)
            self.preview_btn.setEnabled(False)
            return

        core_title = stage.get("core_title")
        self.title.setText(str(core_title or stage.get("title", stage.get("id"))))
        self.desc.setText(str(stage.get("description", "")))
        script = ROOT / str(stage["script"])
        exists = script.is_file()
        self.script_label.setText(
            f"{stage['script']}" + (f"   ·   {stage.get('title')}" if core_title else "") + ("" if exists else "   ← 파일 없음")
        )
        self.run_btn.setEnabled(exists)
        self.preview_btn.setEnabled(exists)
        if not exists:
            self.desc.setText(
                str(stage.get("description", ""))
                + f"\n\n[경고] 스크립트를 찾을 수 없습니다: {script}"
            )

        saved = self._saved_values.get(str(stage.get("id")), {})
        specs = list(stage.get("args", []))
        has_basic = any(s.get("basic") for s in specs)
        n_adv = 0
        for spec in specs:
            field = ArgField(spec, self)
            key = self._field_key(field)
            if key in saved:
                field.set_value(saved[key])
            self.fields.append(field)
            label = str(spec.get("label") or spec.get("flag") or "")
            if spec.get("required"):
                label += " *"
            self.form.addRow(label, field.container)
            # basic 이 하나라도 있으면 basic/required 가 아닌 필드는 고급으로 접는다
            field.advanced = bool(has_basic and not spec.get("basic") and not spec.get("required"))
            n_adv += int(field.advanced)
        self.adv_check.setVisible(has_basic)
        self.adv_check.setText(f"고급 옵션 보기 ({n_adv})")
        self._apply_advanced()

        self._preview()

    def _apply_advanced(self, *_args: Any) -> None:
        show = self.adv_check.isChecked()
        for field in self.fields:
            if not getattr(field, "advanced", False):
                continue
            lab = self.form.labelForField(field.container)
            if lab is not None:
                lab.setVisible(show)
            field.container.setVisible(show)

    # ------------------------------------------------------------------
    @staticmethod
    def _field_key(field: ArgField) -> str:
        return field.flag or str(field.spec.get("label") or "")

    def _remember_values(self) -> None:
        """현재 단계의 입력값을 stage id 별로 저장한다 (set_stage 직전에 부른다)."""
        if self.stage is None or not self.fields:
            return
        sid = str(self.stage.get("id"))
        try:
            self._saved_values[sid] = {self._field_key(f): f.value() for f in self.fields}
        except RuntimeError:
            # 위젯이 이미 파괴된 경우 (창 닫힘 등) — 기억을 건너뛴다
            pass

    def _scan_line(self, text: str) -> None:
        """stdout 한 줄에서 RESULT_HTML / RESULT_SUMMARY 마커를 찾는다.
        계약: 스크립트는 모든 저장을 마친 뒤 각 마커를 한 번만 찍는다. 여러 번 나오면
        마지막 것을 후보로 삼고 경고를 남긴다 (앞선 경로로 조용히 되돌아가지 않는다)."""
        m = RESULT_MARKER_RE.match(text)
        if m:
            self._marker_count += 1
            if self._marker_count == 2:
                self._append("[경고] RESULT_HTML 마커가 여러 번 나왔습니다 — 마지막 경로를 씁니다")
            self._result_path = Path(m.group("path").strip().strip('"'))
            return
        m = SUMMARY_MARKER_RE.match(text)
        if m:
            self._summary_path = Path(m.group("path").strip().strip('"'))

    def _load_warning_count(self) -> Optional[int]:
        """RESULT_SUMMARY sidecar 의 warnings 배열 길이. 없거나 못 읽으면 None."""
        sp = self._summary_path
        if sp is None:
            return None
        if not sp.is_absolute():
            sp = ROOT / sp
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        w = data.get("warnings") if isinstance(data, dict) else None
        return len(w) if isinstance(w, list) else None

    def _open_result(self) -> None:
        p = self._result_path
        if p is None:
            return
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            QMessageBox.warning(self, "결과 없음", f"파일이 없습니다:\n{p}")
            self.open_btn.setEnabled(False)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))

    # ------------------------------------------------------------------
    def _build_cmd(self) -> List[str]:
        if self.stage is None:
            return []
        cmd = [sys.executable, "-u", str(ROOT / str(self.stage["script"]))]
        blocked: List[str] = []
        for field in self.fields:
            if field.flag in DESTRUCTIVE_FLAGS:
                blocked.append(field.flag)
                continue
            cmd.extend(field.to_argv())
        if blocked:
            self._append(
                f"[차단] 파괴적 인자를 제외했습니다: {', '.join(blocked)}"
            )
        return cmd

    def _preview(self) -> None:
        try:
            cmd = self._build_cmd()
        except ValueError as e:
            self.cmd_preview.setText(f"(입력 필요) {e}")
            return
        if not cmd:
            self.cmd_preview.setText("")
            return
        shown = [Path(cmd[2]).name if len(cmd) > 2 else ""] + cmd[3:]
        self.cmd_preview.setText("python -u " + " ".join(shown))

    # ------------------------------------------------------------------
    def _append(self, text: str) -> None:
        self.log.appendPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def _append_html(self, html: str) -> None:
        self.log.appendHtml(html)
        self.log.moveCursor(QTextCursor.End)

    def _run(self) -> None:
        if self.stage is None:
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "실행 중", "이미 실행 중입니다.")
            return

        try:
            cmd = self._build_cmd()
        except ValueError as e:
            QMessageBox.warning(self, "입력 확인", str(e))
            return
        if not cmd:
            return

        self._preview()
        self._append("=" * 70)
        self._append(f"실행: {self.stage.get('title')}")
        self._append(" ".join(cmd))
        self._append("=" * 70)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self._result_path = None
        self._summary_path = None
        self._marker_count = 0
        self._run_started = time.time()
        self.progress.setVisible(True)
        self.state_label.setText("실행 중…")
        self.state_label.setStyleSheet("color:#5a6673; font-weight:600;")

        self.worker = ProcessWorker(cmd, ROOT, self)
        self.worker.line.connect(self._append)
        self.worker.line.connect(self._scan_line)
        self.worker.finished_with.connect(self._done)
        self.worker.start()

    def _stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._append("[중단 요청]")
            self.worker.stop()

    def _done(self, code: int) -> None:
        # 판정: 사용자가 중단 버튼을 눌렀으면 "중단", 아니면 종료 코드 0 만 "완료",
        # 그 외 전부 "실패". 130 같은 특정 값에 의존하지 않는다 (Windows 에서는
        # CTRL_BREAK 로 죽은 자식이 다른 코드를 낼 수 있다).
        elapsed = time.time() - getattr(self, "_run_started", time.time())
        stopped = bool(self.worker is not None and getattr(self.worker, "_stopped", False))
        n_warn = self._load_warning_count() if (code == 0 and not stopped) else None
        if stopped:
            mark, color, word = "■", "#5a6673", "중단됨"
        elif code == 0 and n_warn:
            # 스크립트는 경고가 있어도 유효한 결과를 만들었으면 0 으로 끝난다.
            # sidecar 의 warnings 를 읽어 "완료" 와 구분한다 (호박색).
            mark, color, word = "✔", "#b26a00", f"완료 · 경고 {n_warn}"
        elif code == 0:
            mark, color, word = "✔", "#2e7d32", "완료"
        else:
            mark, color, word = "✖", "#c62828", "실패"
        # 음수 코드는 Windows NTSTATUS 인 경우가 많아 hex 를 병기한다 (예: -1073741510 = 0xC000013A CTRL_BREAK)
        code_text = f"{code}" if code >= 0 else f"{code} (0x{code & 0xFFFFFFFF:08X})"
        summary = f"{mark} {word} · 종료 코드 {code_text} · {elapsed:.1f}s"
        self._append("")
        self._append_html(f'<span style="color:{color}; font-weight:700;">{summary}</span>')
        self._append("")
        self.progress.setVisible(False)
        self.state_label.setText(summary)
        self.state_label.setStyleSheet(f"color:{color}; font-weight:600;")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # 산출물 마커: 정상 종료 + 파일 존재 → "결과 열기" 활성
        rp = self._result_path
        if rp is not None and not rp.is_absolute():
            rp = ROOT / rp
        if rp is not None:
            if code == 0 and not stopped and rp.is_file():
                self._result_path = rp
                self.open_btn.setEnabled(True)
                self._append_html(
                    f'<span style="color:#2a78d6;">결과 HTML: {html_escape(str(rp))}'
                    f' — "결과 열기" 버튼으로 엽니다</span>'
                )
            elif not rp.is_file():
                self._append(f"[경고] 스크립트가 알린 결과 파일이 없습니다: {rp}")


# =============================================================================
# 그룹 페이지 (영상 / 이미지 / 평가)
# =============================================================================

class PipelineGroupPage(QWidget):
    def __init__(self, group: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.group = group

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        left = QWidget()
        left.setMaximumWidth(280)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        head = QLabel(str(group.get("description", "")))
        head.setWordWrap(True)
        head.setStyleSheet("color:#5a6673; padding:2px 4px 8px 4px;")
        ll.addWidget(head)

        # 핵심 단계(core) 가 있으면 그것만 먼저 보이고, 나머지는 '추가 작업 보기' 로 편다.
        stages = list(group.get("stages", []))
        core = [st for st in stages if st.get("core")]
        rest = [st for st in stages if not st.get("core")]
        self.listw = QListWidget()
        self.listw.setWordWrap(True)
        self._extra_items: List[QListWidgetItem] = []
        for st in (core + rest) if core else stages:
            item = QListWidgetItem(str(st.get("core_title") or st.get("title", st.get("id"))))
            item.setData(Qt.UserRole, st)
            if not (ROOT / str(st.get("script", ""))).is_file():
                item.setForeground(Qt.red)
                item.setToolTip(f"스크립트 없음: {st.get('script')}")
            elif st.get("core") and st.get("title"):
                item.setToolTip(str(st.get("title")))
            self.listw.addItem(item)
            if core and not st.get("core"):
                item.setHidden(True)
                self._extra_items.append(item)
        ll.addWidget(self.listw, 1)
        self.more_check: Optional[QCheckBox] = None
        if core and rest:
            self.more_check = QCheckBox(f"추가 작업 보기 ({len(rest)})")
            self.more_check.setToolTip("개별 리포트·라벨·내보내기 등 세부 단계")
            self.more_check.toggled.connect(self._toggle_extra)
            ll.addWidget(self.more_check)
        layout.addWidget(left)

        self.panel = StagePanel(self)
        self.panel.stages_by_id = {str(st.get("id")): st for st in stages}
        layout.addWidget(self.panel, 1)

        self.listw.currentItemChanged.connect(self._on_select)
        if self.listw.count():
            self.listw.setCurrentRow(0)

    def _on_select(self, cur: Optional[QListWidgetItem], _prev) -> None:
        self.panel.set_stage(cur.data(Qt.UserRole) if cur else None)

    def _toggle_extra(self, on: bool) -> None:
        for item in self._extra_items:
            item.setHidden(not on)
        cur = self.listw.currentItem()
        if not on and cur is not None and cur.isHidden() and self.listw.count():
            self.listw.setCurrentRow(0)

    def visible_titles(self) -> List[str]:
        return [self.listw.item(i).text() for i in range(self.listw.count()) if not self.listw.item(i).isHidden()]


class PipelinePage(QWidget):
    """search_gui.py 가 탭으로 붙이는 최상위 위젯."""

    def __init__(self, group_id: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            groups = load_registry()
        except RegistryError as e:
            msg = QLabel(str(e))
            msg.setWordWrap(True)
            msg.setStyleSheet("color:#b3261e; padding:16px;")
            layout.addWidget(msg)
            layout.addStretch(1)
            return

        group = next((g for g in groups if g.get("id") == group_id), None)
        if group is None:
            msg = QLabel(
                f"gui_pipelines.json 에 '{group_id}' 그룹이 없습니다.\n"
                f"있는 그룹: {[g.get('id') for g in groups]}"
            )
            msg.setWordWrap(True)
            msg.setStyleSheet("color:#b3261e; padding:16px;")
            layout.addWidget(msg)
            layout.addStretch(1)
            return

        layout.addWidget(PipelineGroupPage(group, self))


def available_groups() -> List[Dict[str, str]]:
    """MainWindow 가 탭을 만들 때 쓴다. 실패해도 GUI 는 떠야 한다."""
    try:
        return [
            {"id": str(g.get("id")), "title": str(g.get("title", g.get("id")))}
            for g in load_registry()
        ]
    except RegistryError:
        return []
