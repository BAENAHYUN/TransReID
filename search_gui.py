from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QSettings, Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap

from gui.gui_theme import apply_theme

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
except Exception:
    QAudioOutput = None
    QMediaPlayer = None
    QVideoWidget = None
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# GUI에서는 SOLIDER 후보 Pool을 노출하지 않는다.
# 사람 crop 검색의 2차 SOLIDER rerank용 내부 후보 수 기본값.
SOLIDER_POOL_DEFAULT = 200

# Qwen 별도 후처리 기본값. GUI에는 노출하지 않는다.
QWEN_ALPHA_DEFAULT = 0.70
QWEN_THRESHOLD_DEFAULT = 0.50
QWEN_VERIFY_MODE_DEFAULT = "flag"
QWEN_TEXT_SCRIPT = ROOT / "verifiers" / "qwen_stage.py"
QWEN_CROP_SCRIPT = ROOT / "verifiers" / "qwen_crop_stage.py"

# 영상 검색 내부 기본값. GUI에는 노출하지 않는다.
VIDEO_GROUP_SIZE_DEFAULT = 3
VIDEO_TEXT_CANDIDATE_K_DEFAULT = 100
VIDEO_PERSON_VECTOR_DEFAULT = "solider"
VIDEO_OBJECT_VECTOR_DEFAULT = "dinov2"

# GUI 시작 시 검색 backend를 import하지 않는다.
# 모델/Router/SearchEngine 관련 import는 사용자가 검색 버튼을 누른 뒤
# worker thread 안에서만 수행한다.
CONFIG_PATH = ROOT / "pipeline.yaml"
_SEARCH_BACKEND = None


def get_search_backend():
    """검색 실행 버튼으로 생성된 worker job 안에서만 unified_search_4mode를 import한다."""
    global _SEARCH_BACKEND
    if _SEARCH_BACKEND is None:
        import importlib
        _SEARCH_BACKEND = importlib.import_module("search.unified_search_4mode")
    return _SEARCH_BACKEND


# ---------------------------------------------------------------------------
# 검색 탭 모델 선택. 후보는 pipeline.yaml 의 retrievers (config.py 만 읽는다 — torch 없음).
# 콤보 항목의 data 는 항상 retriever 이름(str) 또는 이름 목록(list) 이고, 백엔드가 다시 검증한다.
# ---------------------------------------------------------------------------
MODEL_LABELS = {"siglip2": "SigLIP2", "irra": "IRRA", "solider": "SOLIDER", "dinov2": "DINOv2"}
DEFAULT_STAGE1 = {"person": ["siglip2", "irra"], "object": ["siglip2", "dinov2"]}
DEFAULT_RERANK: Dict[str, Optional[str]] = {"person": "solider", "object": None}
FALLBACK_MODEL_OPTIONS = {
    "person_image": ["siglip2", "irra", "solider"],
    "person_text": ["siglip2", "irra"],
    "object_image": ["siglip2", "dinov2"],
    "object_text": ["siglip2"],
}


def model_label(name: Any) -> str:
    return MODEL_LABELS.get(str(name), str(name))


def combo_label(names: Sequence[str]) -> str:
    names = list(names)
    text = " + ".join(model_label(n) for n in names)
    return f"{text} (RRF 조합)" if len(names) > 1 else text


def load_search_model_options(config_path: str) -> Dict[str, List[str]]:
    """{person|object}_{image|text} → retriever 이름 목록. yaml 을 못 읽으면 운영 기본값."""
    try:
        from config import PipelineConfig

        cfg = PipelineConfig.load(config_path)
        out: Dict[str, List[str]] = {}
        for scope, specs in (("person", cfg.for_person()), ("object", cfg.for_object())):
            out[f"{scope}_image"] = [s.name for s in specs]
            out[f"{scope}_text"] = [s.name for s in specs if s.supports_text]
        if all(out.get(k) for k in ("person_image", "object_image")):
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"[search_gui] retriever 목록을 읽지 못해 기본값을 씁니다: {exc}", file=sys.stderr)
    return {k: list(v) for k, v in FALLBACK_MODEL_OPTIONS.items()}


def stage1_choices(scope: str, names: Sequence[str]) -> List[List[str]]:
    """crop 검색 1차 후보 선택지: 단일 각각 → 운영 기본 조합 → (다르면) 전체 조합."""
    names = list(names)
    out: List[List[str]] = [[n] for n in names]
    preset = [n for n in DEFAULT_STAGE1.get(scope, []) if n in names]
    if len(preset) > 1:
        out.append(preset)
    if len(names) > 1 and names != preset:
        out.append(list(names))
    return out


def fill_combo(combo: QComboBox, items: Sequence[Tuple[str, Any]], default: Any) -> None:
    """(label, data) 로 채우고 data == default 인 항목을 고른다. 시그널은 채우는 동안 막는다."""
    combo.blockSignals(True)
    try:
        combo.clear()
        for label, data in items:
            combo.addItem(label, data)
        for i in range(combo.count()):
            if combo.itemData(i) == default:
                combo.setCurrentIndex(i)
                break
    finally:
        combo.blockSignals(False)


def fmt_time(seconds: Any) -> str:
    """GUI 표시용 시간 포맷. 검색 backend import 없이 동작한다."""
    try:
        value = max(0.0, float(seconds or 0.0))
    except (TypeError, ValueError):
        value = 0.0
    minutes = int(value // 60)
    secs = value - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


# Qwen은 기본 검색 프로세스와 분리한다.
# GUI/기본 검색에서는 qwen_stage.py를 import하지 않으며,
# 자연어 이미지 검색 결과가 나온 뒤 [Qwen 재검증] 버튼을 눌렀을 때만
# 별도 Python subprocess로 qwen_stage.py를 실행한다.


def run_qwen_subprocess(
    payload: Dict[str, Any],
    *,
    top_k: int,
    alpha: float,
    threshold: float,
    verify_mode: str,
    script_path: Optional[Path] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Run a Qwen post-process script only as an external process.

    This helper never imports qwen_stage/qwen_crop_stage. It is called only
    from the [Qwen 검증 실행] button handlers.
    """
    script = Path(script_path or QWEN_TEXT_SCRIPT)
    if not script.is_file():
        raise FileNotFoundError(f"Qwen 후처리 스크립트를 찾을 수 없습니다: {script}")

    with tempfile.TemporaryDirectory(prefix="transreid_qwen_") as tmp_dir:
        tmp = Path(tmp_dir)
        input_json = tmp / "base_search_result.json"
        output_json = tmp / "qwen_result.json"
        input_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--in", str(input_json),
            "--out", str(output_json),
            "--top-k", str(top_k),
            "--alpha", str(alpha),
            "--threshold", str(threshold),
            "--verify-mode", str(verify_mode),
            "--dtype", "bfloat16",
            "--max-pixels", str(768 * 768),
            "--show", str(top_k),
        ]
        # 관찰 배치(>1): 후보당 시간이 크게 줄지만 판정이 일부 달라질 수 있다 (qwen_stage --batch-size 참고)
        if int(batch_size or 1) > 1:
            cmd += ["--batch-size", str(int(batch_size))]

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")

        # Important: Qwen starts here, and nowhere else in the GUI.
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            detail = (stderr or stdout or "").strip()
            raise RuntimeError(
                "qwen_stage.py 별도 후처리 프로세스가 실패했습니다.\n"
                f"returncode={proc.returncode}\n\n{detail}"
            )

        if not output_json.is_file():
            raise RuntimeError(
                "qwen_stage.py는 종료됐지만 결과 JSON이 생성되지 않았습니다.\n"
                f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
            )

        return json.loads(output_json.read_text(encoding="utf-8"))


def qwen_reranker_note(qwen_payload: Dict[str, Any]) -> str:
    """Qwen 후처리 결과에 Reranker 생략 사유가 있으면 상태 문구용 짧은 표시를 만든다."""
    reason = str((qwen_payload or {}).get("reranker_skipped_reason") or "").strip()
    if not reason:
        return ""
    if "qwen_vl_utils" in reason or "qwen-vl-utils" in reason:
        short = "qwen-vl-utils 미설치"
    else:
        short = reason.split("\n")[0][:60]
    return f" · Reranker 생략({short}) → Instruct 검증만 적용"


# =============================================================================
# Worker
# =============================================================================

class SearchWorker(QThread):
    succeeded = Signal(dict)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, job: Callable[[], Dict[str, Any]], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.job = job

    def run(self) -> None:
        try:
            self.progress.emit("검색 실행 중...")
            result = self.job()
            self.succeeded.emit(result)
        except Exception as exc:  # noqa: BLE001
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.failed.emit(detail)


# =============================================================================
# Common widgets
# =============================================================================

def safe_path(value: Any) -> Optional[Path]:
    if not value:
        return None
    try:
        p = Path(str(value))
    except Exception:
        return None
    return p if p.is_file() else None


class PathResolver:
    """Qdrant에 저장된 다른 PC의 절대경로를 현재 PC 경로로 재연결한다."""

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.video_root = self.project_root / "data" / "videos"
        # video_tracks/person, video_tracks/object, crops 등 여러 위치를 한 번에 찾기 위해 data를 기본 root로 둔다.
        self.crop_root = self.project_root / "data"
        self._video_cache: Dict[str, Optional[Path]] = {}
        self._crop_cache: Dict[str, Optional[Path]] = {}

    def set_video_root(self, value: Any) -> None:
        if value:
            self.video_root = Path(str(value)).expanduser()
        self._video_cache.clear()

    def set_crop_root(self, value: Any) -> None:
        if value:
            self.crop_root = Path(str(value)).expanduser()
        self._crop_cache.clear()

    @staticmethod
    def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
        value = row.get("payload")
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _first_file(candidates: List[Any]) -> Optional[Path]:
        for value in candidates:
            p = safe_path(value)
            if p is not None:
                return p
        return None

    @staticmethod
    def _basename(value: Any) -> Optional[str]:
        if not value:
            return None
        try:
            name = Path(str(value).replace("\\", "/")).name
        except Exception:
            return None
        return name or None

    @staticmethod
    def _suffix_after_anchor(value: Any, anchors: tuple[str, ...]) -> Optional[Path]:
        if not value:
            return None
        text = str(value).replace("\\", "/")
        parts = [x for x in text.split("/") if x]
        low = [x.lower() for x in parts]
        for anchor in anchors:
            if anchor.lower() in low:
                idx = low.index(anchor.lower())
                tail = parts[idx + 1:]
                return Path(*tail) if tail else None
        return None

    @staticmethod
    def _search_by_name(root: Path, name: Optional[str], cache: Dict[str, Optional[Path]]) -> Optional[Path]:
        if not name:
            return None
        key = name.lower()
        if key in cache:
            return cache[key]
        if not root.exists() or not root.is_dir():
            cache[key] = None
            return None
        try:
            for p in root.rglob(name):
                if p.is_file():
                    cache[key] = p
                    return p
        except OSError:
            pass
        cache[key] = None
        return None

    def resolve_video(self, row: Dict[str, Any]) -> Optional[Path]:
        payload = self._payload(row)

        # 1) DB에 저장된 실제 경로가 현재 PC에서도 유효하면 그대로 사용.
        direct = self._first_file([
            row.get("video_path"), payload.get("video_path"),
            row.get("video_relpath"), payload.get("video_relpath"),
            row.get("image_id"), payload.get("image_id"),
        ])
        if direct is not None and direct.suffix.lower() in self.VIDEO_EXTS:
            return direct

        # 2) 장기 호환용 relpath 또는 예전 절대경로의 /videos/ 뒤쪽을 현재 VIDEO_ROOT에 붙인다.
        for value in (
            row.get("video_relpath"), payload.get("video_relpath"),
            row.get("video_path"), payload.get("video_path"),
            row.get("image_id"), payload.get("image_id"),
        ):
            if not value:
                continue
            raw = Path(str(value))
            if not raw.is_absolute():
                candidate = self.video_root / raw
                if candidate.is_file():
                    return candidate
            tail = self._suffix_after_anchor(value, ("videos",))
            if tail is not None:
                candidate = self.video_root / tail
                if candidate.is_file():
                    return candidate

        # 3) video / video_name / video_path basename으로 현재 VIDEO_ROOT 재탐색.
        names = [
            self._basename(row.get("video")),
            self._basename(payload.get("video")),
            self._basename(payload.get("video_name")),
            self._basename(row.get("video_path")),
            self._basename(payload.get("video_path")),
        ]
        for name in names:
            if not name:
                continue
            # 'Normal_Videos_935_x264' 처럼 확장자가 빠진 이름(final_db_candidates 계열 payload)은
            # 영상 확장자를 붙여서 찾는다. 확장자가 있으면 그대로 찾는다.
            has_ext = Path(name).suffix.lower() in self.VIDEO_EXTS
            variants = [name] if has_ext else [name + ext for ext in sorted(self.VIDEO_EXTS)]
            for variant in variants:
                direct_path = self.video_root / variant
                if direct_path.is_file():
                    return direct_path
                found = self._search_by_name(self.video_root, variant, self._video_cache)
                if found is not None:
                    return found
        return None

    def resolve_crop(self, row: Dict[str, Any]) -> Optional[Path]:
        payload = self._payload(row)

        # 1) 저장 경로가 현재 PC에서 유효하면 그대로 사용.
        direct = self._first_file([
            row.get("crop_path"), payload.get("crop_path"),
            row.get("crop_relpath"), payload.get("crop_relpath"),
        ])
        if direct is not None:
            return direct

        # 2) 상대경로 또는 과거 절대경로에서 data/ 이하 suffix를 복구.
        for value in (
            row.get("crop_relpath"), payload.get("crop_relpath"),
            row.get("crop_path"), payload.get("crop_path"),
        ):
            if not value:
                continue
            raw = Path(str(value))
            if not raw.is_absolute():
                candidate = self.crop_root / raw
                if candidate.is_file():
                    return candidate

            tail = self._suffix_after_anchor(
                value, ("data", "video_tracks", "crops", "query_crops")
            )
            if tail is not None:
                # data/ 뒤쪽이면 crop_root가 data일 때 바로 붙인다. 다른 anchor면 이름 검색 fallback도 남겨둔다.
                candidate = self.crop_root / tail
                if candidate.is_file():
                    return candidate

        # 3) 최후에는 filename으로 CROP_ROOT 전체를 재탐색.
        names = [
            self._basename(row.get("crop_path")),
            self._basename(payload.get("crop_path")),
        ]
        for name in names:
            found = self._search_by_name(self.crop_root, name, self._crop_cache)
            if found is not None:
                return found
        return None


def resolve_data_root(explicit: Optional[str], config_path: Optional[Path]) -> Path:
    """crop/영상 파일이 있는 data 폴더를 정한다.

    우선순위: --data-root > 환경변수 TRANSREID_DATA_ROOT > <스크립트 폴더>/data > <pipeline yaml 폴더>/data
              > <스크립트 상위 폴더>/data (프로젝트 안에 만든 부분 복사본에서 실행할 때) > <스크립트 폴더>/data (없어도 기본값).
    payload 의 crop_path 는 'data/...' 상대경로라, 스크립트가 다른 폴더에 복사돼 있으면 여기서 실제 위치를 찾아야 미리보기가 뜬다.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("TRANSREID_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [ROOT / "data"]
    if config_path is not None:
        candidates.append(Path(config_path).resolve().parent / "data")
    candidates.append(ROOT.parent / "data")
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return ROOT / "data"


def set_preview(label: QLabel, path_value: Any, *, fallback: str = "미리보기 없음") -> None:
    path = safe_path(path_value)
    if path is None:
        label.setPixmap(QPixmap())
        label.setText(fallback)
        return

    pix = QPixmap(str(path))
    if pix.isNull():
        label.setPixmap(QPixmap())
        label.setText(fallback)
        return

    label.setText("")
    label.setPixmap(
        pix.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    )


def compact_score(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "N/A"


def rows_for_top_k(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Return exactly the first Top-K rows and synchronize the displayed rank."""
    k = max(0, int(top_k))
    out: List[Dict[str, Any]] = []
    for display_rank, original in enumerate(rows[:k], start=1):
        row = dict(original)
        if row.get("backend_rank") is None and row.get("rank") is not None:
            row["backend_rank"] = row.get("rank")
        row["rank"] = display_rank
        out.append(row)
    return out


class ResultCard(QFrame):
    def __init__(self, row: Dict[str, Any], *, video: bool = False, resolver: Optional[PathResolver] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("resultCard")
        self.setFrameShape(QFrame.StyledPanel)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        thumb = QLabel("No image")
        thumb.setFixedSize(92, 92)
        thumb.setAlignment(Qt.AlignCenter)
        thumb.setObjectName("thumbnail")
        crop_for_preview = resolver.resolve_crop(row) if resolver is not None else row.get("crop_path")
        set_preview(thumb, crop_for_preview)
        layout.addWidget(thumb)

        text = QVBoxLayout()
        text.setSpacing(3)

        rank = row.get("rank", "-")
        label = row.get("label") or ("person" if row.get("is_person") else "object")
        if row.get("final_score") is not None:
            score = row.get("final_score")
            score_name = "final"
        elif row.get("rrf_score") is not None:
            score = row.get("rrf_score")
            score_name = "RRF"
        else:
            score = row.get("score")
            score_name = "score"

        verified = row.get("verified")
        verify_mark = ""
        if verified is True:
            verify_mark = "  ✓ Qwen"
        elif verified is False:
            verify_mark = "  ✕ Qwen"
        elif row.get("attr_score") is not None or row.get("attr_skipped"):
            verify_mark = "  ? Qwen"

        title = QLabel(
            f"#{rank}  {label}   {score_name} {compact_score(score)}{verify_mark}"
        )
        title.setObjectName("resultTitle")
        text.addWidget(title)

        if video:
            video_name = row.get("video") or row.get("video_path") or ""
            ts = fmt_time(row.get("timestamp_sec"))
            gs = row.get("group_summary") or {}
            start = fmt_time(gs.get("start_sec"))
            end = fmt_time(gs.get("end_sec"))
            text.addWidget(QLabel(f"video: {video_name}"))
            text.addWidget(QLabel(f"대표 {ts} · 구간 {start} ~ {end}"))
            text.addWidget(QLabel(f"track_key: {row.get('track_key') or row.get('group_id') or ''}"))
            hidden = int(row.get("same_video_hidden") or 0)
            if hidden > 0:
                more = QLabel(f"같은 영상에서 {hidden}건 더 있음 · 영상당 상한으로 접힘")
                more.setObjectName("subtleLabel")
                text.addWidget(more)
        else:
            text.addWidget(QLabel(f"image_id: {row.get('image_id', '')}"))
            retrieval = row.get("retrieval_score")
            if retrieval is not None:
                text.addWidget(QLabel(f"Qdrant retrieval: {compact_score(retrieval)}"))
            payload_row = row.get("payload") or {}
            rerank_score = payload_row.get("rerank_score", payload_row.get("solider_score"))
            if rerank_score is not None:
                rerank_name = model_label(payload_row.get("rerank_vector") or "solider")
                text.addWidget(QLabel(f"{rerank_name}: {compact_score(rerank_score)}"))
            if row.get("attr_score") is not None:
                text.addWidget(
                    QLabel(
                        f"Qwen attr={compact_score(row.get('attr_score'))} "
                        f"coverage={compact_score(row.get('attr_coverage'))}"
                    )
                )

        text.addStretch(1)
        layout.addLayout(text, 1)


class ResultsPanel(QWidget):
    def __init__(
        self,
        *,
        video: bool,
        resolver: Optional[PathResolver] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.video = video
        self.resolver = resolver or PathResolver(ROOT)
        self.rows: List[Dict[str, Any]] = []
        self._loaded_video: Optional[Path] = None
        self._pending_seek_ms = 0
        self._autoplay_after_load = False
        self._seek_pending = False

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        # 결과 목록
        list_box = QWidget()
        list_layout = QVBoxLayout(list_box)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(QLabel("검색 결과"))

        self.list = QListWidget()
        self.list.setSpacing(4)
        self.list.currentRowChanged.connect(self._show_detail)
        list_layout.addWidget(self.list, 1)
        splitter.addWidget(list_box)

        # 상세 영역
        detail_box = QWidget()
        detail_layout = QVBoxLayout(detail_box)
        detail_layout.setContentsMargins(8, 0, 0, 0)
        detail_layout.addWidget(QLabel("선택 결과 상세"))

        self.preview = QLabel("결과를 선택하세요")
        self.preview.setObjectName("detailPreview")
        self.preview.setMinimumSize(320, 220)
        self.preview.setAlignment(Qt.AlignCenter)

        self.media_tabs: Optional[QTabWidget] = None
        self.player = None
        self.audio = None
        self.video_widget = None

        if self.video:
            self.media_tabs = QTabWidget()
            if QMediaPlayer is not None and QAudioOutput is not None and QVideoWidget is not None:
                self.video_widget = QVideoWidget()
                self.video_widget.setMinimumSize(320, 220)
                self.player = QMediaPlayer(self)
                self.audio = QAudioOutput(self)
                self.player.setAudioOutput(self.audio)
                self.player.setVideoOutput(self.video_widget)
                self.player.mediaStatusChanged.connect(self._on_media_status_changed)
                self.player.errorOccurred.connect(self._on_media_error)
                self.media_tabs.addTab(self.video_widget, "영상")
            else:
                unavailable = QLabel("Qt Multimedia를 사용할 수 없어 내부 영상 재생이 비활성화되었습니다.")
                unavailable.setAlignment(Qt.AlignCenter)
                unavailable.setMinimumSize(320, 220)
                self.media_tabs.addTab(unavailable, "영상")
            self.media_tabs.addTab(self.preview, "Crop")
            detail_layout.addWidget(self.media_tabs)
        else:
            detail_layout.addWidget(self.preview)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        detail_layout.addWidget(self.detail, 1)

        if self.video:
            btns = QHBoxLayout()

            self.play_btn = QPushButton("해당 시점 재생")
            self.play_btn.clicked.connect(self._play_current_video)
            btns.addWidget(self.play_btn)

            self.pause_btn = QPushButton("일시정지")
            self.pause_btn.clicked.connect(self._pause_video)
            btns.addWidget(self.pause_btn)

            btns.addStretch(1)
            detail_layout.addLayout(btns)

        splitter.addWidget(detail_box)
        splitter.setSizes([620, 500])

    def clear(self) -> None:
        self.rows = []
        self.list.clear()
        self.detail.clear()
        set_preview(self.preview, None, fallback="결과를 선택하세요")
        if self.player is not None:
            self.player.stop()
        self._loaded_video = None
        self._seek_pending = False

    def set_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.clear()
        self.rows = rows

        for row in rows:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, row)
            card = ResultCard(row, video=self.video, resolver=self.resolver)
            item.setSizeHint(card.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, card)

        if rows:
            self.list.setCurrentRow(0)

    def _current_row(self) -> Optional[Dict[str, Any]]:
        idx = self.list.currentRow()
        if idx < 0 or idx >= len(self.rows):
            return None
        return self.rows[idx]

    @staticmethod
    def _timestamp_sec(row: Dict[str, Any]) -> float:
        payload = row.get("payload") or {}
        value = row.get("timestamp_sec")
        if value is None:
            value = payload.get("timestamp_sec", payload.get("time_sec", 0.0))
        try:
            return max(0.0, float(value or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _show_detail(self, index: int) -> None:
        if index < 0 or index >= len(self.rows):
            return
        row = self.rows[index]
        payload = row.get("payload") or {}

        resolved_crop = self.resolver.resolve_crop(row)
        set_preview(self.preview, resolved_crop)

        lines = [
            f"rank: {row.get('rank')}",
            f"label: {row.get('label')}",
            f"score: {compact_score(row.get('score'))}",
        ]

        if self.video:
            gs = row.get("group_summary") or {}
            resolved_video = self.resolver.resolve_video(row)
            ts = self._timestamp_sec(row)
            bbox_space = payload.get("bbox_space")
            lines.extend([
                f"rrf_score: {compact_score(row.get('rrf_score')) if row.get('rrf_score') is not None else 'N/A'}",
                f"identity_id: {row.get('identity_id')}",
                f"stitched_id: {row.get('stitched_id')}",
                f"track_key: {row.get('track_key') or row.get('group_id')}",
                f"video: {row.get('video') or payload.get('video')}",
                f"DB video_path: {row.get('video_path') or payload.get('video_path')}",
                f"resolved video_path: {str(resolved_video) if resolved_video else 'NOT FOUND'}",
                f"frame_idx: {row.get('frame_idx')}",
                f"timestamp_sec: {ts:.3f}",
                f"timestamp: {fmt_time(ts)}",
                f"range: {fmt_time(gs.get('start_sec'))} ~ {fmt_time(gs.get('end_sec'))}",
                f"points: {gs.get('count')}",
                f"original_tracks: {gs.get('original_tracks')}",
                f"same_video_hidden: {row.get('same_video_hidden')}",
                f"cluster_id: {row.get('cluster_id')}",
                f"DB crop_path: {row.get('crop_path') or payload.get('crop_path')}",
                f"resolved crop_path: {str(resolved_crop) if resolved_crop else 'NOT FOUND'}",
                f"bbox: {payload.get('bbox', row.get('bbox'))}",
                f"bbox_space: {bbox_space}",
            ])
            if bbox_space == "frame":
                lines.append("bbox overlay: 가능 (원본 frame 좌표)")
            elif bbox_space == "crop":
                lines.append("bbox overlay: 금지 (crop 자기 좌표계)")
            elif bbox_space:
                lines.append(f"bbox overlay: 미정의 좌표계 ({bbox_space})")
            if row.get("vector_scores"):
                lines.append("vector_scores: " + json.dumps(row.get("vector_scores"), ensure_ascii=False))
            if row.get("vector_ranks"):
                lines.append("vector_ranks: " + json.dumps(row.get("vector_ranks"), ensure_ascii=False))

            # 결과 선택 시 영상은 미리 로드하고 timestamp 위치로 seek만 한다.
            if resolved_video is not None:
                self._load_video(resolved_video, ts, autoplay=False)
                if self.media_tabs is not None:
                    self.media_tabs.setCurrentIndex(0)
            else:
                if self.player is not None:
                    self.player.stop()
                self._loaded_video = None
                if self.media_tabs is not None:
                    self.media_tabs.setCurrentIndex(1)
        else:
            lines.extend([
                f"retrieval_score: {compact_score(row.get('retrieval_score'))}",
                f"point_id: {row.get('point_id')}",
                f"image_id: {row.get('image_id')}",
                f"media_type: {row.get('media_type')}",
                f"bbox: {row.get('bbox')}",
                f"frame_idx: {row.get('frame_idx')}",
                f"track_id: {row.get('track_id')}",
                f"detection_id: {payload.get('detection_id')}",
                f"person_id: {row.get('person_id')}",
                f"object_id: {row.get('object_id')}",
                f"cluster_id: {row.get('cluster_id')}",
                f"DB crop_path: {row.get('crop_path')}",
                f"resolved crop_path: {str(resolved_crop) if resolved_crop else 'NOT FOUND'}",
            ])
            if payload.get("rerank_score") is not None:
                lines.append(f"rerank_score ({payload.get('rerank_vector') or 'solider'}): {compact_score(payload.get('rerank_score'))}")
            elif payload.get("solider_score") is not None:
                lines.append(f"solider_score: {compact_score(payload.get('solider_score'))}")

            if (
                row.get("attr_score") is not None
                or row.get("verified") is not None
                or row.get("attr_skipped")
            ):
                lines.extend([
                    "",
                    "[Qwen verification]",
                    f"verified: {row.get('verified')}",
                    f"final_score: {compact_score(row.get('final_score'))}",
                    f"attr_score: {compact_score(row.get('attr_score'))}",
                    f"attr_match: {compact_score(row.get('attr_match'))}",
                    f"attr_coverage: {compact_score(row.get('attr_coverage'))}",
                    f"summary: {row.get('attr_summary') or ''}",
                    f"failed_required: {row.get('failed_required') or []}",
                    f"skipped: {row.get('attr_skipped') or ''}",
                ])

        self.detail.setPlainText("\n".join(lines))

    # Qt(FFmpeg 백엔드)는 재생 중에도 LoadedMedia / BufferingMedia / BufferedMedia 상태를 반복해서 보낸다.
    # 예전 코드는 상태가 바뀔 때마다 seek + pause 를 다시 해서 재생 위치가 같은 시점으로 계속 되돌아갔고,
    # 그래서 '해당 시점 재생' 을 눌러도 화면이 멈춘 것처럼 보였다. seek 는 로드 뒤 1회만(_seek_pending) 적용한다.

    def _player_ready(self) -> bool:
        if self.player is None or QMediaPlayer is None:
            return False
        status = self.player.mediaStatus()
        return status in {
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferingMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.EndOfMedia,
        }

    def _apply_seek(self) -> None:
        """대기 중인 seek 를 한 번 적용하고 재생/일시정지 상태를 맞춘다."""
        if self.player is None:
            return
        self._seek_pending = False
        self.player.setPosition(self._pending_seek_ms)
        if self._autoplay_after_load:
            self.player.play()
        else:
            self.player.pause()

    def _load_video(self, path: Path, timestamp_sec: float, *, autoplay: bool) -> None:
        if self.player is None:
            return
        path = path.resolve()
        self._pending_seek_ms = max(0, int(timestamp_sec * 1000.0))
        self._autoplay_after_load = bool(autoplay)
        self._seek_pending = True

        if self._loaded_video != path:
            self._loaded_video = path
            self.player.setSource(QUrl.fromLocalFile(str(path)))
            # LoadedMedia 가 오면 _on_media_status_changed 가 seek 를 적용한다.
            return
        if self._player_ready():
            self._apply_seek()
        # 아직 로딩 중이면 상태 신호가 왔을 때 적용한다.

    def _on_media_status_changed(self, status: Any) -> None:
        if self.player is None or QMediaPlayer is None:
            return
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._seek_pending = False
            self._append_detail_note(f"재생 오류: 영상을 열 수 없습니다 ({self._loaded_video})")
            return
        if not self._seek_pending:
            return
        ready = {
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        }
        if status in ready:
            self._apply_seek()

    def _on_media_error(self, error: Any, message: str = "") -> None:
        self._seek_pending = False
        self._append_detail_note(f"재생 오류: {message or error}")

    def _append_detail_note(self, note: str) -> None:
        current = self.detail.toPlainText()
        if note in current:
            return
        self.detail.setPlainText((current + "\n" if current else "") + note)

    def _play_current_video(self) -> None:
        if not self.video:
            return
        row = self._current_row()
        if not row:
            return
        path = self.resolver.resolve_video(row)
        if path is None:
            QMessageBox.information(
                self,
                "영상 없음",
                "영상 파일을 찾을 수 없습니다. VIDEO_ROOT 재매핑 설정을 확인하세요.",
            )
            return
        if self.player is None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
            return
        self._load_video(path, self._timestamp_sec(row), autoplay=True)
        if self.media_tabs is not None:
            self.media_tabs.setCurrentIndex(0)

    def _pause_video(self) -> None:
        if self.player is not None:
            self.player.pause()



# =============================================================================
# Image search page
# =============================================================================

class ImageSearchPage(QWidget):
    """이미지 DB 검색 화면.

    화면 흐름을 검색 파이프라인과 동일하게 분리한다.

    1) Crop 기반 검색
       - person: SigLIP2 + IRRA 1차 검색 -> SOLIDER 2차 재정렬
       - object: SigLIP2 + DINOv2

    2) 자연어 검색
       - person: SigLIP2 + IRRA
       - object: SigLIP2

    3) Qwen 검증 (선택)
       - Crop 기반 검색 결과 또는 자연어 검색 결과에 적용
       - 버튼 클릭 시 별도 Qwen subprocess로 실행
    """

    def __init__(self, config_path: str, resolver: Optional[PathResolver] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config_path = config_path
        self.resolver = resolver or PathResolver(ROOT)
        self.query_image: Optional[str] = None
        self.worker: Optional[SearchWorker] = None

        # 검색 결과는 crop/text를 분리해서 보관한다.
        self.last_search_result: Optional[Dict[str, Any]] = None
        self.last_search_results: Dict[str, Optional[Dict[str, Any]]] = {
            "crop": None,
            "text": None,
        }
        self.last_qwen_result: Optional[Dict[str, Any]] = None
        self._last_qwen_source: str = "text"
        self._last_qwen_top_k: int = 0
        self._last_backend_top_k: Dict[str, int] = {"crop": 0, "text": 0}
        self.model_options = load_search_model_options(config_path)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(8)

        # ------------------------------------------------------------------
        # 검색 방식은 한 콤보박스에 섞지 않고 단계별 탭으로 분리한다.
        # ------------------------------------------------------------------
        self.search_tabs = QTabWidget()
        self.search_tabs.setObjectName("searchStageTabs")

        # ====================== 1. 이미지 기반 검색 ======================
        image_page = QWidget()
        image_root = QVBoxLayout(image_page)
        image_root.setContentsMargins(8, 10, 8, 8)
        image_root.setSpacing(8)

        image_box = QGroupBox("1. Crop 기반 검색")
        image_grid = QGridLayout(image_box)
        image_grid.setHorizontalSpacing(12)
        image_grid.setVerticalSpacing(10)

        self.image_scope = QComboBox()
        self.image_scope.addItem("사람", "person")
        self.image_scope.addItem("객체", "object")
        self.image_scope.currentIndexChanged.connect(self._refill_image_models)
        image_grid.addWidget(QLabel("검색 대상"), 0, 0)
        image_grid.addWidget(self.image_scope, 0, 1)

        self.image_limit = QSpinBox()
        self.image_limit.setRange(1, 500)
        self.image_limit.setValue(20)
        self.image_limit.editingFinished.connect(lambda: self._sync_top_k_results("crop"))
        self.image_limit.valueChanged.connect(self._sync_qwen_top_k_limit)
        image_grid.addWidget(QLabel("Top-K"), 0, 2)
        image_grid.addWidget(self.image_limit, 0, 3)

        self.query_preview = QLabel("Crop을 선택하세요")
        self.query_preview.setObjectName("queryPreview")
        self.query_preview.setFixedSize(180, 150)
        self.query_preview.setAlignment(Qt.AlignCenter)
        image_grid.addWidget(self.query_preview, 1, 0, 2, 1)

        choose = QPushButton("Crop 선택")
        choose.setFixedHeight(68)
        choose.clicked.connect(self._choose_image)
        image_grid.addWidget(choose, 1, 1)

        self.image_pipeline = QLabel()
        self.image_pipeline.setObjectName("pipelineLabel")
        self.image_pipeline.setWordWrap(True)
        # Query preview(150px)와 같은 두 행을 공유하므로, 버튼은 키우고
        # 파이프라인 캡션은 글자가 잘리지 않는 선에서 낮게 고정한다.
        self.image_pipeline.setFixedHeight(62)
        image_grid.addWidget(self.image_pipeline, 2, 1, 1, 3)

        # 검색 모델 선택: 1차 후보 retriever(단일/RRF 조합) + 2차 재정렬 retriever. 후보는 pipeline.yaml.
        self.image_stage1 = QComboBox()
        self.image_stage1.setToolTip("1차 후보를 뽑는 임베더. pipeline.yaml 의 retrievers 중 선택 (여러 개면 RRF 조합)")
        self.image_stage1.currentIndexChanged.connect(self._sync_image_pipeline)
        self.image_rerank = QComboBox()
        self.image_rerank.setToolTip("1차 후보를 다시 정렬할 임베더. '없음' 이면 1차 순위 그대로")
        self.image_rerank.currentIndexChanged.connect(self._sync_image_pipeline)
        image_grid.addWidget(QLabel("1차 검색 모델"), 3, 0)
        image_grid.addWidget(self.image_stage1, 3, 1)
        image_grid.addWidget(QLabel("2차 재정렬"), 3, 2)
        image_grid.addWidget(self.image_rerank, 3, 3)

        self.image_search_btn = QPushButton("이미지 검색 실행")
        self.image_search_btn.setObjectName("primaryButton")
        self.image_search_btn.clicked.connect(self._search_image)
        image_grid.addWidget(self.image_search_btn, 4, 0, 1, 4)

        image_root.addWidget(image_box)
        image_root.addStretch(1)
        self.search_tabs.addTab(image_page, "1  Crop 기반 검색")

        # ====================== 2. 자연어 검색 ==========================
        text_page = QWidget()
        text_root = QVBoxLayout(text_page)
        text_root.setContentsMargins(8, 10, 8, 8)
        text_root.setSpacing(8)

        text_box = QGroupBox("2. 자연어 검색")
        text_grid = QGridLayout(text_box)
        text_grid.setHorizontalSpacing(12)
        text_grid.setVerticalSpacing(10)

        self.text_scope = QComboBox()
        self.text_scope.addItem("사람", "person")
        self.text_scope.addItem("객체", "object")
        self.text_scope.currentIndexChanged.connect(self._refill_text_models)
        text_grid.addWidget(QLabel("검색 대상"), 0, 0)
        text_grid.addWidget(self.text_scope, 0, 1)

        self.text_limit = QSpinBox()
        self.text_limit.setRange(1, 500)
        self.text_limit.setValue(20)
        self.text_limit.editingFinished.connect(lambda: self._sync_top_k_results("text"))
        self.text_limit.valueChanged.connect(self._sync_qwen_top_k_limit)
        text_grid.addWidget(QLabel("Top-K"), 0, 2)
        text_grid.addWidget(self.text_limit, 0, 3)

        self.text = QLineEdit()
        self.text.setPlaceholderText("예: 검은 상의를 입은 사람 / 검은 가방")
        text_grid.addWidget(QLabel("자연어 Query"), 1, 0)
        text_grid.addWidget(self.text, 1, 1, 1, 3)

        self.text_pipeline = QLabel()
        self.text_pipeline.setObjectName("pipelineLabel")
        self.text_pipeline.setWordWrap(True)
        text_grid.addWidget(self.text_pipeline, 2, 0, 1, 4)

        # 자연어 검색 모델: supports_text 인 retriever 단일 또는 RRF 조합.
        self.text_vectors = QComboBox()
        self.text_vectors.setToolTip("자연어를 임베딩할 모델 (pipeline.yaml 에서 supports_text=true 인 것). 여러 개면 RRF 조합")
        self.text_vectors.currentIndexChanged.connect(self._sync_text_pipeline)
        text_grid.addWidget(QLabel("검색 모델"), 3, 0)
        text_grid.addWidget(self.text_vectors, 3, 1, 1, 3)

        self.text_search_btn = QPushButton("자연어 검색 실행")
        self.text_search_btn.setObjectName("primaryButton")
        self.text_search_btn.clicked.connect(self._search_text)
        text_grid.addWidget(self.text_search_btn, 4, 0, 1, 4)

        text_root.addWidget(text_box)
        text_root.addStretch(1)
        self.search_tabs.addTab(text_page, "2  자연어 검색")

        # ====================== 3. Qwen 검증 ============================
        qwen_page = QWidget()
        qwen_root = QVBoxLayout(qwen_page)
        qwen_root.setContentsMargins(8, 10, 8, 8)
        qwen_root.setSpacing(8)

        qwen_box = QGroupBox("3. Qwen 검증  ·  Crop/자연어 검색 결과 후처리")
        qwen_grid = QGridLayout(qwen_box)
        qwen_grid.setHorizontalSpacing(12)
        qwen_grid.setVerticalSpacing(8)

        qwen_note = QLabel(
            "1. Crop 기반 검색 결과 또는 2. 자연어 검색 결과가 나온 뒤 선택적으로 실행합니다. "
            "Crop 기반 결과는 qwen_crop_stage.py, 자연어 결과는 qwen_stage.py를 사용하며, "
            "버튼을 눌렀을 때만 별도 프로세스로 실행됩니다."
        )
        qwen_note.setWordWrap(True)
        qwen_note.setObjectName("subtleLabel")
        qwen_grid.addWidget(qwen_note, 0, 0, 1, 4)

        self.qwen_source = QComboBox()
        self.qwen_source.addItem("Crop 기반 검색 결과", "crop")
        self.qwen_source.addItem("자연어 검색 결과", "text")
        self.qwen_source.currentIndexChanged.connect(self._sync_qwen_top_k_limit)
        qwen_grid.addWidget(QLabel("검증 대상"), 1, 0)
        qwen_grid.addWidget(self.qwen_source, 1, 1)

        self.qwen_top_k = QSpinBox()
        self.qwen_top_k.setRange(1, 200)
        self.qwen_top_k.setValue(20)
        qwen_grid.addWidget(QLabel("검증 후보 Top-K"), 1, 2)
        qwen_grid.addWidget(self.qwen_top_k, 1, 3)

        self.qwen_btn = QPushButton("Qwen 검증 실행")
        self.qwen_btn.clicked.connect(self._run_qwen)
        self.qwen_btn.setEnabled(True)
        qwen_grid.addWidget(self.qwen_btn, 2, 0, 1, 4)

        qwen_root.addWidget(qwen_box)
        qwen_root.addStretch(1)
        self.search_tabs.addTab(qwen_page, "3  Qwen 검증")

        root.addWidget(self.search_tabs)

        status_row = QHBoxLayout()
        self.status = QLabel("준비됨")
        self.status.setObjectName("statusLabel")
        status_row.addWidget(self.status, 1)
        # 검색 worker 가 돌 때만 보이는 진행 표시. 범위 (0,0) = 진행 중(indeterminate).
        self.progress = QProgressBar()
        self.progress.setObjectName("busyBar")
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(160)
        self.progress.setVisible(False)
        status_row.addWidget(self.progress)
        root.addLayout(status_row)

        self.results = ResultsPanel(video=False, resolver=self.resolver)
        root.addWidget(self.results, 1)

        self._sync_qwen_top_k_limit()
        self._refill_image_models()
        self._refill_text_models()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _refill_image_models(self, *_args: Any) -> None:
        """검색 대상(scope)이 바뀌면 그 scope 의 retriever 로 1차/2차 콤보를 다시 채운다."""
        scope = str(self.image_scope.currentData() or "person")
        names = list(self.model_options.get(f"{scope}_image") or FALLBACK_MODEL_OPTIONS[f"{scope}_image"])
        choices = stage1_choices(scope, names)
        preset = [n for n in DEFAULT_STAGE1.get(scope, []) if n in names]
        fill_combo(self.image_stage1, [(combo_label(c), c) for c in choices], preset if preset in choices else choices[-1])
        rerank_default = DEFAULT_RERANK.get(scope)
        fill_combo(self.image_rerank, [("없음", None)] + [(model_label(n), n) for n in names],
                   rerank_default if rerank_default in names else None)
        self._sync_image_pipeline()

    def _refill_text_models(self, *_args: Any) -> None:
        scope = str(self.text_scope.currentData() or "person")
        names = list(self.model_options.get(f"{scope}_text") or FALLBACK_MODEL_OPTIONS[f"{scope}_text"])
        items = [(model_label(n), [n]) for n in names]
        if len(names) > 1:
            items.append((combo_label(names), list(names)))
        fill_combo(self.text_vectors, items, list(names))
        self._sync_text_pipeline()

    def image_model_selection(self) -> Tuple[List[str], Optional[str]]:
        """(1차 retriever 목록, 2차 재정렬 retriever 또는 None)."""
        stage1 = list(self.image_stage1.currentData() or [])
        rerank = self.image_rerank.currentData()
        return stage1, (str(rerank) if rerank else None)

    def text_model_selection(self) -> List[str]:
        return list(self.text_vectors.currentData() or [])

    def _sync_image_pipeline(self, *_args: Any) -> None:
        if not hasattr(self, "image_stage1"):
            return
        scope = self.image_scope.currentData()
        stage1, rerank = self.image_model_selection()
        who = "사람" if scope == "person" else "객체"
        rerank_text = f"2차 재정렬: {model_label(rerank)}" if rerank else "재정렬 없음"
        self.image_pipeline.setText(
            f"{who} 검색  ·  1차 후보 검색: {combo_label(stage1) or '—'}  →  {rerank_text}  →  최종 Top-K"
        )

    def _sync_text_pipeline(self, *_args: Any) -> None:
        if not hasattr(self, "text_vectors"):
            return
        scope = self.text_scope.currentData()
        who = "사람" if scope == "person" else "객체"
        self.text_pipeline.setText(
            f"{who} 자연어 검색  ·  {combo_label(self.text_model_selection()) or '—'}  →  검색 결과  →  "
            "필요 시 Qwen 재검증"
        )

    def _sync_qwen_top_k_limit(self, *_args: Any) -> None:
        if not hasattr(self, "qwen_top_k"):
            return
        source = self.qwen_source.currentData() if hasattr(self, "qwen_source") else "text"
        base_limit = self.image_limit.value() if source == "crop" else self.text_limit.value()
        limit = max(1, int(base_limit))
        self.qwen_top_k.setMaximum(limit)
        if self.qwen_top_k.value() > limit:
            self.qwen_top_k.setValue(limit)

    def _sync_top_k_results(self, mode: str) -> None:
        """현재 검색 결과와 해당 모드의 Top-K를 동기화한다."""
        if self.worker is not None and self.worker.isRunning():
            return

        if mode == "text":
            self._sync_qwen_top_k_limit()
            top_k = int(self.text_limit.value())
        else:
            top_k = int(self.image_limit.value())

        result = self.last_search_results.get(mode)
        if result is None:
            return

        rows = list(result.get("results") or [])
        if top_k > self._last_backend_top_k.get(mode, 0):
            self.status.setText(
                f"Top-K를 {top_k}로 변경했습니다 · 반영하려면 검색 실행 버튼을 누르세요."
            )
            return

        self.last_qwen_result = None
        visible = rows_for_top_k(rows, top_k)
        self.results.set_rows(visible)
        self.status.setText(f"Top-K 동기화 · {len(visible)}건 표시")

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Query crop 선택",
            str(ROOT),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*.*)",
        )
        if not path:
            return
        self.query_image = path
        set_preview(self.query_preview, path)

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.image_search_btn,
            self.text_search_btn,
            self.image_scope,
            self.text_scope,
            self.image_stage1,
            self.image_rerank,
            self.text_vectors,
            self.image_limit,
            self.text_limit,
            self.qwen_top_k,
            self.qwen_btn,
        ):
            widget.setEnabled(not busy)

        self.status.setText(
            "검색 버튼 실행됨 · 모델 로드 / 임베딩 / Qdrant 검색을 수행합니다."
            if busy else "준비됨"
        )
        self.progress.setVisible(busy)

    # ------------------------------------------------------------------
    # 1. 이미지 기반 검색
    # ------------------------------------------------------------------
    def _search_image(self) -> None:
        if not self.query_image:
            QMessageBox.warning(self, "Query 필요", "검색할 crop 이미지를 선택하세요.")
            return

        scope = self.image_scope.currentData()
        limit = int(self.image_limit.value())
        solider_pool = max(SOLIDER_POOL_DEFAULT, limit)
        stage1, rerank = self.image_model_selection()
        # 백엔드에서 None 은 "운영 기본" 이므로, 사용자가 고른 '없음' 은 명시적으로 "none" 으로 보낸다.
        rerank_arg = rerank if rerank else "none"

        def job() -> Dict[str, Any]:
            backend = get_search_backend()
            searcher = backend.CropGeneralSearcher(self.config_path)
            try:
                res = searcher.search(
                    self.query_image or "",
                    scope=scope,
                    limit=limit,
                    solider_pool=solider_pool,
                    stage1=stage1 or None,
                    rerank=rerank_arg,
                )
                res = dict(res)
                rows = [backend.hit_to_dict(h) for h in res.pop("hits")]
                res["results"] = rows_for_top_k(rows, limit)
                res["mode"] = "crop"
                res["_gui_requested_top_k"] = limit
                return res
            finally:
                searcher.release()

        self._start_search(job)

    # ------------------------------------------------------------------
    # 2. 자연어 검색
    # ------------------------------------------------------------------
    def _search_text(self) -> None:
        text = self.text.text().strip()
        if not text:
            QMessageBox.warning(self, "Query 필요", "자연어 검색어를 입력하세요.")
            return

        scope = self.text_scope.currentData()
        limit = int(self.text_limit.value())
        vectors = self.text_model_selection()

        def job() -> Dict[str, Any]:
            backend = get_search_backend()
            searcher = backend.TextGeneralSearcher(
                self.config_path,
                translate_backend="opus",
                translate_model_id=None,
                expand=False,
            )
            try:
                res = searcher.search(
                    text,
                    scope=scope,
                    limit=limit,
                    translate=True,
                    vectors=vectors or None,
                )
                res = dict(res)
                rows = [backend.hit_to_dict(h) for h in res.pop("hits")]
                res["results"] = rows_for_top_k(rows, limit)
                res["mode"] = "text"
                res["_gui_requested_top_k"] = limit
                return res
            finally:
                searcher.release()

        self._start_search(job)

    def _start_search(self, job: Callable[[], Dict[str, Any]]) -> None:
        self.results.clear()
        self.last_qwen_result = None
        self._set_busy(True)
        self.worker = SearchWorker(job, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.succeeded.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _done(self, res: Dict[str, Any]) -> None:
        mode = str(res.get("mode") or "")
        if mode not in {"crop", "text"}:
            mode = "text" if res.get("query") else "crop"
            res["mode"] = mode

        requested_top_k = int(
            res.get("_gui_requested_top_k")
            or (self.text_limit.value() if mode == "text" else self.image_limit.value())
        )
        self._last_backend_top_k[mode] = requested_top_k

        res = dict(res)
        rows = rows_for_top_k(list(res.get("results") or []), requested_top_k)
        res["results"] = rows
        self.last_search_result = copy.deepcopy(res)
        self.last_search_results[mode] = copy.deepcopy(res)
        self.last_qwen_result = None
        self.results.set_rows(rows)

        timing = res.get("timing") or {}
        timing_text = " / ".join(f"{k}={float(v):.2f}s" for k, v in timing.items())
        translated = ""
        if res.get("translated"):
            translated = f" · 번역: {res.get('query_en')}"

        label = "자연어 검색" if mode == "text" else "이미지 검색"
        collection = res.get("collection") or ""
        suffix = f" · {timing_text}" if timing_text else ""
        models = f" · 모델 {res.get('pipeline')}" if res.get("pipeline") else ""
        self.status.setText(
            f"{label} 완료 · {collection} · {len(rows)}건{models}{suffix}{translated}"
        )

    # ------------------------------------------------------------------
    # 3. Qwen 별도 후처리
    # ------------------------------------------------------------------
    def _qwen_payload(self, source: str) -> Dict[str, Any]:
        base = self.last_search_results.get(source)
        if base is None:
            raise RuntimeError("Qwen에 넘길 검색 결과가 없습니다.")

        rows = copy.deepcopy(list(base.get("results") or []))
        for i, row in enumerate(rows, 1):
            resolved_crop = self.resolver.resolve_crop(row)
            if resolved_crop is not None:
                row["crop_path"] = str(resolved_crop)
                payload = row.get("payload")
                if isinstance(payload, dict):
                    payload["crop_path"] = str(resolved_crop)
            row.setdefault("pre_qwen_rank", int(row.get("rank") or i))
            base_score = row.get("score")
            if base_score is None:
                base_score = row.get("retrieval_score", 0.0)
            row.setdefault("pre_qwen_score", float(base_score or 0.0))
            row.setdefault(
                "qdrant_score",
                float(row.get("retrieval_score", base_score) or 0.0),
            )

        if source == "crop":
            if not self.query_image:
                raise RuntimeError("Qwen crop 검증에 사용할 query crop이 없습니다.")
            query_path = str(Path(self.query_image).resolve())
            return {
                "search_type": "crop",
                "query_image": query_path,
                "qwen": False,
                "crops": [
                    {
                        "crop_index": 1,
                        "kind": "crop",
                        "query_image": query_path,
                        "scope": base.get("scope"),
                        "collection": base.get("collection"),
                        "results": rows,
                    }
                ],
            }

        query_original = str(base.get("query") or self.text.text()).strip()
        query_en = str(base.get("query_en") or query_original).strip()
        return {
            "search_type": "text",
            "query": query_original,
            "query_en": query_en,
            "qwen": False,
            "crops": [
                {
                    "crop_index": 1,
                    "kind": "text",
                    "query_text_original": query_original,
                    "query_text": query_en,
                    "scope": base.get("scope"),
                    "collection": base.get("collection"),
                    "results": rows,
                }
            ],
        }

    def _run_qwen(self) -> None:
        # 버튼 클릭 전에는 Qwen subprocess를 절대 실행하지 않는다.
        source = str(self.qwen_source.currentData() or "text")
        base = self.last_search_results.get(source)
        if not base:
            label = "1. Crop 기반 검색" if source == "crop" else "2. 자연어 검색"
            QMessageBox.warning(
                self,
                "검색 결과 필요",
                f"먼저 {label}을 실행한 뒤 Qwen 검증을 눌러주세요.",
            )
            return
        if not base.get("results"):
            QMessageBox.warning(self, "검색 결과 없음", "Qwen으로 검증할 후보가 없습니다.")
            return

        payload = self._qwen_payload(source)
        top_k = min(self.qwen_top_k.value(), len(payload["crops"][0]["results"]))
        script = QWEN_CROP_SCRIPT if source == "crop" else QWEN_TEXT_SCRIPT
        self._last_qwen_source = source
        self._last_qwen_top_k = top_k

        def job() -> Dict[str, Any]:
            out = run_qwen_subprocess(
                payload,
                top_k=top_k,
                alpha=QWEN_ALPHA_DEFAULT,
                threshold=QWEN_THRESHOLD_DEFAULT,
                verify_mode=QWEN_VERIFY_MODE_DEFAULT,
                script_path=script,
            )
            crops = out.get("crops") or []
            rows = list(crops[0].get("results") or []) if crops else []
            return {"qwen_payload": out, "results": rows}

        self._set_busy(True)
        label = "Crop 기반" if source == "crop" else "자연어"
        self.status.setText(
            f"Qwen 검증 시작 · {label} 검색 결과 후처리 실행 중..."
        )
        self.worker = SearchWorker(job, self)
        self.worker.succeeded.connect(self._qwen_done)
        self.worker.failed.connect(self._qwen_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _qwen_done(self, res: Dict[str, Any]) -> None:
        rows = list(res.get("results") or [])
        qwen_payload = res.get("qwen_payload") or {}
        self.last_qwen_result = copy.deepcopy(qwen_payload)

        visible = rows_for_top_k(rows, self._last_qwen_top_k or len(rows))
        self.results.set_rows(visible)
        shift = qwen_payload.get("qwen_rank_shift")
        shift_text = ""
        if isinstance(shift, dict):
            shift_text = (
                f" · 상승 {shift.get('moved_up', 0)} / "
                f"하락 {shift.get('moved_down', 0)} / 유지 {shift.get('unchanged', 0)}"
            )
        label = "Crop 기반" if self._last_qwen_source == "crop" else "자연어"
        self.status.setText(
            f"Qwen 검증 완료 · {label} 결과 {qwen_payload.get('qwen_scored_candidates', 0)}건"
            f" · {qwen_payload.get('qwen_elapsed_sec', 0)}s{shift_text}"
            + qwen_reranker_note(qwen_payload)
        )

    def _qwen_failed(self, detail: str) -> None:
        self.status.setText("Qwen 재검증 실패")
        QMessageBox.critical(self, "Qwen 재검증 실패", detail)

    def _failed(self, detail: str) -> None:
        self.status.setText("검색 실패")
        QMessageBox.critical(self, "검색 실패", detail)


# =============================================================================
# Video search page
# =============================================================================

class VideoSearchPage(QWidget):
    """영상 DB 검색 화면.

    화면 흐름을 이미지 탭과 같은 방식으로 단순화한다.

    1) Crop 기반 영상 검색
       - person: SOLIDER (내부 기본값)
       - object: DINOv2 (내부 기본값)
       - media_type=video, group_by=track_key

    2) 자연어 영상 검색
       - person: SigLIP2 + IRRA -> track_key 그룹별 RRF
       - object: SigLIP2 -> track_key 그룹 검색
       - 검색 완료 후 사용자가 Qwen 버튼을 누르면 대표 Crop 기준 별도 재검증

    Qwen은 검색과 자동 연동하지 않는다.
    group_size / candidate_k / vector 선택은 운영 내부값으로 유지하고 GUI에는 노출하지 않는다.
    """

    def __init__(self, config_path: str, resolver: Optional[PathResolver] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config_path = config_path
        self.resolver = resolver or PathResolver(ROOT)
        self.query_image: Optional[str] = None
        self.worker: Optional[SearchWorker] = None

        self.last_search_result: Dict[str, Optional[Dict[str, Any]]] = {
            "image-video": None,
            "text-video": None,
        }
            # 영상 Qwen 후처리 결과는 기본 검색 결과와 별도 보관한다.
        self.last_qwen_result: Optional[Dict[str, Any]] = None
        self._last_qwen_source: str = "text-video"
        self._last_qwen_top_k: int = 0

        self._last_backend_top_k: Dict[str, int] = {
            "image-video": 0,
            "text-video": 0,
        }
        self.model_options = load_search_model_options(config_path)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(8)

        self.search_tabs = QTabWidget()
        self.search_tabs.setObjectName("searchStageTabs")

        # ====================== 1. Crop 기반 영상 검색 ======================
        image_page = QWidget()
        image_root = QVBoxLayout(image_page)
        image_root.setContentsMargins(8, 10, 8, 8)
        image_root.setSpacing(8)

        image_box = QGroupBox("1. Crop 기반 검색")
        image_grid = QGridLayout(image_box)
        image_grid.setHorizontalSpacing(12)
        image_grid.setVerticalSpacing(10)

        self.image_scope = QComboBox()
        self.image_scope.addItem("사람", "person")
        self.image_scope.addItem("객체", "object")
        self.image_scope.currentIndexChanged.connect(self._refill_image_models)
        image_grid.addWidget(QLabel("검색 대상"), 0, 0)
        image_grid.addWidget(self.image_scope, 0, 1)

        self.image_top_k = QSpinBox()
        self.image_top_k.setRange(1, 200)
        self.image_top_k.setValue(20)
        self.image_top_k.editingFinished.connect(lambda: self._sync_top_k_results("image-video"))
        self.image_top_k.valueChanged.connect(self._sync_video_qwen_top_k_limit)
        image_grid.addWidget(QLabel("Top-K"), 0, 2)
        image_grid.addWidget(self.image_top_k, 0, 3)

        # 영상당 표시 상한. 같은 영상의 여러 track 은 대개 서로 다른 사람이므로 합치지 않고 개수만 제한한다.
        self.image_per_video = QSpinBox()
        self.image_per_video.setRange(0, 50)
        self.image_per_video.setValue(0)
        self.image_per_video.setSpecialValueText("제한 없음")
        self.image_per_video.setToolTip(
            "같은 영상에서 최대 몇 개의 Track 까지 보일지. 0 = 제한 없음. "
            "접힌 결과는 지워지는 것이 아니라 카드에 '같은 영상 N건 더 있음' 으로 표시됩니다."
        )
        self.image_per_video.valueChanged.connect(lambda: self._note_per_video_change("image-video"))
        image_grid.addWidget(QLabel("영상당 최대"), 0, 4)
        image_grid.addWidget(self.image_per_video, 0, 5)

        self.query_preview = QLabel("Crop을 선택하세요")
        self.query_preview.setObjectName("queryPreview")
        self.query_preview.setFixedSize(180, 150)
        self.query_preview.setAlignment(Qt.AlignCenter)
        image_grid.addWidget(self.query_preview, 1, 0, 2, 1)

        choose = QPushButton("Crop 선택")
        choose.setFixedHeight(68)
        choose.clicked.connect(self._choose_image)
        image_grid.addWidget(choose, 1, 1)

        self.image_pipeline = QLabel()
        self.image_pipeline.setObjectName("pipelineLabel")
        self.image_pipeline.setWordWrap(True)
        # Query preview(150px)와 같은 두 행을 공유하므로, 버튼은 키우고
        # 파이프라인 캡션은 글자가 잘리지 않는 선에서 낮게 고정한다.
        self.image_pipeline.setFixedHeight(62)
        image_grid.addWidget(self.image_pipeline, 2, 1, 1, 5)

        # 영상 identity 검색에 쓸 named vector (pipeline.yaml 의 해당 scope retriever 중 선택)
        self.image_vector = QComboBox()
        self.image_vector.setToolTip("Crop 을 임베딩해 track 을 찾을 모델. 기본 사람 SOLIDER / 객체 DINOv2")
        self.image_vector.currentIndexChanged.connect(self._sync_image_pipeline)
        image_grid.addWidget(QLabel("검색 모델"), 3, 0)
        image_grid.addWidget(self.image_vector, 3, 1)

        self.image_search_btn = QPushButton("영상 검색 실행")
        self.image_search_btn.setObjectName("primaryButton")
        self.image_search_btn.clicked.connect(self._search_image_video)
        image_grid.addWidget(self.image_search_btn, 4, 0, 1, 6)

        image_root.addWidget(image_box)
        image_root.addStretch(1)
        self.search_tabs.addTab(image_page, "1  Crop 기반 검색")

        # ====================== 2. 자연어 영상 검색 ==========================
        text_page = QWidget()
        text_root = QVBoxLayout(text_page)
        text_root.setContentsMargins(8, 10, 8, 8)
        text_root.setSpacing(8)

        text_box = QGroupBox("2. 자연어 검색")
        text_grid = QGridLayout(text_box)
        text_grid.setHorizontalSpacing(12)
        text_grid.setVerticalSpacing(10)

        self.text_scope = QComboBox()
        self.text_scope.addItem("사람", "person")
        self.text_scope.addItem("객체", "object")
        self.text_scope.currentIndexChanged.connect(self._refill_text_models)
        text_grid.addWidget(QLabel("검색 대상"), 0, 0)
        text_grid.addWidget(self.text_scope, 0, 1)

        self.text_top_k = QSpinBox()
        self.text_top_k.setRange(1, 200)
        self.text_top_k.setValue(20)
        self.text_top_k.editingFinished.connect(lambda: self._sync_top_k_results("text-video"))
        self.text_top_k.valueChanged.connect(self._sync_video_qwen_top_k_limit)
        text_grid.addWidget(QLabel("Top-K"), 0, 2)
        text_grid.addWidget(self.text_top_k, 0, 3)

        self.text_per_video = QSpinBox()
        self.text_per_video.setRange(0, 50)
        self.text_per_video.setValue(0)
        self.text_per_video.setSpecialValueText("제한 없음")
        self.text_per_video.setToolTip(
            "같은 영상에서 최대 몇 개의 Track 까지 보일지. 0 = 제한 없음. "
            "접힌 결과는 지워지는 것이 아니라 카드에 '같은 영상 N건 더 있음' 으로 표시됩니다."
        )
        self.text_per_video.valueChanged.connect(lambda: self._note_per_video_change("text-video"))
        text_grid.addWidget(QLabel("영상당 최대"), 0, 4)
        text_grid.addWidget(self.text_per_video, 0, 5)

        self.text = QLineEdit()
        self.text.setPlaceholderText("예: 검은 상의를 입은 사람 / 검은 가방")
        text_grid.addWidget(QLabel("자연어 Query"), 1, 0)
        text_grid.addWidget(self.text, 1, 1, 1, 5)

        self.text_pipeline = QLabel()
        self.text_pipeline.setObjectName("pipelineLabel")
        self.text_pipeline.setWordWrap(True)
        text_grid.addWidget(self.text_pipeline, 2, 0, 1, 6)

        self.text_vectors = QComboBox()
        self.text_vectors.setToolTip("자연어를 임베딩할 모델 (supports_text). 여러 개면 track 그룹별 RRF 조합")
        self.text_vectors.currentIndexChanged.connect(self._sync_text_pipeline)
        text_grid.addWidget(QLabel("검색 모델"), 3, 0)
        text_grid.addWidget(self.text_vectors, 3, 1, 1, 3)

        self.text_search_btn = QPushButton("자연어 영상 검색 실행")
        self.text_search_btn.setObjectName("primaryButton")
        self.text_search_btn.clicked.connect(self._search_text_video)
        text_grid.addWidget(self.text_search_btn, 4, 0, 1, 6)

        text_root.addWidget(text_box)
        text_root.addStretch(1)
        self.search_tabs.addTab(text_page, "2  자연어 검색")

        # ====================== 3. Qwen 검증 ============================
        video_qwen_page = QWidget()
        video_qwen_root = QVBoxLayout(video_qwen_page)
        video_qwen_root.setContentsMargins(8, 10, 8, 8)
        video_qwen_root.setSpacing(8)

        video_qwen_box = QGroupBox("3. Qwen 검증  ·  Crop/자연어 영상 검색 결과 후처리")
        video_qwen_grid = QGridLayout(video_qwen_box)
        video_qwen_grid.setHorizontalSpacing(12)
        video_qwen_grid.setVerticalSpacing(8)

        video_qwen_note = QLabel(
            "1. Crop 기반 검색 결과 또는 2. 자연어 검색 결과가 나온 뒤 선택적으로 실행합니다. "
            "각 Track의 대표 Crop을 기준으로 검증하며, "
            "Crop 기반 결과는 qwen_crop_stage.py, 자연어 결과는 qwen_stage.py를 사용합니다."
        )
        video_qwen_note.setWordWrap(True)
        video_qwen_note.setObjectName("subtleLabel")
        video_qwen_grid.addWidget(video_qwen_note, 0, 0, 1, 4)

        self.video_qwen_source = QComboBox()
        self.video_qwen_source.addItem("Crop 기반 검색 결과", "image-video")
        self.video_qwen_source.addItem("자연어 검색 결과", "text-video")
        self.video_qwen_source.currentIndexChanged.connect(self._sync_video_qwen_top_k_limit)
        video_qwen_grid.addWidget(QLabel("검증 대상"), 1, 0)
        video_qwen_grid.addWidget(self.video_qwen_source, 1, 1)

        self.video_qwen_top_k = QSpinBox()
        self.video_qwen_top_k.setRange(1, 200)
        self.video_qwen_top_k.setValue(10)
        video_qwen_grid.addWidget(QLabel("검증 후보 Top-K"), 1, 2)
        video_qwen_grid.addWidget(self.video_qwen_top_k, 1, 3)

        self.video_qwen_btn = QPushButton("Qwen 검증 실행")
        self.video_qwen_btn.clicked.connect(self._run_video_qwen)
        self.video_qwen_btn.setEnabled(True)
        video_qwen_grid.addWidget(self.video_qwen_btn, 2, 0, 1, 4)

        video_qwen_root.addWidget(video_qwen_box)
        video_qwen_root.addStretch(1)
        self.search_tabs.addTab(video_qwen_page, "3  Qwen 검증")

        root.addWidget(self.search_tabs)

        status_row = QHBoxLayout()
        self.status = QLabel("준비됨")
        self.status.setObjectName("statusLabel")
        status_row.addWidget(self.status, 1)
        # 검색 worker 가 돌 때만 보이는 진행 표시. 범위 (0,0) = 진행 중(indeterminate).
        self.progress = QProgressBar()
        self.progress.setObjectName("busyBar")
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(160)
        self.progress.setVisible(False)
        status_row.addWidget(self.progress)
        root.addLayout(status_row)

        self.results = ResultsPanel(video=True, resolver=self.resolver)
        root.addWidget(self.results, 1)

        self._refill_image_models()
        self._refill_text_models()
        self._sync_video_qwen_top_k_limit()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _refill_image_models(self, *_args: Any) -> None:
        scope = str(self.image_scope.currentData() or "person")
        names = list(self.model_options.get(f"{scope}_image") or FALLBACK_MODEL_OPTIONS[f"{scope}_image"])
        default = VIDEO_PERSON_VECTOR_DEFAULT if scope == "person" else VIDEO_OBJECT_VECTOR_DEFAULT
        fill_combo(self.image_vector, [(model_label(n), n) for n in names], default if default in names else names[0])
        self._sync_image_pipeline()

    def _refill_text_models(self, *_args: Any) -> None:
        scope = str(self.text_scope.currentData() or "person")
        names = list(self.model_options.get(f"{scope}_text") or FALLBACK_MODEL_OPTIONS[f"{scope}_text"])
        items = [(model_label(n), [n]) for n in names]
        if len(names) > 1:
            items.append((combo_label(names), list(names)))
        fill_combo(self.text_vectors, items, list(names))
        self._sync_text_pipeline()

    def image_model_selection(self) -> str:
        scope = str(self.image_scope.currentData() or "person")
        default = VIDEO_PERSON_VECTOR_DEFAULT if scope == "person" else VIDEO_OBJECT_VECTOR_DEFAULT
        return str(self.image_vector.currentData() or default)

    def text_model_selection(self) -> List[str]:
        return list(self.text_vectors.currentData() or [])

    def _sync_image_pipeline(self, *_args: Any) -> None:
        if not hasattr(self, "image_vector"):
            return
        scope = self.image_scope.currentData()
        who, what = ("사람", "인물") if scope == "person" else ("객체", "객체")
        self.image_pipeline.setText(
            f"{who} 영상 검색  ·  {model_label(self.image_model_selection())}  →  track_key 기준 동일 {what} Track 검색  →  최종 Top-K"
        )

    def _sync_text_pipeline(self, *_args: Any) -> None:
        if not hasattr(self, "text_vectors"):
            return
        scope = self.text_scope.currentData()
        who = "사람" if scope == "person" else "객체"
        vectors = self.text_model_selection()
        fusion = "  →  RRF" if len(vectors) > 1 else ""
        self.text_pipeline.setText(
            f"{who} 자연어 영상 검색  ·  {combo_label(vectors) or '—'}  →  track_key 그룹 검색{fusion}  →  최종 Top-K"
        )

    def _sync_video_qwen_top_k_limit(self, *_args: Any) -> None:
        if not hasattr(self, "video_qwen_top_k"):
            return
        source = self.video_qwen_source.currentData() if hasattr(self, "video_qwen_source") else "text-video"
        base_limit = self.image_top_k.value() if source == "image-video" else self.text_top_k.value()
        limit = max(1, int(base_limit))
        self.video_qwen_top_k.setMaximum(limit)
        if self.video_qwen_top_k.value() > limit:
            self.video_qwen_top_k.setValue(limit)

    def _note_per_video_change(self, mode: str) -> None:
        """영상당 상한은 backend 후보 선택 단계에서 적용되므로 바꾸면 재검색이 필요하다는 것을 알린다."""
        if not hasattr(self, "status") or self.last_search_result.get(mode) is None:
            return
        spin = self.image_per_video if mode == "image-video" else self.text_per_video
        value = int(spin.value())
        label = f"{value}개" if value > 0 else "제한 없음"
        self.status.setText(f"영상당 최대 표시를 '{label}' 로 변경했습니다 · 반영하려면 검색 실행 버튼을 누르세요.")

    def _sync_top_k_results(self, mode: str) -> None:
        """현재 결과 개수와 해당 영상 검색 모드의 Top-K를 동기화한다."""
        if self.worker is not None and self.worker.isRunning():
            return

        result = self.last_search_result.get(mode)
        if result is None:
            return

        top_k = int(
            self.image_top_k.value()
            if mode == "image-video"
            else self.text_top_k.value()
        )
        if mode == "text-video":
            self._sync_video_qwen_top_k_limit()

        rows = list(result.get("results") or [])

        if top_k > self._last_backend_top_k.get(mode, 0):
            self.status.setText(
                f"Top-K를 {top_k}그룹으로 변경했습니다 · 반영하려면 검색 실행 버튼을 누르세요."
            )
            return

        visible = rows_for_top_k(rows, top_k)
        self.results.set_rows(visible)
        self.status.setText(
            f"Top-K 동기화 · {len(visible)}그룹 표시"
            + (f" / 요청 {top_k}" if len(visible) != top_k else "")
        )

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Query crop 선택",
            str(ROOT),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*.*)",
        )
        if not path:
            return
        self.query_image = path
        set_preview(self.query_preview, path)

    def _set_busy(self, busy: bool) -> None:
        self.image_search_btn.setEnabled(not busy)
        self.text_search_btn.setEnabled(not busy)
        self.image_scope.setEnabled(not busy)
        self.text_scope.setEnabled(not busy)
        self.image_vector.setEnabled(not busy)
        self.text_vectors.setEnabled(not busy)
        self.image_top_k.setEnabled(not busy)
        self.text_top_k.setEnabled(not busy)
        self.video_qwen_top_k.setEnabled(not busy)
        self.video_qwen_btn.setEnabled(not busy)
        self.search_tabs.setEnabled(not busy)
        self.status.setText(
            "검색 버튼 실행됨 · 모델 로드/임베딩/Qdrant 그룹 검색을 수행합니다."
            if busy
            else "준비됨"
        )
        self.progress.setVisible(busy)

    # ------------------------------------------------------------------
    # 1. Crop -> video
    # ------------------------------------------------------------------
    def _search_image_video(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.query_image:
            QMessageBox.warning(self, "Query 필요", "검색할 crop 이미지를 선택하세요.")
            return

        scope = self.image_scope.currentData()
        requested_top_k = int(self.image_top_k.value())
        per_video_max = int(self.image_per_video.value())
        vector = self.image_model_selection()

        args = argparse.Namespace(
            image=self.query_image,
            config=self.config_path,
            scope=scope,
            vector=vector,
            top_k=requested_top_k,
            group_size=VIDEO_GROUP_SIZE_DEFAULT,
            per_video_max=per_video_max,
        )

        def job() -> Dict[str, Any]:
            backend = get_search_backend()
            out = dict(backend.run_image_video(args))
            out["results"] = rows_for_top_k(
                list(out.get("results") or []),
                requested_top_k,
            )
            out["_gui_requested_top_k"] = requested_top_k
            return out

        self.results.clear()
        self.last_qwen_result = None
        self._set_busy(True)
        self.worker = SearchWorker(job, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.succeeded.connect(lambda res: self._done("image-video", res))
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    # ------------------------------------------------------------------
    # 2. Text -> video
    # ------------------------------------------------------------------
    def _search_text_video(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        text = self.text.text().strip()
        if not text:
            QMessageBox.warning(self, "Query 필요", "자연어 검색어를 입력하세요.")
            return

        scope = self.text_scope.currentData()
        requested_top_k = int(self.text_top_k.value())
        per_video_max = int(self.text_per_video.value())

        args = argparse.Namespace(
            text=text,
            config=self.config_path,
            scope=scope,
            top_k=requested_top_k,
            # 영상당 상한을 걸면 상위권에서 접히는 track 이 생기므로 후보를 더 받아 둔다.
            candidate_k=max(
                VIDEO_TEXT_CANDIDATE_K_DEFAULT,
                requested_top_k * (5 if per_video_max > 0 else 1),
            ),
            group_size=VIDEO_GROUP_SIZE_DEFAULT,
            per_video_max=per_video_max,
            vectors=self.text_model_selection() or None,
            translate_backend="opus",
            translate_model_id=None,
            no_translate=False,
            expand=False,
        )

        def job() -> Dict[str, Any]:
            backend = get_search_backend()
            out = dict(backend.run_text_video(args))
            out["results"] = rows_for_top_k(
                list(out.get("results") or []),
                requested_top_k,
            )
            out["_gui_requested_top_k"] = requested_top_k
            return out

        self.results.clear()
        self.last_qwen_result = None
        self._set_busy(True)
        self.worker = SearchWorker(job, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.succeeded.connect(lambda res: self._done("text-video", res))
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _done(self, mode: str, res: Dict[str, Any]) -> None:
        requested_top_k = int(
            res.get("_gui_requested_top_k")
            or (
                self.image_top_k.value()
                if mode == "image-video"
                else self.text_top_k.value()
            )
        )
        self._last_backend_top_k[mode] = requested_top_k

        res = dict(res)
        rows = rows_for_top_k(list(res.get("results") or []), requested_top_k)
        res["results"] = rows
        self.last_search_result[mode] = copy.deepcopy(res)
        self.results.set_rows(rows)

        timing = res.get("timing") or {}
        timing_text = " / ".join(
            f"{k}={float(v):.2f}s" for k, v in timing.items()
        )
        translated = ""
        if res.get("translated"):
            translated = f" · 번역: {res.get('query_en')}"

        cap = int(res.get("per_video_max") or 0)
        hidden = res.get("per_video_hidden") or {}
        hidden_total = sum(int(v) for v in hidden.values()) if isinstance(hidden, dict) else 0
        cap_text = ""
        if cap > 0:
            cap_text = f" · 영상당 최대 {cap}"
            if hidden_total:
                cap_text += f" (같은 영상 {hidden_total}건 접힘)"
            if len(rows) < requested_top_k:
                cap_text += f" · 상한 때문에 {len(rows)}그룹만 표시"

        self.status.setText(
            f"완료 · {res.get('collection')} · {len(rows)}그룹"
            + (f" · 모델 {res.get('pipeline')}" if res.get("pipeline") else "")
            + cap_text
            + (f" · {timing_text}" if timing_text else "")
            + translated
        )

    # ------------------------------------------------------------------
    # Qwen 별도 후처리: 자연어 -> 영상 결과의 Track 대표 Crop 재검증
    # ------------------------------------------------------------------
    def _video_qwen_payload(self, source: str) -> Dict[str, Any]:
        base = self.last_search_result.get(source)
        if not base:
            raise RuntimeError("Qwen에 넘길 영상 검색 결과가 없습니다.")

        rows = copy.deepcopy(list(base.get("results") or []))
        for i, row in enumerate(rows, 1):
            resolved_crop = self.resolver.resolve_crop(row)
            if resolved_crop is not None:
                row["crop_path"] = str(resolved_crop)
                payload = row.get("payload")
                if isinstance(payload, dict):
                    payload["crop_path"] = str(resolved_crop)

            row.setdefault("pre_qwen_rank", int(row.get("rank") or i))
            base_score = row.get("score")
            if base_score is None:
                base_score = row.get("rrf_score", 0.0)
            row.setdefault("pre_qwen_score", float(base_score or 0.0))
            row.setdefault("qdrant_score", float(base_score or 0.0))

        if source == "image-video":
            if not self.query_image:
                raise RuntimeError("Qwen crop 검증에 사용할 query crop이 없습니다.")
            query_path = str(Path(self.query_image).resolve())
            return {
                "search_type": "crop",
                "query_image": query_path,
                "qwen": False,
                "media": "video",
                "crops": [
                    {
                        "crop_index": 1,
                        "kind": "crop",
                        "query_image": query_path,
                        "scope": base.get("scope"),
                        "collection": base.get("collection"),
                        "media": "video",
                        "results": rows,
                    }
                ],
            }

        query_original = str(base.get("query") or self.text.text()).strip()
        query_en = str(base.get("query_en") or query_original).strip()
        return {
            "search_type": "text",
            "query": query_original,
            "query_en": query_en,
            "qwen": False,
            "media": "video",
            "crops": [
                {
                    "crop_index": 1,
                    "kind": "text",
                    "query_text_original": query_original,
                    "query_text": query_en,
                    "scope": base.get("scope"),
                    "collection": base.get("collection"),
                    "media": "video",
                    "results": rows,
                }
            ],
        }

    def _run_video_qwen(self) -> None:
        # 사용자가 버튼을 눌렀을 때만 Qwen을 실행한다.
        source = str(self.video_qwen_source.currentData() or "text-video")
        base = self.last_search_result.get(source)
        if not base:
            label = "1. Crop 기반 검색" if source == "image-video" else "2. 자연어 검색"
            QMessageBox.warning(
                self,
                "영상 검색 결과 필요",
                f"먼저 {label}을 실행한 뒤 Qwen 검증을 눌러주세요.",
            )
            return
        if not base.get("results"):
            QMessageBox.warning(
                self,
                "검색 결과 없음",
                "Qwen으로 재검증할 영상 Track 후보가 없습니다.",
            )
            return

        payload = self._video_qwen_payload(source)
        top_k = min(
            self.video_qwen_top_k.value(),
            len(payload["crops"][0]["results"]),
        )
        script = QWEN_CROP_SCRIPT if source == "image-video" else QWEN_TEXT_SCRIPT
        self._last_qwen_source = source
        self._last_qwen_top_k = top_k

        def job() -> Dict[str, Any]:
            out = run_qwen_subprocess(
                payload,
                top_k=top_k,
                alpha=QWEN_ALPHA_DEFAULT,
                threshold=QWEN_THRESHOLD_DEFAULT,
                verify_mode=QWEN_VERIFY_MODE_DEFAULT,
                script_path=script,
            )
            crops = out.get("crops") or []
            rows = list(crops[0].get("results") or []) if crops else []
            return {"qwen_payload": out, "results": rows}

        self._set_busy(True)
        label = "Crop 기반" if source == "image-video" else "자연어"
        self.status.setText(
            f"영상 Qwen 검증 시작 · {label} 검색 결과 후처리 실행 중..."
        )
        self.worker = SearchWorker(job, self)
        self.worker.succeeded.connect(self._video_qwen_done)
        self.worker.failed.connect(self._video_qwen_failed)
        self.worker.finished.connect(lambda: self._set_busy(False))
        self.worker.start()

    def _video_qwen_done(self, res: Dict[str, Any]) -> None:
        rows = list(res.get("results") or [])
        qwen_payload = res.get("qwen_payload") or {}
        self.last_qwen_result = copy.deepcopy(qwen_payload)

        visible = rows_for_top_k(rows, self._last_qwen_top_k or len(rows))
        self.results.set_rows(visible)

        shift = qwen_payload.get("qwen_rank_shift")
        shift_text = ""
        if isinstance(shift, dict):
            shift_text = (
                f" · 상승 {shift.get('moved_up', 0)} / "
                f"하락 {shift.get('moved_down', 0)} / 유지 {shift.get('unchanged', 0)}"
            )

        label = "Crop 기반" if self._last_qwen_source == "image-video" else "자연어"
        self.status.setText(
            f"영상 Qwen 검증 완료 · {label} 결과 "
            f"{qwen_payload.get('qwen_scored_candidates', 0)}그룹"
            f" · {qwen_payload.get('qwen_elapsed_sec', 0)}s{shift_text}"
            + qwen_reranker_note(qwen_payload)
        )

    def _video_qwen_failed(self, detail: str) -> None:
        self.status.setText("영상 Qwen 재검증 실패")
        QMessageBox.critical(self, "영상 Qwen 재검증 실패", detail)

    def _failed(self, detail: str) -> None:
        self.status.setText("검색 실패")
        QMessageBox.critical(self, "검색 실패", detail)


# =============================================================================
# Main window
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self, config_path: Optional[str] = None, data_root: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Forensic Visual Retrieval · Pipeline & Search")
        self.resize(1320, 900)

        # 창 위치/크기 기억. 조직/앱 이름은 gui_theme.apply_theme 이 설정한다.
        self._settings = QSettings()
        geometry = self._settings.value("main/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        self.path_resolver = PathResolver(ROOT)

        # 검색 페이지가 읽는 pipeline yaml. --config 로 바꿀 수 있다 (기본 pipeline.yaml).
        # DB 를 다른 collection_prefix 로 구축했으면 같은 yaml 을 여기에도 넘겨야 검색이 그 컬렉션을 본다.
        self.config_path = Path(config_path).expanduser().resolve() if config_path else CONFIG_PATH
        if not self.config_path.is_file():
            raise FileNotFoundError(f"pipeline yaml 이 없습니다: {self.config_path}")
        if self.config_path != CONFIG_PATH:
            self.setWindowTitle(self.windowTitle() + f"  [{self.config_path.name}]")

        # crop/영상 파일 위치. 스크립트 폴더에 data 가 없으면(부분 복사본 실행) yaml 폴더·상위 폴더에서 찾는다.
        self.data_root = resolve_data_root(data_root, self.config_path)
        self.path_resolver.set_crop_root(self.data_root)
        self.path_resolver.set_video_root(self.data_root / "videos")
        if not self.data_root.is_dir():
            self.statusBar().showMessage(f"data 폴더를 찾지 못했습니다: {self.data_root} — 미리보기가 비어 보일 수 있습니다 (--data-root 로 지정)")
        elif self.data_root != (ROOT / "data").resolve():
            self.setWindowTitle(self.windowTitle() + f"  [data: {self.data_root}]")

        tabs = QTabWidget()

        # 전체 페이지는 2개만 둔다.
        # 각 페이지 내부에 1. Crop 기반 검색 / 2. 자연어 검색 / 3. Qwen 검증
        # 세부 탭을 동일하게 구성한다.
        image_page = ImageSearchPage(str(self.config_path), self.path_resolver)
        video_page = VideoSearchPage(str(self.config_path), self.path_resolver)

        tabs.addTab(image_page, "이미지 검색")
        tabs.addTab(video_page, "영상 검색")

        # ------------------------------------------------------------------
        # DB 구축 / 클러스터링 / 평가 탭.
        #
        # 단계 정의는 gui_pipelines.json 에 있고 pipeline_page.py 가 폼을
        # 자동 생성한다. 단계를 바꾸려면 JSON 만 고치면 된다.
        #
        # pipeline_page 를 못 불러와도 검색 GUI 는 그대로 떠야 하므로
        # import 실패를 삼키고 상태바로만 알린다.
        # ------------------------------------------------------------------
        pipeline_note = ""
        try:
            from gui.pipeline_page import PipelinePage, available_groups

            groups = available_groups()
            if groups:
                for g in groups:
                    tabs.addTab(PipelinePage(g["id"]), g["title"])
            else:
                pipeline_note = (
                    "  ·  gui_pipelines.json 을 읽지 못해 파이프라인 탭이 없습니다"
                )
        except Exception as exc:  # noqa: BLE001
            pipeline_note = f"  ·  파이프라인 탭 로드 실패: {exc}"

        # 벤치마크 탭 (P5): 원장 리더보드 · 채택 기준 색 · verify · 채택→yaml · 그래프.
        # 마찬가지로 실패해도 검색 GUI 는 떠야 한다.
        try:
            from gui.bench_page import BenchPage

            tabs.addTab(BenchPage(), "벤치마크")
        except Exception as exc:  # noqa: BLE001
            pipeline_note += f"  ·  벤치마크 탭 로드 실패: {exc}"

        self.setCentralWidget(tabs)

        self.statusBar().showMessage(
            "검색(이미지/영상) · 구축 · 클러스터링 · 평가"
            + pipeline_note
        )


    def closeEvent(self, event) -> None:  # noqa: N802 (Qt 시그니처)
        self._settings.setValue("main/geometry", self.saveGeometry())
        super().closeEvent(event)


def main() -> int:
    # Qt 자체 인자(-style 등)는 남겨 두고 우리 인자만 먼저 읽는다.
    parser = argparse.ArgumentParser(description="Forensic Visual Retrieval GUI")
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="검색 페이지가 읽을 pipeline yaml (기본: pipeline.yaml). "
             "예: --config pipeline_image.yaml",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="crop/영상 파일이 있는 data 폴더 (기본: 스크립트 폴더/data, 없으면 yaml 폴더·상위 폴더의 data 를 자동 탐색). "
             "환경변수 TRANSREID_DATA_ROOT 로도 지정 가능",
    )
    cli, qt_args = parser.parse_known_args(sys.argv[1:])

    app = QApplication([sys.argv[0]] + qt_args)
    apply_theme(app)

    # GUI 시작만으로는 unified_search_4mode / 임베더 / 모델을 로드하지 않는다.
    # 실제 검색 backend는 검색 실행 버튼을 누른 뒤 worker 안에서 lazy import된다.
    window = MainWindow(config_path=cli.config, data_root=cli.data_root)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
