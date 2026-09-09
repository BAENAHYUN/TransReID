from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class AdaptiveSamplingConfig:
    """Content-aware adaptive video sampling.

    Motion/change is estimated on small grayscale probe frames.
    Low change  -> sparse sampling (default 1 FPS)
    Medium      -> moderate sampling (default 3 FPS)
    High change -> dense sampling (default 10 FPS)

    The detector/tracker still receives normal single frames.  Contact sheets
    are intentionally a separate review/export feature so bbox coordinates and
    timestamps remain valid.
    """

    low_motion_fps: float = 1.0
    medium_motion_fps: float = 3.0
    high_motion_fps: float = 10.0
    probe_fps: float = 10.0

    # mean absolute grayscale difference / 255.0
    low_change_threshold: float = 0.025
    high_change_threshold: float = 0.080
    scene_cut_threshold: float = 0.180

    # Larger -> faster reaction, smaller -> smoother mode switching.
    ema_alpha: float = 0.60

    # Small probe image keeps CPU overhead low.
    probe_width: int = 160
    probe_height: int = 90

    # Always emit at least once within this interval.
    max_sample_gap_sec: float = 1.0


def _probe_gray(frame: np.ndarray, cfg: AdaptiveSamplingConfig) -> np.ndarray:
    small = cv2.resize(
        frame,
        (int(cfg.probe_width), int(cfg.probe_height)),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _change_score(previous: np.ndarray, current: np.ndarray) -> float:
    diff = cv2.absdiff(previous, current)
    return float(np.mean(diff) / 255.0)


def _target_fps(score: float, cfg: AdaptiveSamplingConfig) -> tuple[str, float]:
    if score < cfg.low_change_threshold:
        return "LOW", max(0.1, float(cfg.low_motion_fps))
    if score < cfg.high_change_threshold:
        return "MEDIUM", max(0.1, float(cfg.medium_motion_fps))
    return "HIGH", max(0.1, float(cfg.high_motion_fps))


def adaptive_extract_frames(
    video_path: str | Path,
    cfg: AdaptiveSamplingConfig | None = None,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Yield adaptively selected frames as (frame_number, timestamp, frame).

    Important: every source frame is *not* sent to RF-DETR/embedding models.
    A lightweight probe runs up to ``probe_fps`` and controls the expensive
    detector cadence dynamically.
    """

    cfg = cfg or AdaptiveSamplingConfig()
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if total_frames > 0 else 0.0

    # Probe cadence controls the maximum density. For 30~32 FPS sources,
    # probe_step is normally 3 -> about 10 FPS.
    probe_step = max(1, int(round(fps / max(0.1, cfg.probe_fps))))

    print(f"FPS             : {fps:.2f}")
    print(f"Total frames    : {total_frames}")
    print(f"Duration        : {duration:.2f} sec")
    print(f"Sampling        : ADAPTIVE / content-aware")
    print(f"Probe FPS       : ~{fps / probe_step:.2f}")
    print(
        "Target FPS      : "
        f"LOW={cfg.low_motion_fps:g} / "
        f"MEDIUM={cfg.medium_motion_fps:g} / "
        f"HIGH={cfg.high_motion_fps:g}"
    )
    print(
        "Change threshold: "
        f"LOW<{cfg.low_change_threshold:.3f} / "
        f"HIGH>={cfg.high_change_threshold:.3f} / "
        f"CUT>={cfg.scene_cut_threshold:.3f}"
    )

    frame_number = 0
    previous_probe = None
    change_ema = 0.0
    last_emit_time = -1e9
    current_mode = "LOW"
    current_target_fps = max(0.1, cfg.low_motion_fps)

    emitted = 0
    mode_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CUT": 0}

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break

            # Only decode/retrieve probe frames. This keeps the cheap change
            # estimator much lighter than running RF-DETR on every frame.
            is_probe = (frame_number % probe_step == 0)
            if not is_probe:
                frame_number += 1
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                frame_number += 1
                continue

            timestamp = frame_number / fps
            gray = _probe_gray(frame, cfg)

            scene_cut = False
            raw_change = 0.0
            if previous_probe is not None:
                raw_change = _change_score(previous_probe, gray)
                alpha = min(1.0, max(0.0, float(cfg.ema_alpha)))
                change_ema = alpha * raw_change + (1.0 - alpha) * change_ema
                scene_cut = raw_change >= cfg.scene_cut_threshold
            previous_probe = gray

            current_mode, current_target_fps = _target_fps(change_ema, cfg)
            target_interval = 1.0 / max(0.1, current_target_fps)
            max_gap = max(target_interval, float(cfg.max_sample_gap_sec))

            due_by_target = (timestamp - last_emit_time) >= (target_interval - 0.5 / fps)
            due_by_max_gap = (timestamp - last_emit_time) >= (max_gap - 0.5 / fps)
            first = emitted == 0

            if first or scene_cut or due_by_target or due_by_max_gap:
                mode_key = "CUT" if scene_cut else current_mode
                mode_counts[mode_key] = mode_counts.get(mode_key, 0) + 1
                emitted += 1
                last_emit_time = timestamp
                yield frame_number, timestamp, frame

            frame_number += 1
    finally:
        cap.release()
        print(
            "Adaptive summary: "
            f"selected={emitted} / "
            f"LOW={mode_counts.get('LOW', 0)} / "
            f"MEDIUM={mode_counts.get('MEDIUM', 0)} / "
            f"HIGH={mode_counts.get('HIGH', 0)} / "
            f"CUT={mode_counts.get('CUT', 0)}"
        )


def make_contact_sheet(
    frames: Sequence[np.ndarray],
    *,
    columns: int = 4,
    rows: int = 4,
    cell_width: int = 320,
    cell_height: int = 180,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Create a review-only contact sheet, e.g. 16 frames -> 4x4.

    This is for investigator/Qwen review, not for RF-DETR tracking input.
    """

    capacity = max(1, int(columns) * int(rows))
    selected = list(frames[:capacity])
    if not selected:
        raise ValueError("frames is empty")

    canvas = np.zeros(
        (int(rows) * int(cell_height), int(columns) * int(cell_width), 3),
        dtype=np.uint8,
    )

    for idx, frame in enumerate(selected):
        r = idx // columns
        c = idx % columns
        resized = cv2.resize(frame, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        y1, y2 = r * cell_height, (r + 1) * cell_height
        x1, x2 = c * cell_width, (c + 1) * cell_width
        canvas[y1:y2, x1:x2] = resized

        if labels and idx < len(labels):
            cv2.rectangle(canvas, (x1, y1), (x2, min(y1 + 28, y2)), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                str(labels[idx]),
                (x1 + 6, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return canvas
