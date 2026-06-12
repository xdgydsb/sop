"""Lightweight motion scoring for retaining fast operator-action frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


Roi = tuple[int, int, int, int]


@dataclass(frozen=True)
class MotionKeyframeConfig:
    pixel_delta: int = 18
    roi_motion_ratio: float = 0.025
    global_motion_ratio: float = 0.45
    min_keyframe_gap_ms: int = 80
    heartbeat_ms: int = 500
    analysis_width: int = 320
    blur_kernel: int = 5

    def __post_init__(self) -> None:
        if not 1 <= self.pixel_delta <= 255:
            raise ValueError("pixel_delta must be between 1 and 255")
        if not 0 < self.roi_motion_ratio <= 1:
            raise ValueError("roi_motion_ratio must be in (0, 1]")
        if not 0 < self.global_motion_ratio <= 1:
            raise ValueError("global_motion_ratio must be in (0, 1]")
        if self.min_keyframe_gap_ms < 0:
            raise ValueError("min_keyframe_gap_ms must be non-negative")
        if self.heartbeat_ms <= 0:
            raise ValueError("heartbeat_ms must be positive")
        if self.analysis_width < 64:
            raise ValueError("analysis_width must be at least 64")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd number")


@dataclass(frozen=True)
class KeyframeDecision:
    keyframe: bool
    reason: str
    score: float
    global_motion_ratio: float
    roi_motion: Mapping[str, float]


class MotionKeyframeScorer:
    """Detect local action bursts without treating camera shake as an action.

    ROIs use source-frame pixel coordinates. Scoring runs on a downscaled gray
    image, so it is cheap enough to execute on every captured frame.
    """

    def __init__(
        self,
        rois: Mapping[str, Sequence[int]],
        config: MotionKeyframeConfig | None = None,
    ) -> None:
        self.config = config or MotionKeyframeConfig()
        self._source_rois = {
            name: self._validate_roi(name, roi) for name, roi in rois.items()
        }
        if not self._source_rois:
            raise ValueError("at least one ROI is required")
        self._previous_gray: np.ndarray | None = None
        self._last_keyframe_ms: int | None = None
        self._last_emitted_ms: int | None = None

    def reset(self) -> None:
        self._previous_gray = None
        self._last_keyframe_ms = None
        self._last_emitted_ms = None

    def score(self, frame: np.ndarray, timestamp_ms: int) -> KeyframeDecision:
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        gray, scale = self._prepare(frame)

        if self._previous_gray is None:
            self._previous_gray = gray
            self._last_keyframe_ms = timestamp_ms
            self._last_emitted_ms = timestamp_ms
            return KeyframeDecision(
                keyframe=True,
                reason="bootstrap",
                score=1.0,
                global_motion_ratio=0.0,
                roi_motion={name: 0.0 for name in self._source_rois},
            )

        delta = cv2.absdiff(self._previous_gray, gray)
        motion_mask = delta >= self.config.pixel_delta
        global_ratio = float(np.count_nonzero(motion_mask) / motion_mask.size)
        roi_motion = self._score_rois(motion_mask, scale)
        self._previous_gray = gray

        strongest_name, strongest_ratio = max(
            roi_motion.items(), key=lambda item: item[1]
        )
        local_action = (
            strongest_ratio >= self.config.roi_motion_ratio
            and global_ratio < self.config.global_motion_ratio
        )
        gap_ok = (
            self._last_keyframe_ms is None
            or timestamp_ms - self._last_keyframe_ms
            >= self.config.min_keyframe_gap_ms
        )
        heartbeat_due = (
            self._last_emitted_ms is None
            or timestamp_ms - self._last_emitted_ms >= self.config.heartbeat_ms
        )

        if local_action and gap_ok:
            keyframe = True
            reason = f"roi_motion:{strongest_name}"
            self._last_keyframe_ms = timestamp_ms
        elif heartbeat_due:
            keyframe = True
            reason = "heartbeat"
            self._last_keyframe_ms = timestamp_ms
        elif global_ratio >= self.config.global_motion_ratio:
            keyframe = False
            reason = "global_motion_suppressed"
        elif local_action:
            keyframe = False
            reason = "keyframe_gap"
        else:
            keyframe = False
            reason = "routine"

        self._last_emitted_ms = timestamp_ms
        return KeyframeDecision(
            keyframe=keyframe,
            reason=reason,
            score=strongest_ratio,
            global_motion_ratio=global_ratio,
            roi_motion=roi_motion,
        )

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        if frame is None or frame.size == 0:
            raise ValueError("frame must not be empty")
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] in (3, 4):
            code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(frame, code)
        else:
            raise ValueError("frame must be grayscale, BGR, or BGRA")

        source_width = gray.shape[1]
        target_width = min(source_width, self.config.analysis_width)
        scale = target_width / source_width
        if target_width != source_width:
            target_height = max(1, round(gray.shape[0] * scale))
            gray = cv2.resize(
                gray, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
        if self.config.blur_kernel > 1:
            gray = cv2.GaussianBlur(
                gray,
                (self.config.blur_kernel, self.config.blur_kernel),
                0,
            )
        return gray, scale

    def _score_rois(
        self, motion_mask: np.ndarray, scale: float
    ) -> dict[str, float]:
        height, width = motion_mask.shape
        result: dict[str, float] = {}
        for name, roi in self._source_rois.items():
            x1, y1, x2, y2 = (
                round(roi[0] * scale),
                round(roi[1] * scale),
                round(roi[2] * scale),
                round(roi[3] * scale),
            )
            x1, x2 = sorted((max(0, x1), min(width, x2)))
            y1, y2 = sorted((max(0, y1), min(height, y2)))
            if x2 <= x1 or y2 <= y1:
                result[name] = 0.0
                continue
            roi_mask = motion_mask[y1:y2, x1:x2]
            result[name] = float(np.count_nonzero(roi_mask) / roi_mask.size)
        return result

    @staticmethod
    def _validate_roi(name: str, roi: Sequence[int]) -> Roi:
        if len(roi) != 4:
            raise ValueError(f"ROI {name!r} must contain four coordinates")
        x1, y1, x2, y2 = (int(value) for value in roi)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"ROI {name!r} must have positive area")
        return x1, y1, x2, y2
