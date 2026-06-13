"""Realtime performance budgets for fast industrial operator actions."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class RealtimePerformanceBudget:
    capture_fps: float
    inference_fps: float
    shortest_action_ms: int = 180
    queue_capacity: int = 3

    def __post_init__(self) -> None:
        if self.capture_fps <= 0 or self.inference_fps <= 0:
            raise ValueError("FPS values must be positive")
        if self.shortest_action_ms <= 0:
            raise ValueError("shortest_action_ms must be positive")
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")

    @property
    def captured_action_frames(self) -> float:
        return self.capture_fps * self.shortest_action_ms / 1000

    @property
    def inferred_action_frames(self) -> float:
        return self.inference_fps * self.shortest_action_ms / 1000

    @property
    def worst_queue_delay_ms(self) -> float:
        return self.queue_capacity / self.inference_fps * 1000

    @property
    def required_capture_fps(self) -> int:
        # Before, transition, and after should each have an observation.
        return ceil(3000 / self.shortest_action_ms)

    def assessment(self) -> dict[str, float | int | str]:
        if self.captured_action_frames < 3:
            risk = "CAPTURE_UNDERSAMPLED"
        elif self.inferred_action_frames < 2:
            risk = "INFERENCE_UNDERSAMPLED"
        elif self.worst_queue_delay_ms > self.shortest_action_ms:
            risk = "QUEUE_DELAY_EXCEEDS_ACTION"
        else:
            risk = "KEYFRAME_PRIORITY_REQUIRED"
        return {
            "risk": risk,
            "capturedActionFrames": round(self.captured_action_frames, 2),
            "inferredActionFrames": round(self.inferred_action_frames, 2),
            "worstQueueDelayMs": round(self.worst_queue_delay_ms, 1),
            "requiredCaptureFps": self.required_capture_fps,
        }
