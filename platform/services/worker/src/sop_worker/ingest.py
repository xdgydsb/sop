"""Frame ingestion controller for keyframe-aware realtime processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .frame_buffer import FrameEnvelope, RealtimeFrameBuffer
from .keyframe import KeyframeDecision, MotionKeyframeScorer


@dataclass(frozen=True)
class IngestMetrics:
    captured: int
    keyframes: int
    routine_frames: int
    queued: int
    dropped_routine: int
    dropped_keyframes: int


class FrameIngestController:
    def __init__(
        self,
        scorer: MotionKeyframeScorer,
        buffer: RealtimeFrameBuffer | None = None,
    ) -> None:
        self.scorer = scorer
        self.buffer = buffer if buffer is not None else RealtimeFrameBuffer()
        self._captured = 0
        self._keyframes = 0
        self._routine_frames = 0

    def ingest(
        self, sequence_no: int, timestamp_ms: int, frame: Any
    ) -> KeyframeDecision:
        decision = self.scorer.score(frame, timestamp_ms)
        self._captured += 1
        if decision.keyframe:
            self._keyframes += 1
        else:
            self._routine_frames += 1
        self.buffer.push(
            FrameEnvelope(
                sequence_no=sequence_no,
                timestamp_ms=timestamp_ms,
                payload=frame,
                keyframe=decision.keyframe,
                reason=decision.reason,
            )
        )
        return decision

    def next_frame(self) -> FrameEnvelope | None:
        return self.buffer.pop()

    def metrics(self) -> IngestMetrics:
        return IngestMetrics(
            captured=self._captured,
            keyframes=self._keyframes,
            routine_frames=self._routine_frames,
            queued=len(self.buffer),
            dropped_routine=self.buffer.dropped_routine,
            dropped_keyframes=self.buffer.dropped_keyframes,
        )
