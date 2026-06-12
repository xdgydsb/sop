"""Bounded frame buffer that favors action evidence over repeated frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrameEnvelope:
    sequence_no: int
    timestamp_ms: int
    payload: Any
    keyframe: bool = False
    reason: str = ""


class RealtimeFrameBuffer:
    """Keep latency bounded while retaining scarce action/interaction frames.

    Routine frames are dropped before keyframes. Frames that remain are always
    consumed in timestamp order, so the temporal runtime never sees reordered
    evidence.
    """

    def __init__(self, capacity: int = 6) -> None:
        if capacity < 2:
            raise ValueError("capacity must be at least 2")
        self.capacity = capacity
        self._frames: deque[FrameEnvelope] = deque()
        self.dropped_routine = 0
        self.dropped_keyframes = 0

    def push(self, frame: FrameEnvelope) -> None:
        if self._frames and frame.sequence_no <= self._frames[-1].sequence_no:
            raise ValueError("frame sequence_no must be strictly increasing")
        self._frames.append(frame)
        if len(self._frames) <= self.capacity:
            return

        drop_index = next(
            (index for index, item in enumerate(self._frames) if not item.keyframe),
            0,
        )
        dropped = self._frames[drop_index]
        del self._frames[drop_index]
        if dropped.keyframe:
            self.dropped_keyframes += 1
        else:
            self.dropped_routine += 1

    def pop(self) -> FrameEnvelope | None:
        if not self._frames:
            return None
        return self._frames.popleft()

    def __len__(self) -> int:
        return len(self._frames)

    def snapshot(self) -> tuple[FrameEnvelope, ...]:
        return tuple(self._frames)
