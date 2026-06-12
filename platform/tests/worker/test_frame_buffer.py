from __future__ import annotations

import sys
import unittest
from pathlib import Path


PLATFORM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM_ROOT / "services" / "worker" / "src"))

from sop_worker import FrameEnvelope, RealtimeFrameBuffer


def frame(sequence_no: int, *, keyframe: bool = False) -> FrameEnvelope:
    return FrameEnvelope(
        sequence_no=sequence_no,
        timestamp_ms=sequence_no * 50,
        payload=f"frame-{sequence_no}",
        keyframe=keyframe,
        reason="interaction" if keyframe else "",
    )


class RealtimeFrameBufferTests(unittest.TestCase):
    def test_overflow_drops_routine_frame_before_action_keyframe(self) -> None:
        buffer = RealtimeFrameBuffer(capacity=4)
        buffer.push(frame(1))
        buffer.push(frame(2, keyframe=True))
        buffer.push(frame(3))
        buffer.push(frame(4, keyframe=True))
        buffer.push(frame(5))

        self.assertEqual([2, 3, 4, 5], [item.sequence_no for item in buffer.snapshot()])
        self.assertEqual(1, buffer.dropped_routine)
        self.assertEqual(0, buffer.dropped_keyframes)

    def test_all_keyframes_still_keep_latency_bounded(self) -> None:
        buffer = RealtimeFrameBuffer(capacity=3)
        for sequence_no in range(1, 5):
            buffer.push(frame(sequence_no, keyframe=True))

        self.assertEqual([2, 3, 4], [item.sequence_no for item in buffer.snapshot()])
        self.assertEqual(1, buffer.dropped_keyframes)

    def test_frames_are_consumed_in_temporal_order(self) -> None:
        buffer = RealtimeFrameBuffer(capacity=3)
        buffer.push(frame(1))
        buffer.push(frame(2, keyframe=True))
        buffer.push(frame(3))

        self.assertEqual([1, 2, 3], [buffer.pop().sequence_no for _ in range(3)])


if __name__ == "__main__":
    unittest.main()
