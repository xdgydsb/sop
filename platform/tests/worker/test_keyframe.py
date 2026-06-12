from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PLATFORM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM_ROOT / "services" / "worker" / "src"))

from sop_worker import (
    FrameIngestController,
    MotionKeyframeConfig,
    MotionKeyframeScorer,
    RealtimeFrameBuffer,
)


def blank() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


class MotionKeyframeScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = MotionKeyframeScorer(
            {
                "box": (120, 60, 280, 210),
                "material": (20, 60, 110, 210),
            },
            MotionKeyframeConfig(
                pixel_delta=10,
                roi_motion_ratio=0.02,
                global_motion_ratio=0.40,
                min_keyframe_gap_ms=80,
                heartbeat_ms=500,
                analysis_width=320,
                blur_kernel=1,
            ),
        )

    def test_static_frames_are_routine_between_heartbeats(self) -> None:
        self.assertEqual("bootstrap", self.scorer.score(blank(), 0).reason)
        decision = self.scorer.score(blank(), 50)
        self.assertFalse(decision.keyframe)
        self.assertEqual("routine", decision.reason)
        heartbeat = self.scorer.score(blank(), 550)
        self.assertTrue(heartbeat.keyframe)
        self.assertEqual("heartbeat", heartbeat.reason)

    def test_fast_local_motion_is_keyframe(self) -> None:
        self.scorer.score(blank(), 0)
        moved = blank()
        cv2.rectangle(moved, (150, 90), (220, 160), (255, 255, 255), -1)
        decision = self.scorer.score(moved, 100)
        self.assertTrue(decision.keyframe)
        self.assertEqual("roi_motion:box", decision.reason)
        self.assertGreater(decision.roi_motion["box"], 0.02)

    def test_global_camera_change_is_suppressed(self) -> None:
        self.scorer.score(blank(), 0)
        flashed = np.full((240, 320, 3), 255, dtype=np.uint8)
        decision = self.scorer.score(flashed, 100)
        self.assertFalse(decision.keyframe)
        self.assertEqual("global_motion_suppressed", decision.reason)

    def test_keyframe_gap_prevents_queue_flood(self) -> None:
        self.scorer.score(blank(), 0)
        first = blank()
        cv2.rectangle(first, (150, 90), (220, 160), (255, 255, 255), -1)
        self.assertTrue(self.scorer.score(first, 100).keyframe)
        second = blank()
        cv2.rectangle(second, (180, 100), (250, 170), (255, 255, 255), -1)
        decision = self.scorer.score(second, 140)
        self.assertFalse(decision.keyframe)
        self.assertEqual("keyframe_gap", decision.reason)

    def test_ingest_controller_preserves_action_frame_under_congestion(self) -> None:
        controller = FrameIngestController(
            self.scorer, RealtimeFrameBuffer(capacity=3)
        )
        controller.ingest(1, 0, blank())
        controller.ingest(2, 50, blank())
        moved = blank()
        cv2.rectangle(moved, (150, 90), (220, 160), (255, 255, 255), -1)
        controller.ingest(3, 100, moved)
        controller.ingest(4, 150, moved)

        queued = controller.buffer.snapshot()
        self.assertIn(3, [item.sequence_no for item in queued])
        action_frame = next(item for item in queued if item.sequence_no == 3)
        self.assertTrue(action_frame.keyframe)
        self.assertEqual(1, controller.metrics().dropped_routine)


if __name__ == "__main__":
    unittest.main()
