from __future__ import annotations

import unittest

from sop_worker.performance import RealtimePerformanceBudget


class RealtimePerformanceBudgetTests(unittest.TestCase):
    def test_old_runtime_profile_exposes_fast_action_risk(self) -> None:
        budget = RealtimePerformanceBudget(
            capture_fps=20,
            inference_fps=12,
            shortest_action_ms=180,
            queue_capacity=6,
        )
        result = budget.assessment()
        self.assertEqual(3.6, result["capturedActionFrames"])
        self.assertEqual(2.16, result["inferredActionFrames"])
        self.assertEqual(500.0, result["worstQueueDelayMs"])
        self.assertEqual("QUEUE_DELAY_EXCEEDS_ACTION", result["risk"])

    def test_bounded_queue_keeps_delay_inside_action_window(self) -> None:
        budget = RealtimePerformanceBudget(
            capture_fps=30,
            inference_fps=25,
            shortest_action_ms=180,
            queue_capacity=3,
        )
        result = budget.assessment()
        self.assertLess(result["worstQueueDelayMs"], 180)
        self.assertEqual("KEYFRAME_PRIORITY_REQUIRED", result["risk"])


if __name__ == "__main__":
    unittest.main()
