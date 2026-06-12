from __future__ import annotations

import sys
import unittest
from pathlib import Path


PLATFORM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM_ROOT / "services" / "worker" / "src"))

from sop_worker import PipelineLatencyTracker


class PipelineLatencyTrackerTests(unittest.TestCase):
    def test_reports_pipeline_percentiles(self) -> None:
        tracker = PipelineLatencyTracker()
        for sequence_no, base, queue_wait, inference, total in (
            (1, 1000, 20, 40, 80),
            (2, 2000, 40, 60, 120),
            (3, 3000, 60, 80, 160),
        ):
            tracker.captured(sequence_no, base)
            tracker.enqueued(sequence_no, base + 2)
            tracker.inference_started(sequence_no, base + queue_wait)
            tracker.inference_finished(
                sequence_no, base + queue_wait + inference
            )
            tracker.judged(sequence_no, base + total)

        summary = tracker.summary()
        self.assertEqual(3, summary.completed_frames)
        self.assertEqual(40.0, summary.capture_to_inference_p50_ms)
        self.assertEqual(60.0, summary.inference_p50_ms)
        self.assertEqual(120.0, summary.end_to_end_p50_ms)
        self.assertEqual(156.0, summary.end_to_end_p95_ms)

    def test_rejects_unknown_sequence(self) -> None:
        tracker = PipelineLatencyTracker()
        with self.assertRaisesRegex(KeyError, "unknown sequence_no"):
            tracker.judged(99, 1000)


if __name__ == "__main__":
    unittest.main()
