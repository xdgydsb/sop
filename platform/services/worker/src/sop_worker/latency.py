"""Per-frame pipeline timing and percentile summaries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrameLatency:
    sequence_no: int
    captured_ms: int
    enqueued_ms: int | None = None
    inference_started_ms: int | None = None
    inference_finished_ms: int | None = None
    judged_ms: int | None = None


@dataclass(frozen=True)
class LatencySummary:
    completed_frames: int
    capture_to_inference_p50_ms: float | None
    capture_to_inference_p95_ms: float | None
    inference_p50_ms: float | None
    inference_p95_ms: float | None
    end_to_end_p50_ms: float | None
    end_to_end_p95_ms: float | None


class PipelineLatencyTracker:
    def __init__(self, max_records: int = 1000) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.max_records = max_records
        self._records: dict[int, FrameLatency] = {}
        self._order: list[int] = []

    def captured(self, sequence_no: int, timestamp_ms: int) -> None:
        if sequence_no in self._records:
            raise ValueError("sequence_no already exists")
        self._records[sequence_no] = FrameLatency(sequence_no, timestamp_ms)
        self._order.append(sequence_no)
        self._trim()

    def enqueued(self, sequence_no: int, timestamp_ms: int) -> None:
        self._record(sequence_no).enqueued_ms = timestamp_ms

    def inference_started(self, sequence_no: int, timestamp_ms: int) -> None:
        self._record(sequence_no).inference_started_ms = timestamp_ms

    def inference_finished(self, sequence_no: int, timestamp_ms: int) -> None:
        self._record(sequence_no).inference_finished_ms = timestamp_ms

    def judged(self, sequence_no: int, timestamp_ms: int) -> None:
        self._record(sequence_no).judged_ms = timestamp_ms

    def summary(self) -> LatencySummary:
        records = list(self._records.values())
        queue_wait = [
            item.inference_started_ms - item.captured_ms
            for item in records
            if item.inference_started_ms is not None
        ]
        inference = [
            item.inference_finished_ms - item.inference_started_ms
            for item in records
            if item.inference_started_ms is not None
            and item.inference_finished_ms is not None
        ]
        end_to_end = [
            item.judged_ms - item.captured_ms
            for item in records
            if item.judged_ms is not None
        ]
        return LatencySummary(
            completed_frames=len(end_to_end),
            capture_to_inference_p50_ms=self._percentile(queue_wait, 0.50),
            capture_to_inference_p95_ms=self._percentile(queue_wait, 0.95),
            inference_p50_ms=self._percentile(inference, 0.50),
            inference_p95_ms=self._percentile(inference, 0.95),
            end_to_end_p50_ms=self._percentile(end_to_end, 0.50),
            end_to_end_p95_ms=self._percentile(end_to_end, 0.95),
        )

    def _record(self, sequence_no: int) -> FrameLatency:
        try:
            return self._records[sequence_no]
        except KeyError as exc:
            raise KeyError(f"unknown sequence_no: {sequence_no}") from exc

    def _trim(self) -> None:
        while len(self._order) > self.max_records:
            oldest = self._order.pop(0)
            self._records.pop(oldest, None)

    @staticmethod
    def _percentile(values: list[int], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return round(
            ordered[lower] + (ordered[upper] - ordered[lower]) * fraction,
            2,
        )
