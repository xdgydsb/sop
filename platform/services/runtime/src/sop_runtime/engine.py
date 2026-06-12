"""State-transition based SOP runtime.

The runtime consumes normalized observations. It intentionally knows nothing
about YOLO, camera SDKs, HTTP, databases, or user interfaces.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from sop_contracts import (
    Deviation,
    DeviationType,
    Judgement,
    JudgementStatus,
    Observation,
    OperationDefinition,
    SopDefinition,
)


@dataclass(frozen=True)
class RuntimeSnapshot:
    current_operation_id: str | None
    statuses: dict[str, JudgementStatus]
    judgements: tuple[Judgement, ...]
    deviations: tuple[Deviation, ...]
    complete: bool
    interrupted: bool


class SopRuntime:
    def __init__(self, definition: SopDefinition) -> None:
        self.definition = definition
        self._index_by_id = {
            operation.operation_id: index
            for index, operation in enumerate(definition.operations)
        }
        self.reset()

    def reset(self) -> None:
        self._previous: Observation | None = None
        self._history: deque[Observation] = deque()
        self._cycle_start_ms: int | None = None
        self._operation_started_ms: int | None = None
        self._current_index = 0
        self._candidate_started_ms: int | None = None
        self._candidate_operation_id: str | None = None
        self._statuses = {
            operation.operation_id: JudgementStatus.WAITING
            for operation in self.definition.operations
        }
        self._judgements: list[Judgement] = []
        self._deviations: list[Deviation] = []
        self._deviation_keys: set[tuple[DeviationType, str, int]] = set()
        self._interrupted = False

    def update(self, observation: Observation) -> RuntimeSnapshot:
        if (
            self._previous is not None
            and observation.timestamp_ms <= self._previous.timestamp_ms
        ):
            raise ValueError("observations must have strictly increasing timestamps")
        if self._cycle_start_ms is None:
            self._cycle_start_ms = observation.timestamp_ms
            self._operation_started_ms = observation.timestamp_ms

        if not observation.source_healthy:
            self._record_interruption(observation)
            self._previous = observation
            self._history.clear()
            return self.snapshot()

        if not observation.confidence_sufficient:
            self._record_uncertain(observation)
            self._previous = observation
            self._append_history(observation)
            return self.snapshot()

        self._interrupted = False
        if self._previous is None or not self._previous.source_healthy:
            self._previous = observation
            self._append_history(observation)
            return self.snapshot()

        self._detect_wrong_order(observation)
        if not self.complete:
            operation = self.definition.operations[self._current_index]
            self._evaluate_current(operation, observation)

        self._previous = observation
        self._append_history(observation)
        return self.snapshot()

    @property
    def complete(self) -> bool:
        return self._current_index >= len(self.definition.operations)

    def snapshot(self) -> RuntimeSnapshot:
        current = None
        if not self.complete:
            current = self.definition.operations[self._current_index].operation_id
        return RuntimeSnapshot(
            current_operation_id=current,
            statuses=dict(self._statuses),
            judgements=tuple(self._judgements),
            deviations=tuple(self._deviations),
            complete=self.complete,
            interrupted=self._interrupted,
        )

    def _evaluate_current(
        self, operation: OperationDefinition, observation: Observation
    ) -> None:
        assert self._previous is not None

        if self._candidate_operation_id == operation.operation_id:
            if self._conditions_match(operation.postconditions, observation):
                assert self._candidate_started_ms is not None
                if observation.timestamp_ms - self._candidate_started_ms >= operation.hold_ms:
                    self._pass(operation, observation)
                return
            self._candidate_operation_id = None
            self._candidate_started_ms = None
            self._statuses[operation.operation_id] = JudgementStatus.WAITING

        if self._timed_out(operation, observation):
            self._record_timeout(operation, observation)
            return

        prerequisites_passed = all(
            self._statuses[operation_id] is JudgementStatus.PASS
            for operation_id in operation.prerequisites
        )
        transition_start = self._find_transition_start(
            operation.trigger, observation
        )
        preconditions_match = (
            transition_start is not None
            and self._conditions_match(operation.preconditions, transition_start)
        )
        trigger_matches = transition_start is not None
        if not (prerequisites_passed and preconditions_match and trigger_matches):
            return

        self._candidate_operation_id = operation.operation_id
        self._candidate_started_ms = observation.timestamp_ms
        self._statuses[operation.operation_id] = JudgementStatus.IN_PROGRESS
        self._judgements.append(
            Judgement(
                operation_id=operation.operation_id,
                status=JudgementStatus.IN_PROGRESS,
                timestamp_ms=observation.timestamp_ms,
                reason="required state transition observed; verifying postcondition",
                frame_ref=observation.frame_ref,
            )
        )
        if operation.hold_ms == 0 and self._conditions_match(
            operation.postconditions, observation
        ):
            self._pass(operation, observation)

    def _detect_wrong_order(self, observation: Observation) -> None:
        assert self._previous is not None
        for future in self.definition.operations[self._current_index + 1 :]:
            if self._find_transition_start(future.trigger, observation) is None:
                continue
            key = (
                DeviationType.WRONG_ORDER,
                future.operation_id,
                observation.timestamp_ms,
            )
            if key in self._deviation_keys:
                continue
            self._deviation_keys.add(key)
            expected = self.definition.operations[self._current_index].operation_id
            self._deviations.append(
                Deviation(
                    deviation_type=DeviationType.WRONG_ORDER,
                    operation_id=future.operation_id,
                    timestamp_ms=observation.timestamp_ms,
                    reason=(
                        f"observed {future.operation_id} while waiting for {expected}"
                    ),
                    frame_ref=observation.frame_ref,
                    metadata={"expected_operation_id": expected},
                )
            )

    def _pass(self, operation: OperationDefinition, observation: Observation) -> None:
        self._statuses[operation.operation_id] = JudgementStatus.PASS
        self._judgements.append(
            Judgement(
                operation_id=operation.operation_id,
                status=JudgementStatus.PASS,
                timestamp_ms=observation.timestamp_ms,
                reason="transition and stable postcondition satisfied",
                frame_ref=observation.frame_ref,
            )
        )
        self._current_index += 1
        self._operation_started_ms = observation.timestamp_ms
        self._candidate_operation_id = None
        self._candidate_started_ms = None

    def _timed_out(
        self, operation: OperationDefinition, observation: Observation
    ) -> bool:
        if operation.timeout_ms is None or self._operation_started_ms is None:
            return False
        return (
            observation.timestamp_ms - self._operation_started_ms
            >= operation.timeout_ms
        )

    def _record_timeout(
        self, operation: OperationDefinition, observation: Observation
    ) -> None:
        if self._statuses[operation.operation_id] is JudgementStatus.TIMEOUT:
            return
        self._statuses[operation.operation_id] = JudgementStatus.TIMEOUT
        self._judgements.append(
            Judgement(
                operation_id=operation.operation_id,
                status=JudgementStatus.TIMEOUT,
                timestamp_ms=observation.timestamp_ms,
                reason="operation exceeded its allowed cycle time",
                frame_ref=observation.frame_ref,
            )
        )
        self._deviations.append(
            Deviation(
                deviation_type=DeviationType.TIMEOUT,
                operation_id=operation.operation_id,
                timestamp_ms=observation.timestamp_ms,
                reason="operation timeout",
                frame_ref=observation.frame_ref,
            )
        )

    def _record_interruption(self, observation: Observation) -> None:
        self._interrupted = True
        self._cancel_candidate()
        operation_id = self._current_operation_id()
        self._judgements.append(
            Judgement(
                operation_id=operation_id,
                status=JudgementStatus.INTERRUPTED,
                timestamp_ms=observation.timestamp_ms,
                reason="observation source is unhealthy",
                frame_ref=observation.frame_ref,
            )
        )
        self._deviations.append(
            Deviation(
                deviation_type=DeviationType.SYSTEM_INTERRUPTED,
                operation_id=operation_id,
                timestamp_ms=observation.timestamp_ms,
                reason="camera or inference source interrupted",
                frame_ref=observation.frame_ref,
            )
        )

    def _record_uncertain(self, observation: Observation) -> None:
        self._cancel_candidate()
        operation_id = self._current_operation_id()
        self._judgements.append(
            Judgement(
                operation_id=operation_id,
                status=JudgementStatus.UNCERTAIN,
                timestamp_ms=observation.timestamp_ms,
                reason="observation confidence is insufficient",
                frame_ref=observation.frame_ref,
            )
        )
        self._deviations.append(
            Deviation(
                deviation_type=DeviationType.VISION_UNCERTAIN,
                operation_id=operation_id,
                timestamp_ms=observation.timestamp_ms,
                reason="vision observation cannot support a reliable judgement",
                frame_ref=observation.frame_ref,
            )
        )

    def _current_operation_id(self) -> str:
        if self.complete:
            return "cycle"
        return self.definition.operations[self._current_index].operation_id

    def _cancel_candidate(self) -> None:
        if self._candidate_operation_id is not None:
            self._statuses[self._candidate_operation_id] = JudgementStatus.WAITING
        self._candidate_operation_id = None
        self._candidate_started_ms = None

    def _find_transition_start(
        self, trigger, current: Observation
    ) -> Observation | None:
        if current.facts.get(trigger.key) != trigger.to_value:
            return None

        candidates = list(self._history)
        if self._previous is not None and (
            not candidates or candidates[-1] is not self._previous
        ):
            candidates.append(self._previous)

        for index in range(len(candidates) - 1, -1, -1):
            start = candidates[index]
            if current.timestamp_ms - start.timestamp_ms > trigger.max_gap_ms:
                break
            if start.facts.get(trigger.key) != trigger.from_value:
                continue
            if trigger.interaction is None:
                return start
            interval = candidates[index:] + [current]
            if any(trigger.interaction in item.interactions for item in interval):
                return start
        return None

    def _append_history(self, observation: Observation) -> None:
        self._history.append(observation)
        max_gap_ms = max(
            operation.trigger.max_gap_ms for operation in self.definition.operations
        )
        cutoff = observation.timestamp_ms - max_gap_ms
        while self._history and self._history[0].timestamp_ms < cutoff:
            self._history.popleft()

    @staticmethod
    def _conditions_match(
        conditions: tuple, observation: Observation
    ) -> bool:
        return all(condition.matches(observation.facts) for condition in conditions)
