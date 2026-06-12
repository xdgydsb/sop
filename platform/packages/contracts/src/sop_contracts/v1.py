"""Execution contracts for protocol version 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class FactOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_OR_EQUAL = "greater_or_equal"
    LESS_OR_EQUAL = "less_or_equal"


class JudgementStatus(str, Enum):
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    PASS = "pass"
    REJECT = "reject"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    UNCERTAIN = "uncertain"
    INTERRUPTED = "interrupted"


class DeviationType(str, Enum):
    WRONG_ORDER = "wrong_order"
    INVALID_ACTION = "invalid_action"
    TIMEOUT = "timeout"
    VISION_UNCERTAIN = "vision_uncertain"
    SYSTEM_INTERRUPTED = "system_interrupted"


@dataclass(frozen=True)
class FactCondition:
    key: str
    operator: FactOperator
    value: Any

    def matches(self, facts: Mapping[str, Any]) -> bool:
        if self.key not in facts:
            return False
        actual = facts[self.key]
        if self.operator is FactOperator.EQUALS:
            return actual == self.value
        if self.operator is FactOperator.NOT_EQUALS:
            return actual != self.value
        if self.operator is FactOperator.GREATER_OR_EQUAL:
            return actual >= self.value
        if self.operator is FactOperator.LESS_OR_EQUAL:
            return actual <= self.value
        return False


@dataclass(frozen=True)
class TransitionCondition:
    key: str
    from_value: Any
    to_value: Any
    interaction: str | None = None

    def matches(self, previous: "Observation", current: "Observation") -> bool:
        changed = (
            previous.facts.get(self.key) == self.from_value
            and current.facts.get(self.key) == self.to_value
        )
        if not changed:
            return False
        if self.interaction is None:
            return True
        return (
            self.interaction in previous.interactions
            or self.interaction in current.interactions
        )


@dataclass(frozen=True)
class Observation:
    timestamp_ms: int
    facts: Mapping[str, Any]
    interactions: frozenset[str] = frozenset()
    source_healthy: bool = True
    confidence_sufficient: bool = True
    frame_ref: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))
        object.__setattr__(self, "interactions", frozenset(self.interactions))


@dataclass(frozen=True)
class OperationDefinition:
    operation_id: str
    name: str
    trigger: TransitionCondition
    prerequisites: tuple[str, ...] = ()
    preconditions: tuple[FactCondition, ...] = ()
    postconditions: tuple[FactCondition, ...] = ()
    hold_ms: int = 300
    timeout_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.operation_id.strip():
            raise ValueError("operation_id is required")
        if self.hold_ms < 0:
            raise ValueError("hold_ms must be non-negative")
        if self.timeout_ms is not None and self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")


@dataclass(frozen=True)
class SopDefinition:
    sop_id: str
    version: int
    operations: tuple[OperationDefinition, ...]

    def __post_init__(self) -> None:
        if not self.sop_id.strip():
            raise ValueError("sop_id is required")
        if self.version <= 0:
            raise ValueError("version must be positive")
        ids = [operation.operation_id for operation in self.operations]
        if not ids:
            raise ValueError("at least one operation is required")
        if len(ids) != len(set(ids)):
            raise ValueError("operation_id values must be unique")
        known: set[str] = set()
        for operation in self.operations:
            missing = set(operation.prerequisites) - known
            if missing:
                raise ValueError(
                    f"{operation.operation_id} has unknown or future prerequisites: "
                    f"{sorted(missing)}"
                )
            known.add(operation.operation_id)


@dataclass(frozen=True)
class Judgement:
    operation_id: str
    status: JudgementStatus
    timestamp_ms: int
    reason: str
    frame_ref: str | None = None


@dataclass(frozen=True)
class Deviation:
    deviation_type: DeviationType
    operation_id: str
    timestamp_ms: int
    reason: str
    frame_ref: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
