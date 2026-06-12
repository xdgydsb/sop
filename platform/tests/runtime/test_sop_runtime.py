from __future__ import annotations

import sys
import unittest
from pathlib import Path


PLATFORM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM_ROOT / "packages" / "contracts" / "src"))
sys.path.insert(0, str(PLATFORM_ROOT / "services" / "runtime" / "src"))

from sop_contracts import DeviationType, JudgementStatus, Observation
from sop_runtime import SopRuntime
from sop_runtime.examples import package_five_step_sop


BASE_FACTS = {
    "box.state": "closed",
    "earphone.location": "outside",
    "charger.location": "outside",
    "green_bag.location": "outside",
}


def observation(
    timestamp_ms: int,
    *,
    facts: dict | None = None,
    interactions: set[str] | None = None,
    healthy: bool = True,
    confident: bool = True,
) -> Observation:
    values = dict(BASE_FACTS)
    if facts:
        values.update(facts)
    return Observation(
        timestamp_ms=timestamp_ms,
        facts=values,
        interactions=frozenset(interactions or set()),
        source_healthy=healthy,
        confidence_sufficient=confident,
        frame_ref=f"frame-{timestamp_ms}",
    )


class SopRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = SopRuntime(package_five_step_sop())

    def test_correct_sequence_passes_only_after_stable_transitions(self) -> None:
        timeline = [
            observation(0),
            observation(
                100,
                facts={"box.state": "open"},
                interactions={"hand:box"},
            ),
            observation(450, facts={"box.state": "open"}),
            observation(
                600,
                facts={"box.state": "open", "earphone.location": "inside"},
                interactions={"hand:earphone"},
            ),
            observation(
                950,
                facts={"box.state": "open", "earphone.location": "inside"},
            ),
            observation(
                1100,
                facts={
                    "box.state": "open",
                    "earphone.location": "inside",
                    "charger.location": "inside",
                },
                interactions={"hand:charger"},
            ),
            observation(
                1450,
                facts={
                    "box.state": "open",
                    "earphone.location": "inside",
                    "charger.location": "inside",
                },
            ),
            observation(
                1600,
                facts={
                    "box.state": "open",
                    "earphone.location": "inside",
                    "charger.location": "inside",
                    "green_bag.location": "inside",
                },
                interactions={"hand:green_bag"},
            ),
            observation(
                1950,
                facts={
                    "box.state": "open",
                    "earphone.location": "inside",
                    "charger.location": "inside",
                    "green_bag.location": "inside",
                },
            ),
            observation(
                2100,
                facts={
                    "box.state": "closed",
                    "earphone.location": "inside",
                    "charger.location": "inside",
                    "green_bag.location": "inside",
                },
                interactions={"hand:box"},
            ),
            observation(
                2450,
                facts={
                    "box.state": "closed",
                    "earphone.location": "inside",
                    "charger.location": "inside",
                    "green_bag.location": "inside",
                },
            ),
        ]

        snapshot = None
        for item in timeline:
            snapshot = self.runtime.update(item)

        assert snapshot is not None
        self.assertTrue(snapshot.complete)
        self.assertTrue(
            all(status is JudgementStatus.PASS for status in snapshot.statuses.values())
        )
        self.assertEqual((), snapshot.deviations)

    def test_object_on_closed_box_does_not_complete_open_or_place_step(self) -> None:
        self.runtime.update(observation(0))
        snapshot = self.runtime.update(
            observation(
                100,
                facts={"earphone.location": "inside"},
                interactions={"hand:earphone"},
            )
        )
        snapshot = self.runtime.update(
            observation(500, facts={"earphone.location": "inside"})
        )

        self.assertEqual(
            JudgementStatus.WAITING, snapshot.statuses["open_box"]
        )
        self.assertEqual(
            JudgementStatus.WAITING, snapshot.statuses["place_earphone"]
        )
        self.assertTrue(
            any(
                item.deviation_type is DeviationType.WRONG_ORDER
                and item.operation_id == "place_earphone"
                for item in snapshot.deviations
            )
        )

    def test_box_label_change_without_hand_is_not_open_action(self) -> None:
        self.runtime.update(observation(0))
        snapshot = self.runtime.update(
            observation(100, facts={"box.state": "open"})
        )
        snapshot = self.runtime.update(
            observation(500, facts={"box.state": "open"})
        )
        self.assertEqual(
            JudgementStatus.WAITING, snapshot.statuses["open_box"]
        )

    def test_single_frame_transition_is_not_enough(self) -> None:
        self.runtime.update(observation(0))
        snapshot = self.runtime.update(
            observation(
                100,
                facts={"box.state": "open"},
                interactions={"hand:box"},
            )
        )
        self.assertEqual(
            JudgementStatus.IN_PROGRESS, snapshot.statuses["open_box"]
        )
        snapshot = self.runtime.update(observation(150))
        self.assertEqual(
            JudgementStatus.WAITING, snapshot.statuses["open_box"]
        )

    def test_source_failure_is_interrupted_not_operator_reject(self) -> None:
        self.runtime.update(observation(0))
        snapshot = self.runtime.update(observation(100, healthy=False))

        self.assertTrue(snapshot.interrupted)
        self.assertEqual(
            JudgementStatus.WAITING, snapshot.statuses["open_box"]
        )
        self.assertEqual(
            JudgementStatus.INTERRUPTED, snapshot.judgements[-1].status
        )
        self.assertEqual(
            DeviationType.SYSTEM_INTERRUPTED,
            snapshot.deviations[-1].deviation_type,
        )

    def test_uncertain_frame_does_not_advance_operation(self) -> None:
        self.runtime.update(observation(0))
        snapshot = self.runtime.update(
            observation(
                100,
                facts={"box.state": "open"},
                interactions={"hand:box"},
                confident=False,
            )
        )
        self.assertEqual(
            JudgementStatus.WAITING, snapshot.statuses["open_box"]
        )
        self.assertEqual(
            JudgementStatus.UNCERTAIN, snapshot.judgements[-1].status
        )

    def test_interruption_cancels_in_progress_confirmation(self) -> None:
        self.runtime.update(observation(0))
        snapshot = self.runtime.update(
            observation(
                100,
                facts={"box.state": "open"},
                interactions={"hand:box"},
            )
        )
        self.assertEqual(
            JudgementStatus.IN_PROGRESS, snapshot.statuses["open_box"]
        )

        snapshot = self.runtime.update(
            observation(200, facts={"box.state": "open"}, healthy=False)
        )
        self.assertEqual(
            JudgementStatus.WAITING, snapshot.statuses["open_box"]
        )

        self.runtime.update(observation(300, facts={"box.state": "open"}))
        snapshot = self.runtime.update(
            observation(800, facts={"box.state": "open"})
        )
        self.assertEqual(
            JudgementStatus.WAITING, snapshot.statuses["open_box"]
        )


if __name__ == "__main__":
    unittest.main()
