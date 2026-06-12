"""
ActionSegmenter — tracks which SOP action is currently being performed.

Outputs:
  - current_action: S1_open_box / S2_put_earphone / S3_put_charger / S4_put_green_bag / S5_close_box
  - action_phase: WAITING / ACTIVE / COMPLETING / DONE / WRONG_ACTIVE / ABORTED
  - action_start_time, action_duration

Actions are driven by expected_event and physical observations.
"""
import time
from typing import Dict, List, Optional
from enum import Enum, auto
from dataclasses import dataclass, field


class ActionPhase(Enum):
    WAITING = auto()       # waiting for action to begin
    ACTIVE = auto()        # action is being performed
    COMPLETING = auto()    # action nearly complete
    DONE = auto()          # action complete
    WRONG_ACTIVE = auto()  # wrong action detected
    ABORTED = auto()       # action started but aborted


class ActionName(Enum):
    S1_OPEN_BOX = "S1_open_box"
    S2_PUT_EARPHONE = "S2_put_earphone"
    S3_PUT_CHARGER = "S3_put_charger"
    S4_PUT_GREEN_BAG = "S4_put_green_bag"
    S5_CLOSE_BOX = "S5_close_box"


# Mapping from expected_event to action
EVENT_TO_ACTION = {
    "box_opened": ActionName.S1_OPEN_BOX,
    "earphone_in_box": ActionName.S2_PUT_EARPHONE,
    "charger_in_box": ActionName.S3_PUT_CHARGER,
    "green_bag_in_box": ActionName.S4_PUT_GREEN_BAG,
    "box_closed": ActionName.S5_CLOSE_BOX,
}

OBJECT_FOR_EVENT = {
    "earphone_in_box": "earphone",
    "charger_in_box": "charger",
    "green_bag_in_box": "green_bag",
}


@dataclass
class ActionSegment:
    action_name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    expected_event: str = ""
    result: str = ""       # "ok" / "wrong_order" / "aborted"
    error_type: str = ""


class ActionSegmenter:
    """Tracks current action and phase based on physical observations.

    Rules:
    - Action becomes ACTIVE when physical evidence of that action is observed
      (e.g., earphone leaving init ROI when expected_event == earphone_in_box)
    - Action becomes COMPLETING when in_box conditions are nearly met
    - Action becomes DONE when the event fires
    - WRONG_ACTIVE: wrong object moving toward box when expected_event differs
    - ABORTED: action was active but object returned to init or disappeared
    """

    def __init__(self):
        self.current_action: Optional[ActionName] = None
        self.current_phase: ActionPhase = ActionPhase.WAITING
        self._action_start_time: float = 0.0
        self._wrong_object: Optional[str] = None

        # Segment log
        self.segments: List[ActionSegment] = []
        self._current_segment: Optional[ActionSegment] = None

    def reset(self):
        self.current_action = None
        self.current_phase = ActionPhase.WAITING
        self._action_start_time = 0.0
        self._wrong_object = None
        self.segments.clear()
        self._current_segment = None

    def update(self,
               expected_event: str,
               box_state_str: str,
               box_previous_state_str: str,
               obj_tracker_summary: Dict,
               accepted_events: List[str],
               now: float = None,
               open_evidence: int = 0,
               closed_evidence: int = 0) -> Dict:
        """Update action segmentation for this frame.

        Args:
            expected_event: current expected event name
            box_state_str: current box state ("open", "closed", etc.)
            box_previous_state_str: previous box state
            obj_tracker_summary: from ObjectStateTracker.get_summary()
            accepted_events: list of already-accepted event names
            now: current timestamp

        Returns:
            dict with current_action, action_phase, action_duration, etc.
        """
        if now is None:
            now = time.time()

        action = EVENT_TO_ACTION.get(expected_event)
        if action is None:
            return self._result(now)

        # ── Determine phase ──
        new_phase = self.current_phase

        if action == ActionName.S1_OPEN_BOX:
            new_phase = self._phase_for_open_box(box_state_str, box_previous_state_str, accepted_events, open_evidence)
        elif action == ActionName.S5_CLOSE_BOX:
            new_phase = self._phase_for_close_box(box_state_str, box_previous_state_str, accepted_events, closed_evidence)
        else:
            # S2/S3/S4 — object placement
            obj_name = OBJECT_FOR_EVENT.get(expected_event, "")
            obj = obj_tracker_summary.get(obj_name, {})
            new_phase = self._phase_for_placement(obj, expected_event, accepted_events, obj_tracker_summary)

        # ── Detect WRONG_ACTIVE ──
        wrong_active = False
        if new_phase != ActionPhase.DONE and action != ActionName.S1_OPEN_BOX and action != ActionName.S5_CLOSE_BOX:
            # Check if a LATER object is moving toward box (wrong order)
            later_objects = self._get_later_objects(expected_event)
            for later_obj in later_objects:
                later = obj_tracker_summary.get(later_obj, {})
                if later.get("left_init_roi") and later.get("in_box_roi"):
                    wrong_active = True
                    self._wrong_object = later_obj
                    break

        if wrong_active and new_phase != ActionPhase.DONE:
            new_phase = ActionPhase.WRONG_ACTIVE

        # ── State transitions ──
        if new_phase != self.current_phase:
            if new_phase == ActionPhase.ACTIVE and self.current_phase == ActionPhase.WAITING:
                self._action_start_time = now
                self._current_segment = ActionSegment(
                    action_name=action.value,
                    start_time=now,
                    expected_event=expected_event,
                )
            elif new_phase == ActionPhase.DONE and self._current_segment is not None:
                self._current_segment.end_time = now
                self._current_segment.duration = now - self._current_segment.start_time
                self._current_segment.result = "ok"
                self.segments.append(self._current_segment)
                self._current_segment = None
            elif new_phase == ActionPhase.ABORTED and self._current_segment is not None:
                self._current_segment.end_time = now
                self._current_segment.duration = now - self._current_segment.start_time
                self._current_segment.result = "aborted"
                self.segments.append(self._current_segment)
                self._current_segment = None

        # Only expose current_action when the action is actually happening.
        # When WAITING or DONE, the user hasn't started the next action yet —
        # showing it would cause the next step box to falsely highlight yellow.
        if new_phase in (ActionPhase.WAITING, ActionPhase.DONE, ActionPhase.ABORTED):
            self.current_action = None
        else:
            self.current_action = action
        self.current_phase = new_phase

        return self._result(now)

    def _phase_for_open_box(self, box_state_str, box_prev_str, accepted, open_evidence=0):
        if "box_opened" in accepted:
            return ActionPhase.DONE
        if box_state_str == "open":
            return ActionPhase.COMPLETING
        # COMPLETING: was closed, now transition (closed disappeared = box is open)
        if box_state_str == "transition" and box_prev_str == "closed":
            return ActionPhase.COMPLETING
        # ACTIVE: transition state OR open_evidence building
        if box_state_str == "transition":
            return ActionPhase.ACTIVE
        if open_evidence >= 1:
            return ActionPhase.ACTIVE
        return ActionPhase.WAITING

    def _phase_for_close_box(self, box_state_str, box_prev_str, accepted, closed_evidence=0):
        if "box_closed" in accepted:
            return ActionPhase.DONE
        # COMPLETING: box is CLOSED and stable for at least 8 frames.
        # This prevents premature green before the box is truly closed.
        if box_state_str == "closed" and closed_evidence >= 8:
            return ActionPhase.COMPLETING
        # transition from open → closing (genuine transition evidence)
        if box_state_str == "transition" and box_prev_str == "open":
            return ActionPhase.ACTIVE
        # Fix 1B: A bare "transition" state is NOT enough to go ACTIVE.
        # Require at least some positive closing evidence (closed_evidence >= 1).
        # Without this guard, "transition" caused by 0/0 confidences (no one
        # touching the box) would prematurely turn the S5 bar yellow.
        if box_state_str == "transition" and closed_evidence >= 1:
            return ActionPhase.ACTIVE
        # Box is detected as closed but not yet stable enough for COMPLETING
        if box_state_str == "closed":
            return ActionPhase.ACTIVE
        if closed_evidence >= 1:
            return ActionPhase.ACTIVE
        return ActionPhase.WAITING

    def _phase_for_placement(self, obj: Dict, expected_event: str,
                              accepted: List[str], all_objs: Dict) -> ActionPhase:
        """Determine action phase from object state machine state.

        Uses ObjectStateTracker's state directly instead of fragile flag
        combinations. The state machine already encodes the object's
        lifecycle (INIT → VISIBLE_IN_INIT → LEFT_INIT → VISIBLE_IN_BOX →
        STABLE_IN_BOX → CONFIRMED), which maps cleanly to action phases.
        """
        if expected_event in accepted:
            return ActionPhase.DONE
        obj_state = obj.get("state", "INIT")
        stable = obj.get("stable_box_frames", 0)
        visible = obj.get("visible", False)
        in_box = obj.get("in_box_roi", False)

        # COMPLETING: object is stable inside the box
        if obj_state in ("STABLE_IN_BOX",) and stable >= 5:
            return ActionPhase.COMPLETING
        if obj_state in ("VISIBLE_IN_BOX",) and stable >= 3:
            return ActionPhase.COMPLETING

        # ACTIVE: object has left its init position or is entering the box
        if obj_state in ("LEFT_INIT", "VISIBLE_IN_BOX"):
            return ActionPhase.ACTIVE

        # Fallback ACTIVE: object is in box (by spatial check) but state
        # machine hasn't caught up yet (e.g. just after reset)
        if in_box and visible:
            return ActionPhase.ACTIVE

        return ActionPhase.WAITING

    def _get_later_objects(self, expected_event: str) -> List[str]:
        """Get objects that should come AFTER the expected event."""
        order = ["earphone_in_box", "charger_in_box", "green_bag_in_box"]
        obj_map = {
            "earphone_in_box": "earphone",
            "charger_in_box": "charger",
            "green_bag_in_box": "green_bag",
        }
        try:
            idx = order.index(expected_event)
        except ValueError:
            return []
        return [obj_map[e] for e in order[idx + 1:]]

    def _result(self, now: float) -> Dict:
        return {
            "current_action": self.current_action.value if self.current_action else "none",
            "action_phase": self.current_phase.name,
            "action_start_time": round(self._action_start_time, 3),
            "action_duration": round(now - self._action_start_time, 3) if self._action_start_time > 0 else 0.0,
            "wrong_object": self._wrong_object,
        }
