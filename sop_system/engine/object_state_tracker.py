"""
ObjectStateTracker — per-object state machine for SOP item placement.

Tracks each object (earphone, charger, green_bag) through its lifecycle:

  INIT → VISIBLE_IN_INIT → LEFT_INIT → VISIBLE_IN_BOX → STABLE_IN_BOX → CONFIRMED
                                                       ↘ OCCLUDED → LOST

Key invariant: "Disappearance ≠ InBox" — losing track of an object
does NOT mean it was placed in the box. Only positive detection of the
object inside the box for consecutive frames counts.

init_roi is the region where the object starts (typically outside the box).
box_inner_roi is the region inside the box (shrunken box bbox).
"""
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field


class ObjectState(Enum):
    INIT = auto()              # never seen
    VISIBLE_IN_INIT = auto()   # seen in initial ROI
    LEFT_INIT = auto()         # left initial ROI (picked up)
    VISIBLE_IN_BOX = auto()    # seen inside box
    STABLE_IN_BOX = auto()     # stable inside box for N frames
    CONFIRMED = auto()         # event fired — done
    OCCLUDED = auto()          # temporarily lost (hand blocking)
    LOST = auto()              # lost for too long


STATE_NAMES = {s: s.name for s in ObjectState}


@dataclass
class _ObjectRecord:
    name: str
    state: ObjectState = ObjectState.INIT
    # Counters
    visible_frames: int = 0        # total frames visible
    in_init_frames: int = 0        # frames visible in initial ROI
    in_box_frames: int = 0         # consecutive frames center-in-box
    lost_frames: int = 0           # consecutive frames not detected
    occluded_frames: int = 0       # consecutive frames occluded (hand near)
    stable_box_frames: int = 0     # consecutive frames stable in box
    # Booleans
    init_seen: bool = False        # ever seen in initial ROI (or relaxed: seen outside box)
    stable_in_init: bool = False   # confirmed in init_roi for outside_box_min frames
    left_init_roi: bool = False    # left initial ROI
    was_outside_box: bool = False  # seen outside box at least once
    confirmed: bool = False        # event emitted
    # Current state
    visible: bool = False          # detected this frame
    current_bbox: Optional[Tuple] = None
    current_conf: float = 0.0
    in_box_roi: bool = False       # center inside box_inner_roi this frame
    in_init_roi: bool = False      # center inside init_roi this frame
    # Relaxed init: count frames where object is seen outside box but not in init_roi
    seen_outside_box_frames: int = 0
    # Hand-absent tracking: frames since hand left box area while object is in box
    hand_absent_frames: int = 0
    # Timing
    state_entered_at: float = 0.0
    frames_in_state: int = 0


class ObjectStateTracker:
    """Per-object state machine for SOP item placement detection.

    Tracks earphone, charger, green_bag independently.
    Each object must go through: init_seen → left_init → in_box → stable → confirmed.
    """

    def __init__(self,
                 stable_in_box_min: int = 10,
                 occluded_max: int = 8,
                 lost_max: int = 30,
                 outside_box_min: int = 5,
                 hand_absent_min: int = 8):
        self.stable_in_box_min = stable_in_box_min
        self.occluded_max = occluded_max
        self.lost_max = lost_max
        self.outside_box_min = outside_box_min
        self.hand_absent_min = hand_absent_min

        self._objects: Dict[str, _ObjectRecord] = {}
        for name in ["earphone", "charger", "green_bag"]:
            self._objects[name] = _ObjectRecord(name=name)

    def reset(self):
        for name in self._objects:
            self._objects[name] = _ObjectRecord(name=name)

    def reset_object(self, obj_name: str):
        """Soft-reset a single object's tracking state.

        Clears the "has been moved" latch (left_init_roi) and completion
        state, but PRESERVES init_seen and stable_in_init.  This way the
        object does not need to re-accumulate 5 frames of init position
        before it can be picked up — the user can act immediately after
        the previous step completes.
        """
        rec = self._objects.get(obj_name)
        if rec is None:
            return
        rec.left_init_roi = False
        rec.confirmed = False
        rec.stable_box_frames = 0
        rec.hand_absent_frames = 0
        rec.in_box_frames = 0
        rec.lost_frames = 0
        rec.occluded_frames = 0
        if rec.init_seen:
            rec.state = ObjectState.VISIBLE_IN_INIT
        else:
            rec.state = ObjectState.INIT

    def update(self,
               obj_name: str,
               detected: bool,
               bbox: Optional[Tuple],
               conf: float,
               box_inner_roi: Optional[Tuple] = None,
               init_roi: Optional[Tuple] = None,
               hand_in_box: bool = False):
        """Update one object's state for this frame.

        Args:
            obj_name: "earphone", "charger", or "green_bag"
            detected: True if YOLO detected this object this frame
            bbox: [x1,y1,x2,y2] if detected else None
            conf: detection confidence
            box_inner_roi: [x1,y1,x2,y2] of box interior (shrunken)
            init_roi: [x1,y1,x2,y2] of initial position region
            hand_in_box: True if ANY hand bbox overlaps box ROI (user still manipulating)
        """
        rec = self._objects.get(obj_name)
        if rec is None:
            return

        rec.frames_in_state += 1

        # Compute spatial relations
        in_box = False
        in_init = False
        if detected and bbox is not None:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if box_inner_roi is not None:
                bx1, by1, bx2, by2 = box_inner_roi
                in_box = (bx1 <= cx <= bx2 and by1 <= cy <= by2)

            if init_roi is not None:
                ix1, iy1, ix2, iy2 = init_roi
                in_init = (ix1 <= cx <= ix2 and iy1 <= cy <= iy2)
            else:
                # When init_roi is None, "init" = outside the box.
                # Object must be seen OUTSIDE the box before it can enter.
                # This prevents objects already in the box from auto-firing.
                in_init = not in_box

        rec.visible = detected
        rec.current_bbox = bbox
        rec.current_conf = conf
        rec.in_box_roi = in_box
        rec.in_init_roi = in_init

        if detected:
            rec.visible_frames += 1
            rec.lost_frames = 0
            rec.occluded_frames = 0

            if in_init:
                rec.in_init_frames += 1
                if rec.in_init_frames >= self.outside_box_min:
                    rec.init_seen = True
                    rec.stable_in_init = True  # confirmed in configured init_roi
            else:
                rec.in_init_frames = 0

            # Relaxed fallback: if object is seen outside the box (but not in init_roi)
            # for enough frames, accept it as "in initial position" for init_seen.
            # This handles misconfigured or absent init_rois.
            # NOTE: does NOT set stable_in_init — that requires explicit init_roi
            # confirmation. Objects that only pass the relaxed fallback cannot
            # trigger the LEFT_INIT transition (they go straight to VISIBLE_IN_BOX
            # when placed in the box).
            if not in_box:
                rec.seen_outside_box_frames += 1
                if rec.seen_outside_box_frames >= self.outside_box_min:
                    rec.init_seen = True
            else:
                rec.seen_outside_box_frames = 0

            if not in_box:
                rec.was_outside_box = True
            if in_box:
                rec.in_box_frames += 1
            else:
                rec.in_box_frames = max(0, rec.in_box_frames - 1)
        else:
            rec.lost_frames += 1
            rec.in_box_frames = max(0, rec.in_box_frames - 1)
            # Don't reset in_init_frames on brief loss (hand occlusion)
            if hand_in_box:
                rec.occluded_frames += 1

        # Track hand-absent frames: user must release object for event to fire.
        # Preserve counter during brief occlusion when object was already in box.
        was_in_box_or_occluded = rec.state in (
            ObjectState.VISIBLE_IN_BOX, ObjectState.STABLE_IN_BOX, ObjectState.OCCLUDED)
        if in_box and rec.left_init_roi:
            if not hand_in_box:
                rec.hand_absent_frames += 1
            else:
                rec.hand_absent_frames = 0
        elif was_in_box_or_occluded:
            # Object was in box but temporarily not visible (occlusion).
            # Preserve hand_absent counter — don't reset to 0.
            if not hand_in_box:
                rec.hand_absent_frames += 1
            else:
                rec.hand_absent_frames = 0
        else:
            rec.hand_absent_frames = 0

        # ── State machine ──
        prev_state = rec.state

        if rec.state == ObjectState.INIT:
            if rec.init_seen and detected:
                rec.state = ObjectState.VISIBLE_IN_INIT
            elif in_box and detected and rec.was_outside_box:
                # Fast path only after outside evidence exists. Seeing an
                # object already in the box is not enough for S2/S3/S4.
                rec.init_seen = True
                rec.stable_in_init = True
                rec.left_init_roi = True
                rec.state = ObjectState.VISIBLE_IN_BOX
                rec.stable_box_frames = 1
            elif not in_init and detected and not in_box:
                # Fast path: object picked up before init_seen accumulated.
                # Use relaxed fallback count to fast-track init_seen.
                if rec.seen_outside_box_frames >= self.outside_box_min:
                    rec.init_seen = True
                    rec.stable_in_init = True
                    rec.left_init_roi = True
                    rec.state = ObjectState.LEFT_INIT

        elif rec.state == ObjectState.VISIBLE_IN_INIT:
            if in_box and detected and rec.init_seen:
                # Object entering box directly (relaxed mode or init_roi=box)
                rec.left_init_roi = True
                rec.state = ObjectState.VISIBLE_IN_BOX
                rec.stable_box_frames = 1
            elif not in_init and detected and rec.init_seen and rec.stable_in_init:
                # Object left its init position (explicit init_roi mode).
                # REQUIRES stable_in_init: the object must have been confirmed
                # in its init_roi for outside_box_min frames BEFORE leaving.
                # This prevents "not in init_roi at startup" from being
                # mistaken for "user picked up the object".
                rec.left_init_roi = True
                rec.state = ObjectState.LEFT_INIT
            elif not detected and rec.lost_frames > self.occluded_max:
                rec.state = ObjectState.OCCLUDED
            elif not detected and rec.lost_frames > self.lost_max:
                rec.state = ObjectState.LOST

        elif rec.state == ObjectState.LEFT_INIT:
            if in_box and detected:
                rec.left_init_roi = True  # ensure set (may have been False from OCCLUDED recovery)
                rec.state = ObjectState.VISIBLE_IN_BOX
                rec.stable_box_frames = 1
            elif not detected and rec.lost_frames > self.occluded_max:
                rec.state = ObjectState.OCCLUDED
            elif detected and in_init and rec.in_init_frames >= 3:
                # Returned to init (put back outside box) — reset
                rec.state = ObjectState.VISIBLE_IN_INIT
                rec.left_init_roi = False

        elif rec.state == ObjectState.VISIBLE_IN_BOX:
            if in_box and detected:
                rec.stable_box_frames += 1
                if rec.stable_box_frames >= self.stable_in_box_min:
                    rec.state = ObjectState.STABLE_IN_BOX
            elif not in_box and detected:
                # Object moved out of box — back to LEFT_INIT
                rec.state = ObjectState.LEFT_INIT
                rec.stable_box_frames = 0
            elif not detected:
                rec.state = ObjectState.OCCLUDED

        elif rec.state == ObjectState.STABLE_IN_BOX:
            if in_box and detected:
                rec.stable_box_frames += 1
            elif not in_box and detected:
                rec.state = ObjectState.LEFT_INIT
                rec.stable_box_frames = 0
            elif not detected:
                rec.state = ObjectState.OCCLUDED

        elif rec.state == ObjectState.OCCLUDED:
            if detected:
                rec.occluded_frames = 0
                if in_box:
                    # Was occluded while in/near box — restore in-box state.
                    # Set left_init_roi=True: if the object is in the box,
                    # it must have left its init position.
                    rec.left_init_roi = True
                    rec.state = ObjectState.VISIBLE_IN_BOX
                    rec.stable_box_frames = max(1, rec.stable_box_frames)
                elif in_init and rec.in_init_frames >= 3:
                    rec.state = ObjectState.VISIBLE_IN_INIT
                else:
                    rec.state = ObjectState.LEFT_INIT
            elif rec.occluded_frames > self.lost_max:
                rec.state = ObjectState.LOST

        elif rec.state == ObjectState.LOST:
            if detected:
                rec.state = ObjectState.INIT  # complete reset
                rec.lost_frames = 0

        elif rec.state == ObjectState.CONFIRMED:
            pass  # terminal state

        if rec.state != prev_state:
            rec.frames_in_state = 0

    def _release_or_settled(self, rec: _ObjectRecord) -> bool:
        """Return True when placement is released or visibly settled."""
        if rec.hand_absent_frames >= self.hand_absent_min:
            return True
        settled_frames = self.stable_in_box_min + max(self.hand_absent_min, 8)
        return rec.stable_box_frames >= settled_frames

    def is_ready_for_event(self, obj_name: str) -> bool:
        """Check if object satisfies ALL conditions for in_box event.

        CRITICAL: hand must have LEFT the box area after placing.
        Object must be stable in box AND hand absent for consecutive frames.
        This prevents "action still in progress → already green" false positives.
        """
        rec = self._objects.get(obj_name)
        if rec is None:
            return False
        return (
            rec.init_seen
            and rec.left_init_roi
            and rec.visible
            and rec.in_box_roi
            and rec.was_outside_box
            and rec.stable_box_frames >= self.stable_in_box_min
            and self._release_or_settled(rec)
        )

    def get_ready_conditions(self, obj_name: str) -> Dict:
        """Get detailed condition status for debugging."""
        rec = self._objects.get(obj_name)
        if rec is None:
            return {}
        return {
            "init_seen": rec.init_seen,
            "left_init_roi": rec.left_init_roi,
            "visible": rec.visible,
            "in_box_roi": rec.in_box_roi,
            "was_outside_box": rec.was_outside_box,
            "stable_box_frames": rec.stable_box_frames,
            "stable_needed": self.stable_in_box_min,
            "hand_absent_frames": rec.hand_absent_frames,
            "hand_absent_needed": self.hand_absent_min,
            "release_or_settled": self._release_or_settled(rec),
            "ready": self.is_ready_for_event(obj_name),
            "state": STATE_NAMES.get(rec.state, "?"),
        }

    def mark_confirmed(self, obj_name: str):
        """Mark object as confirmed (event emitted)."""
        rec = self._objects.get(obj_name)
        if rec:
            rec.confirmed = True
            rec.state = ObjectState.CONFIRMED

    def get_state(self, obj_name: str) -> ObjectState:
        rec = self._objects.get(obj_name)
        return rec.state if rec else ObjectState.INIT

    def is_visible(self, obj_name: str) -> bool:
        rec = self._objects.get(obj_name)
        return rec.visible if rec else False

    def get_bbox(self, obj_name: str) -> Optional[Tuple]:
        rec = self._objects.get(obj_name)
        return rec.current_bbox if rec else None

    def get_summary(self) -> Dict:
        """Get summary of all objects for debug/UI."""
        result = {}
        for name, rec in self._objects.items():
            result[name] = {
                "state": STATE_NAMES.get(rec.state, "?"),
                "visible": rec.visible,
                "init_seen": rec.init_seen,
                "stable_in_init": rec.stable_in_init,
                "left_init_roi": rec.left_init_roi,
                "in_box_roi": rec.in_box_roi,
                "in_init_roi": rec.in_init_roi,
                "stable_box_frames": rec.stable_box_frames,
                "hand_absent_frames": rec.hand_absent_frames,
                "release_or_settled": self._release_or_settled(rec),
                "lost_frames": rec.lost_frames,
                "ready": self.is_ready_for_event(name),
            }
        return result
