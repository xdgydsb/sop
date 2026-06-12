"""
EventDetector — outputs ONLY standard SOP events based on physical conditions.

Events: box_opened, earphone_in_box, charger_in_box, green_bag_in_box, box_closed
         + early_close_alarm (safety alarm)

S1 (box_opened) uses a HAND-AWARE state machine because "closed detection
disappeared" can mean EITHER "hand is occluding the box" OR "box is actually open".
The state machine requires: hand enters box area → hand leaves → closed stays
absent for N frames → S1 fires. This filters out brief hand occlusion.

NEVER uses TemporalPredictor argmax for decisions.
NEVER uses raw YOLO class names directly.
"""
import time
import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

from engine.box_state_stabilizer import BoxState


# ── Standard event names ──
EVENT_BOX_OPENED = "box_opened"
EVENT_EARPHONE_IN_BOX = "earphone_in_box"
EVENT_CHARGER_IN_BOX = "charger_in_box"
EVENT_GREEN_BAG_IN_BOX = "green_bag_in_box"
EVENT_BOX_CLOSED = "box_closed"
EVENT_EARLY_CLOSE_ALARM = "early_close_alarm"

ALL_EVENTS = [
    EVENT_BOX_OPENED, EVENT_EARPHONE_IN_BOX, EVENT_CHARGER_IN_BOX,
    EVENT_GREEN_BAG_IN_BOX, EVENT_BOX_CLOSED, EVENT_EARLY_CLOSE_ALARM,
]

EVENT_TO_STEP_ID = {
    EVENT_BOX_OPENED: 1,
    EVENT_EARPHONE_IN_BOX: 2,
    EVENT_CHARGER_IN_BOX: 3,
    EVENT_GREEN_BAG_IN_BOX: 4,
    EVENT_BOX_CLOSED: 5,
    EVENT_EARLY_CLOSE_ALARM: 7,
}

PLACED_OBJECTS = ["earphone", "charger", "green_bag"]
PLACED_EVENTS = [EVENT_EARPHONE_IN_BOX, EVENT_CHARGER_IN_BOX, EVENT_GREEN_BAG_IN_BOX]

EVENT_SEQUENCE = [
    EVENT_BOX_OPENED,
    EVENT_EARPHONE_IN_BOX,
    EVENT_CHARGER_IN_BOX,
    EVENT_GREEN_BAG_IN_BOX,
    EVENT_BOX_CLOSED,
]


@dataclass
class DetectedEvent:
    event_name: str
    confidence: float
    conditions_met: List[str]
    rejected_reason: str
    temporal_conf_boost: float
    timestamp: float


class EventDetector:
    """Event-based SOP detection with strict physical conditions.

    S1 (box_opened) — hand-aware state machine:
      idle → hand_near → hand_left → confirmed
      Requires: box was closed → hand entered box area → hand left →
      closed absent for CONFIRM frames → S1 fires.
      If closed EVER returns during hand_left phase → back to idle.
      Fallback: if hand is never detected but closed is absent for FALLBACK
      frames, S1 fires anyway (hand detection may have failed).

    S2/S3/S4 — delegated to ObjectStateTracker get_ready_conditions()

    S5 (box_closed) — box was open + now closed for N frames + S2/S3/S4 done

    early_close_alarm — box closed for MANY frames but S2/S3/S4 not done
    """

    def __init__(self,
                 open_stable_frames: int = 3,
                 closed_stable_frames: int = 15,
                 early_close_confirm_frames: int = 25,
                 event_cooldown: int = 30,
                 debug_dir: Optional[str] = None):
        # S5 / alarm thresholds (Fix 2B: reduced from 30→15 for faster S5)
        self.closed_confirm_frames = closed_stable_frames
        self.early_close_confirm_frames = early_close_confirm_frames
        self.inter_event_gap = max(event_cooldown, 5)

        # S1 is "hand opens a closed box". Missing/occluded box_closed is not
        # enough; a real box_open detection must appear after hand contact.
        self._s1_open_min_conf: float = 0.45
        self._s1_open_margin: float = 0.20
        self._s1_open_frames: int = open_stable_frames

        # Fix 2B: Reduced S5 confirmation thresholds
        # S5 completion: hand must leave box area after closing
        self._s5_hand_absent_needed: int = 8
        # Fix 2C: Fast-path thresholds (stabilizer-confirmed closed → lower bar)
        self._s5_fast_closed_needed: int = 10
        self._s5_fast_hand_absent_needed: int = 5

        # One-shot guard: events accepted by server
        self._emitted: Set[str] = set()

        # ── Consecutive-frame counters ──
        self._open_consec: int = 0
        self._closed_consec: int = 0
        self._box_was_closed: bool = False
        self._box_was_open: bool = False

        # Fix 2B: Hysteresis — tolerate brief YOLO flicker without resetting counters
        self._closed_lost_frames: int = 0
        self._closed_lost_tolerance: int = 3    # allow up to 3 consecutive missed frames
        self._closed_hysteresis_threshold: int = 5  # only apply hysteresis after this many frames

        # ── S1 state machine ──
        # idle → hand_was_in_box → hand_left_closed_absent → S1 fires
        self._s1_hand_was_in_box: bool = False
        self._s1_frames_since_hand_left: int = 0   # frames since hand left, closed absent
        self._s1_closed_absent_total: int = 0       # total frames closed NOT detected (hand-free fallback)

        # ── S5 hand-absent tracking ──
        self._s5_hand_absent_frames: int = 0

        self._last_any_event_frame: int = -999

        # Debug
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._debug_buffer: List[Dict] = []
        self._frame_count: int = 0
        self._debug_f = None
        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            self._debug_f = open(str(self._debug_dir / "debug_trace.jsonl"), "w", encoding="utf-8")

    def reset(self):
        self._emitted.clear()
        self._open_consec = 0
        self._closed_consec = 0
        self._box_was_closed = False
        self._box_was_open = False
        self._s1_hand_was_in_box = False
        self._s1_frames_since_hand_left = 0
        self._s1_closed_absent_total = 0
        self._s5_hand_absent_frames = 0
        self._last_any_event_frame = -999
        self._frame_count = 0
        self._debug_buffer.clear()
        # Fix 2B: reset hysteresis counter
        self._closed_lost_frames = 0

    # ═══════════════════════════════════════════════════════════════════
    # PREVIEW priming
    # ═══════════════════════════════════════════════════════════════════

    def update_evidence(self, box_state_info: Dict):
        """Prime history flags during PREVIEW mode.

        Only sets _box_was_closed / _box_was_open. Does NOT update counters,
        so counter accumulation during PREVIEW won't cause instant firing
        when mode switches to ARMED.
        """
        open_conf = float(box_state_info.get("open_conf", 0))
        closed_conf = float(box_state_info.get("closed_conf", 0))
        open_bbox = box_state_info.get("open_bbox")
        closed_bbox = box_state_info.get("closed_bbox")

        if closed_conf >= 0.35 and closed_bbox is not None:
            self._box_was_closed = True
        if open_conf >= 0.25 and open_bbox is not None:
            self._box_was_open = True

    # ═══════════════════════════════════════════════════════════════════
    # Main entry point
    # ═══════════════════════════════════════════════════════════════════

    def detect(self,
               box_state: BoxState,
               box_state_info: Dict,
               obj_tracker,
               expected_event: str,
               accepted_events: List[str],
               temporal_aux: Optional[Dict] = None,
               hand_in_box: bool = False) -> Optional[DetectedEvent]:
        now = time.time()
        self._frame_count += 1
        accepted_set = set(accepted_events)

        # Update raw counters
        self._update_counters(box_state_info)

        # Update S1 state machine
        closed_detected = self._closed_consec > 0
        self._update_s1_state(hand_in_box, closed_detected)

        # Inter-event gap
        gap_since_last = self._frame_count - self._last_any_event_frame
        in_cooldown = (self._last_any_event_frame > 0 and gap_since_last < self.inter_event_gap)

        # Temporal boost — rewards temporal agreement, never gates events.
        # Top-1 match = strong agreement (+0.10). Top-3 match = moderate (+0.05).
        # Temporal disagreement = no boost but event still fires on physical evidence.
        temporal_conf_boost = 0.0
        if temporal_aux is not None:
            step_probs = temporal_aux.get("step_probs")
            target_step = EVENT_TO_STEP_ID.get(expected_event, -1)
            if step_probs is not None and 0 <= target_step < len(step_probs):
                top3 = temporal_aux.get("top3", [])
                top1_step = int(top3[0][0]) if top3 else -1
                top3_steps = [int(i) for i, _ in top3]
                if top1_step == target_step:
                    temporal_conf_boost = 0.10
                elif target_step in top3_steps:
                    temporal_conf_boost = 0.05

        event_rejected = ""

        # ── 1. early_close_alarm ──
        if expected_event != EVENT_BOX_CLOSED:
            early = self._check_early_close(accepted_set, hand_in_box=hand_in_box)
            if early and not in_cooldown:
                return self._emit(early, temporal_conf_boost, now)

        # ── 2. box_opened ──
        if expected_event == EVENT_BOX_OPENED:
            event = self._check_box_opened(accepted_set)
            if event:
                if in_cooldown:
                    event_rejected = f"cooldown({gap_since_last}/{self.inter_event_gap})"
                else:
                    return self._emit(event, temporal_conf_boost, now)
            else:
                event_rejected = self._box_open_reject_reason()

        # ── 3. object placement ──
        elif expected_event in (EVENT_EARPHONE_IN_BOX, EVENT_CHARGER_IN_BOX, EVENT_GREEN_BAG_IN_BOX):
            obj_name = {
                EVENT_EARPHONE_IN_BOX: "earphone",
                EVENT_CHARGER_IN_BOX: "charger",
                EVENT_GREEN_BAG_IN_BOX: "green_bag",
            }[expected_event]
            event = self._check_object_in_box(expected_event, obj_name, obj_tracker, accepted_set)
            if event:
                if in_cooldown:
                    event_rejected = f"cooldown({gap_since_last}/{self.inter_event_gap})"
                else:
                    return self._emit(event, temporal_conf_boost, now)

        # ── 4. box_closed ──
        elif expected_event == EVENT_BOX_CLOSED:
            # Fix 2C: Update hand-absent tracking (shared by fast and main paths).
            # Moved here from _check_box_closed so both paths share the same counter.
            if self._closed_consec > 0 and not hand_in_box:
                self._s5_hand_absent_frames += 1
            elif hand_in_box:
                self._s5_hand_absent_frames = 0

            # Fix 2C: Fast-path — fire based on strong physical evidence alone,
            # bypassing temporal model entirely.
            event = self._check_box_closed_fast(box_state, accepted_set, hand_in_box)
            if event:
                if in_cooldown:
                    event_rejected = f"cooldown({gap_since_last}/{self.inter_event_gap})"
                else:
                    return self._emit(event, temporal_conf_boost, now)

            # Main path: standard physical checks, temporal model advisory only
            event = self._check_box_closed(accepted_set, hand_in_box=hand_in_box,
                                           temporal_aux=temporal_aux)
            if event:
                if in_cooldown:
                    event_rejected = f"cooldown({gap_since_last}/{self.inter_event_gap})"
                else:
                    return self._emit(event, temporal_conf_boost, now)
            else:
                event_rejected = self._box_close_reject_reason(accepted_set)

        if self._debug_f is not None and self._frame_count % 5 == 0:
            self._write_debug(now, expected_event, event_rejected, box_state_info)

        return None

    # ═══════════════════════════════════════════════════════════════════
    # Counter updates
    # ═══════════════════════════════════════════════════════════════════

    def _update_counters(self, box_state_info: Dict):
        """Update raw consecutive-frame counters from YOLO confidence."""
        open_conf = float(box_state_info.get("open_conf", 0))
        closed_conf = float(box_state_info.get("closed_conf", 0))
        open_bbox = box_state_info.get("open_bbox")
        closed_bbox = box_state_info.get("closed_bbox")

        strong_open = (
            open_bbox is not None
            and open_conf >= self._s1_open_min_conf
            and open_conf >= closed_conf + self._s1_open_margin
        )

        if strong_open:
            self._open_consec += 1
        else:
            self._open_consec = 0

        # Fix 2B: Closed-consec counter with hysteresis.
        # Once we've accumulated some evidence, tolerate brief YOLO flicker
        # (single-frame false negatives) without resetting the counter.
        if closed_conf >= 0.35 and closed_bbox is not None:
            self._closed_consec += 1
            self._closed_lost_frames = 0
            self._s1_closed_absent_total = 0
        else:
            if self._closed_consec >= self._closed_hysteresis_threshold:
                # Hysteresis active: tolerate up to lost_tolerance consecutive misses
                self._closed_lost_frames += 1
                if self._closed_lost_frames > self._closed_lost_tolerance:
                    self._closed_consec = 0
                    self._closed_lost_frames = 0
                # else: closed_consec unchanged (tolerated flicker)
            else:
                self._closed_consec = 0
            if self._box_was_closed:
                self._s1_closed_absent_total += 1

        if closed_conf >= 0.35 and closed_bbox is not None:
            self._box_was_closed = True
        if open_conf >= 0.25 and open_bbox is not None:
            self._box_was_open = True

    # ═══════════════════════════════════════════════════════════════════
    # S1 state machine (hand-aware)
    # ═══════════════════════════════════════════════════════════════════

    def _update_s1_state(self, hand_in_box: bool, closed_detected: bool):
        """Track hand contact with a previously closed box."""
        if not self._box_was_closed:
            return  # haven't seen box closed yet, can't detect opening

        if hand_in_box:
            # Hand is in the box operation area; S1 still needs real box_open.
            self._s1_hand_was_in_box = True
            self._s1_frames_since_hand_left = 0
        elif self._s1_hand_was_in_box:
            self._s1_frames_since_hand_left += 1

    # ═══════════════════════════════════════════════════════════════════
    # Event checks
    # ═══════════════════════════════════════════════════════════════════

    def _check_box_opened(self, accepted: Set[str]) -> Optional[DetectedEvent]:
        """S1: hand changes the box from closed to open."""
        if EVENT_BOX_OPENED in accepted or EVENT_BOX_OPENED in self._emitted:
            return None
        if not self._box_was_closed:
            return None
        if not self._s1_hand_was_in_box:
            return None
        if self._open_consec >= self._s1_open_frames:
            conds = [
                "box_was_closed",
                "hand_was_in_box",
                f"strong_open_consec={self._open_consec}/{self._s1_open_frames}",
            ]
            return DetectedEvent(EVENT_BOX_OPENED, 0.92, conds, "", 0.0, time.time())

        return None

    def _box_open_reject_reason(self) -> str:
        if EVENT_BOX_OPENED in self._emitted:
            return "already_emitted"
        parts = []
        if not self._box_was_closed:
            parts.append("box_never_closed")
        if not self._s1_hand_was_in_box:
            parts.append("hand_never_in_box")
        if self._open_consec < self._s1_open_frames:
            parts.append(f"strong_open_consec={self._open_consec}/{self._s1_open_frames}")
        return "; ".join(parts) if parts else "?"

    def _check_object_in_box(self, event_name: str, obj_name: str,
                              obj_tracker, accepted: Set[str]) -> Optional[DetectedEvent]:
        if event_name in accepted or event_name in self._emitted:
            return None
        if EVENT_BOX_OPENED not in accepted:
            return None
        conds = obj_tracker.get_ready_conditions(obj_name)
        if conds.get("ready", False):
            cond_list = [f"{k}={v}" for k, v in conds.items()]
            return DetectedEvent(event_name, 0.80, cond_list, "", 0.0, time.time())
        return None

    def _check_box_closed(self, accepted: Set[str],
                           hand_in_box: bool = False,
                           temporal_aux: Optional[Dict] = None) -> Optional[DetectedEvent]:
        if EVENT_BOX_CLOSED in accepted or EVENT_BOX_CLOSED in self._emitted:
            return None
        if not self._box_was_open:
            return None
        if not all(e in accepted for e in PLACED_EVENTS):
            return None

        # NOTE: hand-absent tracking moved to detect() so both fast-path and
        # main path share the same counter.

        if self._closed_consec < self.closed_confirm_frames:
            return None
        if self._s5_hand_absent_frames < self._s5_hand_absent_needed:
            return None

        # Fix 2A: Temporal model is ADVISORY only for S5 — it can boost
        # confidence but never blocks S5. Physical evidence (closed_consec
        # + hand_absent) is sufficient.
        temporal_prob = 0.0
        temporal_agrees = False
        if temporal_aux is not None:
            step_probs = temporal_aux.get("step_probs")
            top3 = temporal_aux.get("top3", [])
            if step_probs is not None and len(step_probs) > 5:
                temporal_prob = float(step_probs[5])
                top3_steps = [int(i) for i, _ in top3]
                temporal_agrees = (5 in top3_steps or temporal_prob >= 0.12)

        # Base confidence from physical evidence
        base_conf = 0.85
        # Temporal agreement boosts confidence but does NOT gate the event
        if temporal_agrees:
            base_conf = 0.90
        elif temporal_prob > 0:
            # Temporal has some signal — small boost
            base_conf = 0.87

        conds = ["box_was_open",
                 f"closed_consec={self._closed_consec}/{self.closed_confirm_frames}",
                 f"hand_absent={self._s5_hand_absent_frames}/{self._s5_hand_absent_needed}",
                 f"temporal_step5_prob={temporal_prob:.3f}",
                 f"temporal_agrees={temporal_agrees}",
                 "S2/S3/S4_done"]
        return DetectedEvent(EVENT_BOX_CLOSED, base_conf, conds, "", 0.0, time.time())

    def _check_box_closed_fast(self, box_state,
                                accepted: Set[str],
                                hand_in_box: bool = False) -> Optional[DetectedEvent]:
        """Fix 2C: Fast-path S5 detection based on strong physical evidence alone.

        Fires when the box_state stabilizer is confidently reporting "closed"
        and the hand is absent. Uses lower thresholds than the main path
        because the stabilizer's confidence provides additional signal quality.

        This path does NOT consult the temporal model at all.
        """
        if EVENT_BOX_CLOSED in accepted or EVENT_BOX_CLOSED in self._emitted:
            return None
        if not self._box_was_open:
            return None
        if not all(e in accepted for e in PLACED_EVENTS):
            return None

        # Fast-path requires stabilizer to be confident about CLOSED state
        if box_state != BoxState.CLOSED:
            return None

        # Hand must be absent (the user has released the box)
        if hand_in_box:
            return None

        # Lower thresholds than main path — stabilizer confidence compensates
        if self._closed_consec < self._s5_fast_closed_needed:
            return None
        if self._s5_hand_absent_frames < self._s5_fast_hand_absent_needed:
            return None

        conds = ["box_was_open",
                 "fast_path: stabilizer=closed",
                 f"closed_consec={self._closed_consec}/{self._s5_fast_closed_needed}",
                 f"hand_absent={self._s5_hand_absent_frames}/{self._s5_fast_hand_absent_needed}",
                 "S2/S3/S4_done"]
        return DetectedEvent(EVENT_BOX_CLOSED, 0.92, conds, "", 0.0, time.time())

    def _box_close_reject_reason(self, accepted: Set[str]) -> str:
        if EVENT_BOX_CLOSED in self._emitted:
            return "already_emitted"
        parts = []
        if not self._box_was_open:
            parts.append("box_never_opened")
        if not all(e in accepted for e in PLACED_EVENTS):
            missing = [e for e in PLACED_EVENTS if e not in accepted]
            parts.append(f"missing={missing}")
        if self._closed_consec < self.closed_confirm_frames:
            parts.append(f"closed_consec={self._closed_consec}/{self.closed_confirm_frames}")
        if self._s5_hand_absent_frames < self._s5_hand_absent_needed:
            parts.append(f"hand_absent={self._s5_hand_absent_frames}/{self._s5_hand_absent_needed}")
        # NOTE: temporal is advisory only for S5 — physical evidence is sufficient.
        # This diagnostic remains for debugging when temporal disagrees.
        if self._closed_consec >= self.closed_confirm_frames and self._s5_hand_absent_frames >= self._s5_hand_absent_needed:
            parts.append("temporal_not_ready")
        return "; ".join(parts) if parts else "?"

    def _check_early_close(self, accepted: Set[str],
                           hand_in_box: bool = False) -> Optional[DetectedEvent]:
        # early_close only makes sense after S1 has been truly accepted.
        # Otherwise a missed/late box_opened would incorrectly collapse the
        # whole run into "close early" while the expected event is still S1.
        if EVENT_BOX_OPENED not in accepted:
            return None
        if not self._box_was_open:
            return None
        if self._closed_consec < self.early_close_confirm_frames:
            return None
        if hand_in_box:
            return None
        if all(e in accepted for e in PLACED_EVENTS):
            return None
        missing = [e for e in PLACED_EVENTS if e not in accepted]
        conds = [f"closed_consec={self._closed_consec}/{self.early_close_confirm_frames}",
                 f"missing={missing}"]
        return DetectedEvent(EVENT_EARLY_CLOSE_ALARM, 0.80, conds, "", 0.0, time.time())

    # ═══════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════

    def _emit(self, event: DetectedEvent, temporal_boost: float, now: float) -> DetectedEvent:
        event.temporal_conf_boost = round(temporal_boost, 4)
        event.confidence = round(min(event.confidence + temporal_boost, 1.0), 4)
        event.timestamp = now
        self._emitted.add(event.event_name)
        self._last_any_event_frame = self._frame_count
        return event

    def mark_event_accepted(self, event_name: str, obj_tracker=None):
        if obj_tracker is not None:
            obj_name = {
                EVENT_EARPHONE_IN_BOX: "earphone",
                EVENT_CHARGER_IN_BOX: "charger",
                EVENT_GREEN_BAG_IN_BOX: "green_bag",
            }.get(event_name)
            if obj_name is not None:
                obj_tracker.mark_confirmed(obj_name)

    def _write_debug(self, now, expected_event, reject_reason, box_info):
        entry = {
            "frame_idx": self._frame_count,
            "timestamp": round(now, 3),
            "expected_event": expected_event,
            "raw_open_conf": float(box_info.get("open_conf", 0)),
            "raw_closed_conf": float(box_info.get("closed_conf", 0)),
            "open_consec": self._open_consec,
            "closed_consec": self._closed_consec,
            "box_was_closed": self._box_was_closed,
            "box_was_open": self._box_was_open,
            "s1_hand_was_in_box": self._s1_hand_was_in_box,
            "s1_frames_since_hand_left": self._s1_frames_since_hand_left,
            "s1_closed_absent_total": self._s1_closed_absent_total,
            "emitted_events": list(self._emitted),
            "event_rejected_reason": reject_reason,
            "inter_event_gap": self._frame_count - self._last_any_event_frame,
        }
        self._debug_buffer.append(entry)
        self._debug_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if len(self._debug_buffer) % 50 == 0:
            self._debug_f.flush()

    @property
    def has_seen_box_open(self) -> bool:
        return self._box_was_open

    @property
    def emitted_events(self) -> List[str]:
        return list(self._emitted)

    @property
    def all_placed(self) -> bool:
        return all(e in self._emitted for e in PLACED_EVENTS)

    @property
    def open_evidence_frames(self) -> int:
        return self._open_consec

    @property
    def closed_evidence_frames(self) -> int:
        return self._closed_consec

    def close(self):
        if self._debug_f is not None:
            self._debug_f.flush()
            self._debug_f.close()
            self._debug_f = None
