"""
BoxStateStabilizer — short-window voting for box_open/box_closed.

Key insight: box opening is a FAST action. EMA comparison can't respond
quickly enough because the initial closed EMA suppresses open EMA.

Solution: per-frame candidate + 5-frame short-window majority vote.
No EMA, no hysteresis, no persistence of old bboxes.

States: UNKNOWN → OPEN / CLOSED / TRANSITION
"""
from typing import Optional, Dict, Tuple
from collections import deque
from enum import Enum


def _bbox_iou_simple(box1, box2) -> float:
    """Intersection-over-Union for two bboxes."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1.0)


def _ema_bbox(prev: Optional[Tuple], curr: Optional[Tuple], alpha: float,
              iou_gate: float = 0.0, prefer_larger: bool = False) -> Optional[Tuple]:
    """EMA-smoothed bbox with optional IoU gate to reject false positives.

    When iou_gate > 0 and prev exists: if the new bbox has IoU below the
    gate, the update is REJECTED — prev is returned unchanged.

    When prefer_larger=True: if the new bbox is substantially larger in area
    AND has decent IoU with prev, accept it immediately at higher weight.
    YOLO tends to predict tight boxes; a larger box is usually a correction.
    """
    if curr is None:
        return prev
    if prev is None:
        return curr
    if iou_gate > 0:
        iou = _bbox_iou_simple(curr, prev)
        if iou < iou_gate:
            return prev  # reject — too far from established position

    # Size-aware: if new bbox is significantly larger and overlaps, trust it more
    prev_area = (prev[2] - prev[0]) * (prev[3] - prev[1])
    curr_area = (curr[2] - curr[0]) * (curr[3] - curr[1])
    if prefer_larger and curr_area > prev_area * 1.08:
        # New bbox is >8% larger — likely a correction of YOLO's tight crop
        # Use higher alpha to converge faster to the correct size
        alpha = min(0.75, alpha * 1.5)

    return tuple(
        alpha * c + (1.0 - alpha) * p for c, p in zip(curr, prev))


class BoxState(Enum):
    UNKNOWN = 0
    OPEN = 1
    CLOSED = 2
    TRANSITION = 3


class BoxStateStabilizer:
    """Short-window voting box state tracker.

    Each frame: determine candidate_state from current raw confidences.
    Vote over last 5 frames. Majority (>=2) wins.

    bbox ONLY from current frame. UNKNOWN/TRANSITION returns None.
    """

    def __init__(self,
                 open_thr: float = 0.35,
                 closed_thr: float = 0.35,
                 vote_window: int = 5,
                 vote_need: int = 2,
                 bbox_ema_alpha: float = 0.35,
                 persistence_timeout: int = 30,
                 fast_close_thr: float = 0.6,
                 fast_vote_need: int = 1):
        self.open_thr = open_thr
        self.closed_thr = closed_thr
        self.vote_window = vote_window
        self.vote_need = vote_need
        self.bbox_ema_alpha = bbox_ema_alpha

        # Fix 1A: State persistence — maintain previous confident state
        # when no evidence arrives, with timeout before falling back to UNKNOWN.
        self.persistence_timeout = persistence_timeout

        # Fix 2D: Fast-close acceleration — when confidence is very high,
        # reduce the vote threshold for faster response.
        self.fast_close_thr = fast_close_thr
        self.fast_vote_need = fast_vote_need

        self._state: BoxState = BoxState.UNKNOWN
        self._previous_state: BoxState = BoxState.UNKNOWN
        self._state_frames: int = 0

        # Short-window vote history
        self._votes: deque = deque(maxlen=vote_window)

        # Current frame raw values
        self._open_bbox: Optional[Tuple] = None
        self._closed_bbox: Optional[Tuple] = None
        self._open_bbox_ema: Optional[Tuple] = None
        self._closed_bbox_ema: Optional[Tuple] = None
        self._open_conf: float = 0.0
        self._closed_conf: float = 0.0
        self._candidate_state: BoxState = BoxState.UNKNOWN

        # Hit/miss counters for UI display
        self._open_hits: int = 0
        self._closed_hits: int = 0
        self._open_miss: int = 0
        self._closed_miss: int = 0

        # Fix 1A: consecutive frames with no confident evidence (both conf < thr)
        self._stale_frames: int = 0

    def reset(self):
        self._state = BoxState.UNKNOWN
        self._previous_state = BoxState.UNKNOWN
        self._state_frames = 0
        self._votes.clear()
        self._open_bbox = None
        self._closed_bbox = None
        self._open_bbox_ema = None
        self._closed_bbox_ema = None
        self._open_conf = 0.0
        self._closed_conf = 0.0
        self._candidate_state = BoxState.UNKNOWN
        self._open_hits = 0
        self._closed_hits = 0
        self._open_miss = 0
        self._closed_miss = 0
        self._stale_frames = 0

    def update(self, open_conf: float, closed_conf: float,
               open_bbox=None, closed_bbox=None) -> BoxState:
        """Update with current frame's best open/closed confidence.

        Short-window voting: candidate from current frame → vote → state.
        """
        prev = self._state
        self._open_conf = float(open_conf)
        self._closed_conf = float(closed_conf)
        self._state_frames += 1

        # ── Update bboxes from current frame with EMA + size-aware gating ──
        # The box is stationary — reject detections at completely wrong positions.
        # But: always accept a LARGER bbox (YOLO tends to predict tight boxes
        # that clip the edges; expanding is usually correction, not error).
        # Gate prevents random false positives without blocking valid updates.
        self._open_bbox = open_bbox
        self._closed_bbox = closed_bbox
        gate = 0.20  # moderate: block random false positives, allow normal jitter

        # Size-aware: prefer larger bbox (more likely to contain the full box)
        self._open_bbox_ema = _ema_bbox(
            self._open_bbox_ema, open_bbox, self.bbox_ema_alpha, iou_gate=gate,
            prefer_larger=True)
        self._closed_bbox_ema = _ema_bbox(
            self._closed_bbox_ema, closed_bbox, self.bbox_ema_alpha, iou_gate=gate,
            prefer_larger=True)

        # ── Hit/miss counters ──
        if open_conf >= self.open_thr:
            self._open_hits += 1
            self._open_miss = 0
        else:
            self._open_miss += 1
        if closed_conf >= self.closed_thr:
            self._closed_hits += 1
            self._closed_miss = 0
        else:
            self._closed_miss += 1

        # ── Per-frame candidate with hysteresis ──
        # Hysteresis: require stronger evidence to CHANGE state than to STAY.
        # Without this, small YOLO confidence fluctuations cause open↔closed
        # oscillation every few frames.
        H_ENTER = 0.08   # deadband to ENTER a new state (need clear lead)
        H_EXIT  = 0.02   # deadband to STAY in current state (easy to stay)

        prev_is_open = self._state == BoxState.OPEN
        prev_is_closed = self._state == BoxState.CLOSED

        open_leads = open_conf - closed_conf

        if prev_is_open:
            # Already OPEN — need strong closed lead to flip
            if open_conf >= self.open_thr and open_leads >= -H_EXIT:
                candidate = BoxState.OPEN
            elif closed_conf >= self.closed_thr and open_leads <= -H_ENTER:
                candidate = BoxState.CLOSED
            else:
                candidate = BoxState.TRANSITION
        elif prev_is_closed:
            # Already CLOSED — need strong open lead to flip
            if closed_conf >= self.closed_thr and open_leads <= H_EXIT:
                candidate = BoxState.CLOSED
            elif open_conf >= self.open_thr and open_leads >= H_ENTER:
                candidate = BoxState.OPEN
            else:
                candidate = BoxState.TRANSITION
        else:
            # UNKNOWN/TRANSITION — use wider deadband to enter a state
            if open_conf >= self.open_thr and open_leads >= H_ENTER:
                candidate = BoxState.OPEN
            elif closed_conf >= self.closed_thr and open_leads <= -H_ENTER:
                candidate = BoxState.CLOSED
            else:
                candidate = BoxState.TRANSITION

        self._candidate_state = candidate
        self._votes.append(candidate)

        # ── Majority vote with fast-close and state persistence ──
        open_votes = sum(1 for v in self._votes if v == BoxState.OPEN)
        closed_votes = sum(1 for v in self._votes if v == BoxState.CLOSED)

        # Fix 2D: Fast-close acceleration — reduce vote threshold when
        # closed_conf is very high (decisive closing action).
        if closed_conf >= self.fast_close_thr or open_conf >= self.fast_close_thr:
            effective_vote_need = min(self.fast_vote_need, self.vote_need)
        else:
            effective_vote_need = self.vote_need

        if open_votes >= effective_vote_need:
            new_state = BoxState.OPEN
            self._stale_frames = 0
        elif closed_votes >= effective_vote_need:
            new_state = BoxState.CLOSED
            self._stale_frames = 0
        else:
            # ── Fix 1A: State persistence ──
            # Vote result is TRANSITION. Distinguish "real transition"
            # (mixed evidence exists) from "no evidence" (both conf < thr).
            has_open_evidence = open_conf >= self.open_thr
            has_closed_evidence = closed_conf >= self.closed_thr

            if has_open_evidence or has_closed_evidence:
                # Real transition: some evidence exists but not enough
                # for majority. This is genuine ambiguity.
                new_state = BoxState.TRANSITION
                self._stale_frames = 0
            elif self._state in (BoxState.OPEN, BoxState.CLOSED):
                # Previous state was confident, now no evidence at all.
                # Maintain previous state with timeout.
                self._stale_frames += 1
                if self._stale_frames >= self.persistence_timeout:
                    new_state = BoxState.TRANSITION
                    self._stale_frames = 0
                else:
                    new_state = self._state  # persist current state
            else:
                # Already UNKNOWN/TRANSITION, no evidence → stay TRANSITION.
                new_state = BoxState.TRANSITION
                self._stale_frames = 0

        # ── State transition ──
        if new_state != self._state:
            self._previous_state = self._state
            self._state = new_state
            self._state_frames = 0

        return self._state

    @property
    def state(self) -> BoxState:
        return self._state

    @property
    def previous_state(self) -> BoxState:
        return self._previous_state

    @property
    def state_str(self) -> str:
        return self._state.name.lower()

    @property
    def is_open(self) -> bool:
        return self._state == BoxState.OPEN

    @property
    def is_closed(self) -> bool:
        return self._state == BoxState.CLOSED

    @property
    def is_transition(self) -> bool:
        return self._state == BoxState.TRANSITION

    @property
    def is_unknown(self) -> bool:
        return self._state == BoxState.UNKNOWN

    @property
    def bbox(self):
        """Get EMA-smoothed bbox for current state.

        Uses EMA smoothing to prevent frame-to-frame jitter when the
        underlying YOLO bbox fluctuates slightly. NEVER returns old
        bbox from UNKNOWN/TRANSITION.
        """
        if self._state == BoxState.OPEN and self._open_bbox_ema is not None:
            return self._open_bbox_ema
        if self._state == BoxState.CLOSED and self._closed_bbox_ema is not None:
            return self._closed_bbox_ema
        # Fallback to raw bbox if EMA not yet initialized
        if self._state == BoxState.OPEN and self._open_bbox is not None:
            return self._open_bbox
        if self._state == BoxState.CLOSED and self._closed_bbox is not None:
            return self._closed_bbox
        return None

    @property
    def open_conf(self) -> float:
        return self._open_conf

    @property
    def closed_conf(self) -> float:
        return self._closed_conf

    @property
    def stable_frames(self) -> int:
        return self._state_frames

    @property
    def candidate_state(self) -> BoxState:
        return self._candidate_state

    def get_vote_counts(self) -> Dict:
        open_votes = sum(1 for v in self._votes if v == BoxState.OPEN)
        closed_votes = sum(1 for v in self._votes if v == BoxState.CLOSED)
        trans_votes = sum(1 for v in self._votes if v == BoxState.TRANSITION)
        return {"open": open_votes, "closed": closed_votes, "transition": trans_votes}

    def get_state_info(self) -> Dict:
        votes = self.get_vote_counts()
        return {
            "state": self.state_str,
            "previous_state": self._previous_state.name.lower(),
            "candidate_state": self._candidate_state.name.lower(),
            "is_open": self.is_open,
            "is_closed": self.is_closed,
            "is_transition": self.is_transition,
            "is_unknown": self.is_unknown,
            "open_hits": int(self._open_hits),
            "closed_hits": int(self._closed_hits),
            "open_miss": int(self._open_miss),
            "closed_miss": int(self._closed_miss),
            "open_conf": round(float(self._open_conf), 3),
            "closed_conf": round(float(self._closed_conf), 3),
            "open_bbox": list(self._open_bbox) if self._open_bbox else None,
            "closed_bbox": list(self._closed_bbox) if self._closed_bbox else None,
            "vote_open": votes["open"],
            "vote_closed": votes["closed"],
            "vote_transition": votes["transition"],
            "stable_frames": int(self._state_frames),
        }
