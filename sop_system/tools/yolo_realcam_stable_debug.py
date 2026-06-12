"""
YOLO-Only Stable Detection Debug Server
=======================================
Pauses ALL SOP/FSM/Temporal — only validates YOLO detection stability.

Pipeline:
  Camera frame → YOLO raw → class threshold → ROI/SpatialGate
  → DetectionStabilizer → SimpleTrackerNoPrediction → UI display → debug log

5 target classes: box_open, box_closed, earphone, charger, green_bag

NO FSM. NO TemporalPredictor. NO EventDetector. NO FeatureExtractor.
NO S1-S5 display. NO alarm. NO fallback bboxes. NO tracker predicted bboxes.

Usage:
  python tools/yolo_realcam_stable_debug.py --port 8766
"""

import sys
import os
import json
import time
import struct
import argparse
import asyncio
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.yolo_detector import YOLODetector, Detection

# ═══════════════════════════════════════════════════════════════
# ROI Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class ROIConfig:
    """All ROIs in pixel coordinates (relative to 1280x720 frame)."""
    # Box region — box_open/box_closed ONLY allowed here
    box_roi: Tuple[float, float, float, float] = (350, 200, 900, 600)

    # Box inner region (15% inset from box_roi) — for "in_box" checks
    box_inner_roi: Optional[Tuple[float, float, float, float]] = None

    # Object init regions (where objects start, outside the box)
    earphone_init_roi: Tuple[float, float, float, float] = (50, 300, 300, 600)
    charger_init_roi: Tuple[float, float, float, float] = (50, 300, 300, 600)
    green_bag_init_roi: Tuple[float, float, float, float] = (50, 300, 300, 600)

    # Object transfer regions (between init and box)
    earphone_transfer_roi: Tuple[float, float, float, float] = (0, 100, 1280, 700)
    charger_transfer_roi: Tuple[float, float, float, float] = (0, 100, 1280, 700)
    green_bag_transfer_roi: Tuple[float, float, float, float] = (0, 100, 1280, 700)

    def __post_init__(self):
        if self.box_inner_roi is None:
            bx1, by1, bx2, by2 = self.box_roi
            bw, bh = bx2 - bx1, by2 - by1
            inset_x = bw * 0.15
            inset_y = bh * 0.15
            self.box_inner_roi = (bx1 + inset_x, by1 + inset_y,
                                   bx2 - inset_x, by2 - inset_y)

    def allowed_region_for(self, cls_name: str) -> Optional[Tuple]:
        """Return (x1,y1,x2,y2) allowed region, or None if unrestricted."""
        if cls_name in ("box_open", "box_closed"):
            return self.box_roi
        elif cls_name == "earphone":
            # Union of init + transfer + box_inner — we implement as
            # "not strictly enforced in one rect but checked per-frame"
            return None  # earphone can move between zones
        elif cls_name == "charger":
            return None
        elif cls_name == "green_bag":
            return None
        return None


# ═══════════════════════════════════════════════════════════════
# SpatialGate — ROI filter
# ═══════════════════════════════════════════════════════════════

@dataclass
class RejectedDetection:
    cls_name: str
    conf: float
    bbox: Tuple
    reason: str
    frame_id: int = 0


class SpatialGate:
    """Filters detections by class-specific ROI constraints.

    box_open/box_closed: MUST have center inside box_roi AND overlap >= 0.4
    Other objects: center must be in their allowed union of regions.
    """

    def __init__(self, roi: ROIConfig):
        self.roi = roi

    def filter(self, detections: List[Detection], frame_id: int = 0
              ) -> Tuple[List[Detection], List[RejectedDetection]]:
        kept = []
        rejected = []

        for d in detections:
            reason = self._check(d)
            if reason is None:
                kept.append(d)
            else:
                rejected.append(RejectedDetection(
                    cls_name=d.cls_name, conf=d.confidence,
                    bbox=d.bbox, reason=reason, frame_id=frame_id))

        return kept, rejected

    def _check(self, d: Detection) -> Optional[str]:
        cls = d.cls_name
        cx = (d.bbox[0] + d.bbox[2]) / 2
        cy = (d.bbox[1] + d.bbox[3]) / 2

        if cls in ("box_open", "box_closed"):
            return self._check_box(d, cx, cy)

        # Other objects — check if in any allowed zone
        allowed = self._get_object_allowed_union(cls)
        if allowed is not None:
            ax1, ay1, ax2, ay2 = allowed
            if not (ax1 <= cx <= ax2 and ay1 <= cy <= ay2):
                return "outside_object_allowed_roi"
        return None

    def _check_box(self, d: Detection, cx: float, cy: float) -> Optional[str]:
        """box_open/box_closed must be inside box_roi with sufficient overlap."""
        bx1, by1, bx2, by2 = self.roi.box_roi

        # Center must be inside box_roi
        if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
            return "outside_box_roi"

        # Bbox must have meaningful overlap with box_roi
        overlap = self._overlap_ratio(d.bbox, self.roi.box_roi)
        if overlap < 0.4:
            return f"box_roi_overlap={overlap:.2f}_below_0.4"

        return None

    def _get_object_allowed_union(self, cls: str) -> Optional[Tuple]:
        """Get the bounding rect that covers all allowed zones for this object."""
        if cls == "earphone":
            zones = [self.roi.earphone_init_roi,
                     self.roi.earphone_transfer_roi,
                     self.roi.box_inner_roi]
        elif cls == "charger":
            zones = [self.roi.charger_init_roi,
                     self.roi.charger_transfer_roi,
                     self.roi.box_inner_roi]
        elif cls == "green_bag":
            zones = [self.roi.green_bag_init_roi,
                     self.roi.green_bag_transfer_roi,
                     self.roi.box_inner_roi]
        else:
            return None

        if not zones:
            return None
        x1 = min(z[0] for z in zones if z)
        y1 = min(z[1] for z in zones if z)
        x2 = max(z[2] for z in zones if z)
        y2 = max(z[3] for z in zones if z)
        return (x1, y1, x2, y2)

    @staticmethod
    def _overlap_ratio(bbox: Tuple, roi: Tuple) -> float:
        """Ratio of bbox area that overlaps with roi."""
        x1 = max(bbox[0], roi[0])
        y1 = max(bbox[1], roi[1])
        x2 = min(bbox[2], roi[2])
        y2 = min(bbox[3], roi[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return inter / max(bbox_area, 1.0)


# ═══════════════════════════════════════════════════════════════
# DetectionStabilizer (rewritten per spec)
# ═══════════════════════════════════════════════════════════════

@dataclass
class StableState:
    cls_name: str
    visible: bool = False
    confirmed: bool = False
    current_bbox: Optional[Tuple] = None
    conf: float = 0.0
    hit_frames: int = 0
    lost_frames: int = 0
    min_hits: int = 3
    max_lost: int = 10
    source: str = "yolo"


class DetectionStabilizerV2:
    """Hit/lost frame counting per spec.

    - Current frame has real YOLO detection: hit_frames++, lost_frames=0
    - Current frame no detection: hit_frames=0, lost_frames++
    - visible=True AND confirmed=True only when hit_frames >= min_hits
    - NO held bbox, NO last_bbox persistence
    """

    def __init__(self):
        self._state: Dict[str, StableState] = {}

        self._min_hits = {
            "box_open": 3, "box_closed": 3,
            "earphone": 2, "charger": 2, "green_bag": 2,
        }
        self._max_lost = {
            "box_open": 10, "box_closed": 10,
            "earphone": 10, "charger": 10, "green_bag": 10,
        }

    def reset(self):
        self._state.clear()

    def update(self, detections: List[Detection]) -> Dict[str, StableState]:
        """Update per-class stability and return state dict."""
        # Group best per class
        best: Dict[str, Detection] = {}
        for d in detections:
            if d.cls_name not in best or d.confidence > best[d.cls_name].confidence:
                best[d.cls_name] = d

        for cls_name in self._min_hits:
            if cls_name not in self._state:
                self._state[cls_name] = StableState(
                    cls_name=cls_name,
                    min_hits=self._min_hits[cls_name],
                    max_lost=self._max_lost[cls_name],
                )
            st = self._state[cls_name]
            det = best.get(cls_name)

            if det is not None:
                st.hit_frames += 1
                st.lost_frames = 0
                st.current_bbox = det.bbox
                st.conf = det.confidence
                st.source = getattr(det, 'source', 'yolo')
                if st.hit_frames >= st.min_hits:
                    st.visible = True
                    st.confirmed = True
            else:
                st.hit_frames = 0
                st.lost_frames += 1
                # NO held bbox — immediately clear
                st.current_bbox = None
                st.conf = 0.0
                if st.lost_frames > st.max_lost:
                    st.visible = False
                    st.confirmed = False

        return self._state

    def get_display_detections(self) -> List[Dict]:
        """Return only visible AND confirmed detections with current bbox."""
        result = []
        for cls_name, st in self._state.items():
            if st.visible and st.confirmed and st.current_bbox is not None:
                result.append({
                    "class": cls_name,
                    "conf": round(st.conf, 3),
                    "bbox": [int(v) for v in st.current_bbox],
                    "hit_frames": st.hit_frames,
                    "source": st.source,
                })
        return result


# ═══════════════════════════════════════════════════════════════
# SimpleTrackerNoPrediction
# ═══════════════════════════════════════════════════════════════

@dataclass
class SimpleTrack:
    track_id: int
    cls_name: str
    current_bbox: Optional[Tuple] = None
    last_real_bbox: Optional[Tuple] = None
    visible: bool = False
    lost_frames: int = 0


class SimpleTrackerNoPrediction:
    """ID association only. NO predicted bbox. NO Kalman.

    - Current frame has stable detection → associate by IoU, update track
    - Current frame no detection → track remembers last_real_bbox internally
      but visible=False, track is NOT output to UI
    """

    def __init__(self, max_lost: int = 30):
        self.max_lost = max_lost
        self._tracks: Dict[int, SimpleTrack] = {}
        self._next_id: int = 0
        self._cls_track: Dict[str, int] = {}  # cls_name → track_id (one per class)

    def reset(self):
        self._tracks.clear()
        self._next_id = 0
        self._cls_track.clear()

    def update(self, display_dets: List[Dict]) -> List[Dict]:
        """Associate current detections with existing tracks. No prediction."""
        # Mark all existing tracks as not-yet-seen
        unmatched_tracks = set(self._tracks.keys())

        for det in display_dets:
            cls = det["class"]
            bbox = det["bbox"]

            # Try to match existing track for this class
            matched_tid = None
            if cls in self._cls_track:
                tid = self._cls_track[cls]
                if tid in self._tracks:
                    track = self._tracks[tid]
                    if track.cls_name == cls:
                        # Simple center-distance match
                        if track.last_real_bbox is not None:
                            dist = self._center_dist(bbox, track.last_real_bbox)
                            # Match if within reasonable distance
                            obj_w = bbox[2] - bbox[0]
                            obj_h = bbox[3] - bbox[1]
                            max_dist = max(obj_w, obj_h) * 2.0
                            if dist < max_dist:
                                matched_tid = tid

            if matched_tid is not None:
                track = self._tracks[matched_tid]
                track.current_bbox = bbox
                track.last_real_bbox = bbox
                track.visible = True
                track.lost_frames = 0
                unmatched_tracks.discard(matched_tid)
                det["track_id"] = matched_tid
            else:
                # New track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = SimpleTrack(
                    track_id=tid, cls_name=cls,
                    current_bbox=bbox, last_real_bbox=bbox,
                    visible=True, lost_frames=0,
                )
                self._cls_track[cls] = tid
                det["track_id"] = tid

        # Update unmatched tracks — mark as lost
        for tid in unmatched_tracks:
            track = self._tracks[tid]
            track.visible = False
            track.current_bbox = None  # NO predicted bbox
            track.lost_frames += 1
            if track.lost_frames > self.max_lost:
                # Clean up stale track
                if track.cls_name in self._cls_track:
                    if self._cls_track[track.cls_name] == tid:
                        del self._cls_track[track.cls_name]
                del self._tracks[tid]

        return display_dets

    @staticmethod
    def _center_dist(bbox1, bbox2):
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


# ═══════════════════════════════════════════════════════════════
# YOLO Debug Server
# ═══════════════════════════════════════════════════════════════

class YOLODebugServer:
    """Standalone YOLO-only debug server. NO SOP, NO FSM, NO temporal."""

    def __init__(self, yolo_path: str, roi: ROIConfig = None,
                 debug_dir: str = None):
        print("[YOLO-Debug] Loading YOLO model...")
        self.yolo = YOLODetector(
            yolo_path, conf_thresh=0.10, device="cuda",
            imgsz=640, use_tracker=False,  # ← disable ByteTrack
            bbox_ema_alpha=0.0,  # ← no EMA (we handle stabilization ourselves)
            use_clahe=True,
        )
        self.roi = roi or ROIConfig()
        self.gate = SpatialGate(self.roi)
        self.stabilizer = DetectionStabilizerV2()
        self.tracker = SimpleTrackerNoPrediction(max_lost=30)

        # Debug logging
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._debug_f = None
        self._frame_count = 0
        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            self._debug_f = open(
                str(self._debug_dir / "debug_trace.jsonl"), "w", encoding="utf-8")

        # FPS tracking
        self._fps_times = deque(maxlen=30)
        self._count = 0

        print(f"[YOLO-Debug] ROIs: box={self.roi.box_roi} "
              f"box_inner={self.roi.box_inner_roi}")
        print("[YOLO-Debug] Pipeline: YOLO → SpatialGate → Stabilizer → "
              "SimpleTracker → Display")
        print("[YOLO-Debug] NO FSM. NO Temporal. NO SOP. NO fallback.")
        print("[YOLO-Debug] Ready.")

    def reset(self):
        self.stabilizer.reset()
        self.tracker.reset()
        self._frame_count = 0

    def process(self, jpeg_bytes: bytes, frame_id: int,
                timestamp: float) -> Dict:
        t0 = time.time()
        self._frame_count += 1

        # Decode
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"frame_id": frame_id, "error": "decode failed"}

        h, w = frame.shape[:2]

        # ── Step 1: YOLO raw detections ──
        raw_detections = self.yolo.detect(frame)

        # ── Step 2: Class threshold ──
        class_thresholds = {
            "box_open": 0.60, "box_closed": 0.60,
            "earphone": 0.20, "charger": 0.20, "green_bag": 0.20,
        }
        threshold_rejected = []
        threshold_kept = []
        for d in raw_detections:
            thr = class_thresholds.get(d.cls_name, 0.30)
            if d.confidence < thr:
                threshold_rejected.append(RejectedDetection(
                    cls_name=d.cls_name, conf=d.confidence, bbox=d.bbox,
                    reason=f"below_threshold_{thr}", frame_id=self._frame_count))
            else:
                threshold_kept.append(d)

        # ── Step 3: box_open/box_closed mutual exclusion ──
        box_open_dets = [d for d in threshold_kept if d.cls_name == "box_open"]
        box_closed_dets = [d for d in threshold_kept if d.cls_name == "box_closed"]
        box_conflict_rejected = []

        if box_open_dets and box_closed_dets:
            best_open = max(box_open_dets, key=lambda d: d.confidence)
            best_closed = max(box_closed_dets, key=lambda d: d.confidence)
            if best_open.confidence >= best_closed.confidence:
                # Keep open, reject closed
                for d in box_closed_dets:
                    box_conflict_rejected.append(RejectedDetection(
                        cls_name=d.cls_name, conf=d.confidence, bbox=d.bbox,
                        reason="box_open_closed_conflict", frame_id=self._frame_count))
                threshold_kept = [d for d in threshold_kept if d.cls_name != "box_closed"]
            else:
                # Keep closed, reject open
                for d in box_open_dets:
                    box_conflict_rejected.append(RejectedDetection(
                        cls_name=d.cls_name, conf=d.confidence, bbox=d.bbox,
                        reason="box_open_closed_conflict", frame_id=self._frame_count))
                threshold_kept = [d for d in threshold_kept if d.cls_name != "box_open"]

        # ── Step 4: SpatialGate / ROI filter ──
        roi_kept, roi_rejected = self.gate.filter(
            threshold_kept, frame_id=self._frame_count)

        # ── Step 5: DetectionStabilizer ──
        self.stabilizer.update(roi_kept)

        # ── Step 6: Get display-ready detections ──
        display_dets = self.stabilizer.get_display_detections()

        # ── Step 7: SimpleTracker (ID only, no prediction) ──
        display_dets = self.tracker.update(display_dets)

        # ── Collect all rejections ──
        all_rejected = threshold_rejected + box_conflict_rejected + roi_rejected
        not_stable = []
        for cls_name, st in self.stabilizer._state.items():
            if st.current_bbox is not None and not (st.visible and st.confirmed):
                not_stable.append(RejectedDetection(
                    cls_name=cls_name, conf=st.conf,
                    bbox=st.current_bbox,
                    reason=f"not_stable_enough_hits={st.hit_frames}/{st.min_hits}",
                    frame_id=self._frame_count))

        # ── FPS ──
        self._count += 1
        self._fps_times.append(t0)
        fps = 0.0
        if len(self._fps_times) >= 2:
            fps = (len(self._fps_times) - 1) / (
                self._fps_times[-1] - self._fps_times[0])

        total_ms = (time.time() - t0) * 1000

        # ── Periodic debug ──
        if self._frame_count % 60 == 0:
            n_raw = len(raw_detections)
            n_display = len(display_dets)
            n_rej = len(all_rejected) + len(not_stable)
            display_classes = [d["class"] for d in display_dets]
            print(f"[YOLO-Debug #{self._frame_count}] "
                  f"raw={n_raw} display={n_display} "
                  f"classes={display_classes} "
                  f"rejected={n_rej} fps={fps:.1f}", flush=True)

        # ── Debug log ──
        if self._debug_f:
            log_entry = {
                "frame_id": self._frame_count,
                "timestamp": round(timestamp, 3),
                "raw_detections": [
                    {"class": d.cls_name, "conf": round(d.confidence, 3),
                     "bbox": [round(v, 1) for v in d.bbox]}
                    for d in raw_detections
                ],
                "rejected_detections": [
                    {"class": r.cls_name, "conf": round(r.conf, 3),
                     "bbox": [round(v, 1) for v in r.bbox],
                     "reason": r.reason}
                    for r in (all_rejected + not_stable)
                ],
                "stable_detections": [
                    {"class": d["class"], "conf": d["conf"],
                     "bbox": d["bbox"], "track_id": d.get("track_id", -1)}
                    for d in display_dets
                ],
                "display_detections_count": len(display_dets),
            }
            self._debug_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            if self._frame_count % 30 == 0:
                self._debug_f.flush()

        # ── Response ──
        return {
            "frame_id": frame_id,
            "mode": "YOLO_DEBUG",
            "server_fps": round(fps, 1),
            "latency_ms": round(total_ms, 1),
            "detections": display_dets,
            "raw_count": len(raw_detections),
            "rejected_count": len(all_rejected) + len(not_stable),
            "rejected_summary": self._rejected_summary(all_rejected + not_stable),
            "box_state": self._box_candidate(),
            "box_roi": list(self.roi.box_roi),
            "fps": round(fps, 1),
        }

    def _box_candidate(self) -> str:
        """Simple box state hint based on which class is visible."""
        box_open = self.stabilizer._state.get("box_open")
        box_closed = self.stabilizer._state.get("box_closed")
        if box_open and box_open.visible and box_open.confirmed:
            return "open"
        if box_closed and box_closed.visible and box_closed.confirmed:
            return "closed"
        return "unknown"

    def _rejected_summary(self, rejected: List[RejectedDetection]) -> Dict:
        """Count rejected detections by reason."""
        summary = {}
        for r in rejected:
            reason = r.reason.split("_below")[0] if "_below" in r.reason else r.reason
            reason = reason.split("=")[0] if "=" in reason else reason
            if reason not in summary:
                summary[reason] = 0
            summary[reason] += 1
        return summary

    def close(self):
        if self._debug_f:
            self._debug_f.flush()
            self._debug_f.close()


# ═══════════════════════════════════════════════════════════════
# WebSocket Server
# ═══════════════════════════════════════════════════════════════

async def handle_client(websocket, server: YOLODebugServer):
    """Handle one client connection."""
    client_addr = websocket.remote_address
    print(f"[YOLO-Debug] Client connected: {client_addr}")

    _processing = False
    last_frame_id = -1
    recv_count = 0

    try:
        async for message in websocket:
            if _processing:
                continue
            _processing = True
            recv_count += 1

            try:
                # Text control messages
                if isinstance(message, str):
                    cmd = message.strip().lower()
                    if cmd == "reset":
                        server.reset()
                        await websocket.send(json.dumps({
                            "control_ack": "reset",
                            "mode": "YOLO_DEBUG",
                            "message": "Reset complete",
                        }))
                    elif cmd == "roi":
                        # Return current ROI config
                        await websocket.send(json.dumps({
                            "control_ack": "roi",
                            "box_roi": list(server.roi.box_roi),
                            "box_inner_roi": list(server.roi.box_inner_roi),
                        }))
                    _processing = False
                    continue

                # Binary frame
                if isinstance(message, bytes) and len(message) >= 12:
                    frame_id = struct.unpack('<I', message[:4])[0]
                    ts = struct.unpack('<d', message[4:12])[0]
                    jpeg_bytes = message[12:]

                    if frame_id <= last_frame_id:
                        _processing = False
                        continue
                    last_frame_id = frame_id

                    result = server.process(jpeg_bytes, frame_id, ts)
                    await websocket.send(json.dumps(result, ensure_ascii=False))
            finally:
                _processing = False
    except Exception as e:
        print(f"[YOLO-Debug] Client disconnected: {e}")


async def main_async(args):
    from websockets.asyncio.server import serve

    roi = ROIConfig(
        box_roi=tuple(args.box_roi) if args.box_roi else (
            350, 200, 900, 600),
    )

    server = YOLODebugServer(
        yolo_path=args.yolo,
        roi=roi,
        debug_dir=args.debug_dir,
    )

    print(f"\n{'='*60}")
    print(f"YOLO-Only Stable Detection Debug Server")
    print(f"  Listening: ws://{args.host}:{args.port}")
    print(f"  Mode: YOLO_DEBUG (NO FSM, NO SOP, NO Temporal)")
    print(f"  Debug dir: {args.debug_dir or 'disabled'}")
    print(f"{'='*60}\n")

    try:
        async with serve(
            lambda ws: handle_client(ws, server),
            args.host, args.port,
            max_size=5 * 1024 * 1024,
        ):
            await asyncio.Future()  # run forever
    finally:
        server.close()


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-Only Stable Detection Debug Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--yolo",
                        default=str(ROOT / "models" / "yolo_final_v1.pt"))
    parser.add_argument("--debug-dir",
                        default=str(ROOT / "reports" / "yolo_stable_debug"))
    parser.add_argument("--box-roi", nargs=4, type=float,
                        default=None,
                        help="Box ROI: x1 y1 x2 y2 (default: 350 200 900 600)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
