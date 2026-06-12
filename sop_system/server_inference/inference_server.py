"""
SOP 实时推理服务器 — 接收 JPEG 流，GPU 推理，返回 SOP 状态

Usage (on 4090 server):
  ~/miniconda3/envs/three_env/bin/python server_inference/inference_server.py --stage 3

Stages:
  1 = echo only (接收图像，统计 FPS，返回基本信息)
  2 = YOLO detection (返回检测结果)
  3 = full SOP pipeline (YOLO + MediaPipe + 特征提取 + 时序模型 + FSM)

RuntimeMode: PREVIEW → ARMED → RUNNING → COMPLETE → ERROR
  - PREVIEW:  only camera + stable detection boxes, NO FSM, NO alarms
  - RUNNING:  full SOP detection with FSM + temporal model

依赖: torch, ultralytics, mediapipe, cv2, websockets
"""
import sys
import os
import json
import time
import struct
import asyncio
import argparse
import yaml
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List
from enum import Enum

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import websockets
from websockets.asyncio.server import serve
from engine.event_detector import EVENT_SEQUENCE, EVENT_TO_STEP_ID
from engine.sop_fsm import SOPStateMachine, SOPStep, STEP_NAMES
from engine.object_state_tracker import ObjectState


class RuntimeMode(Enum):
    PREVIEW = "PREVIEW"        # 未开始，只预览相机 + 稳定框
    ARMED = "ARMED"            # SPACE已按，等待workspace lock
    RUNNING = "RUNNING"        # 正式SOP检测
    COMPLETE = "COMPLETE"      # 流程完成
    ERROR = "ERROR"            # 报警状态

# ── Config ──
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
FPS_TARGET = 20.0


def load_roi_config(config_path: str = None) -> Dict:
    """Load ROI configuration from YAML file."""
    if config_path is None:
        config_path = str(ROOT / "configs" / "realcam_sop.yaml")
    path = Path(config_path)
    if not path.exists():
        print(f"[Config] WARNING: {path} not found, using defaults")
        return {
            "box_roi": [300, 150, 950, 650],
            "box_inner_roi": [380, 200, 870, 600],
            "earphone_init_roi": [100, 200, 350, 450],
            "charger_init_roi": [600, 100, 900, 350],
            "green_bag_init_roi": [50, 400, 300, 650],
        }
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[Config] Loaded ROI config from {path}")
    return cfg


def _bbox_area(bbox) -> float:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w * h


def _bbox_overlap_ratio(bbox, roi) -> float:
    """Intersection-over-bbox ratio: how much of bbox is inside ROI."""
    x1 = max(bbox[0], roi[0]); y1 = max(bbox[1], roi[1])
    x2 = min(bbox[2], roi[2]); y2 = min(bbox[3], roi[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    bbox_area = _bbox_area(bbox)
    return inter / max(bbox_area, 1.0)


def _bbox_aspect_ok(bbox) -> bool:
    """Check bbox aspect ratio is reasonable for a box."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return False
    ar = w / max(h, 1)
    return 0.3 < ar < 3.0


def _validate_box_detection(bbox, roi, min_area=800, max_area=300000,
                             min_overlap=0.35) -> str:
    """Validate a box_open/box_closed detection. Returns '' if OK or reject reason."""
    area = _bbox_area(bbox)
    if area < min_area:
        return "small_box_area"
    if area > max_area:
        return "large_box_area"
    if not _bbox_aspect_ok(bbox):
        return "bad_box_aspect"
    if _bbox_overlap_ratio(bbox, roi) < min_overlap:
        return "outside_box_roi"
    return ""

# ── Global state (shared between receiver and processor) ──
_latest_frame = None      # (frame_id, timestamp, jpeg_bytes)
_latest_lock = asyncio.Lock()
_fps_times = deque(maxlen=30)
_recv_count = 0
_skip_count = 0
_send_count = 0


# ═══════════════════════════════════════════════════════════════
# Stage 1: Echo only
# ═══════════════════════════════════════════════════════════════

class EchoServer:
    """Stage 1: Just receive JPEG, decode, echo back with FPS."""

    def __init__(self):
        self._fps_times = deque(maxlen=30)
        self._count = 0

    def process(self, jpeg_bytes: bytes, frame_id: int,
                timestamp: float) -> Dict:
        t0 = time.time()
        # Decode to verify integrity
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2] if img is not None else (0, 0)
        decode_ms = (time.time() - t0) * 1000

        self._count += 1
        self._fps_times.append(t0)
        fps = 0.0
        if len(self._fps_times) >= 2:
            fps = (len(self._fps_times) - 1) / (
                self._fps_times[-1] - self._fps_times[0])

        return {
            "frame_id": frame_id,
            "server_fps": round(fps, 1),
            "latency_ms": round(decode_ms, 1),
            "image_size": f"{w}x{h}",
            "stage": 1,
        }


# ═══════════════════════════════════════════════════════════════
# Fallback detection for difficult objects (white plug, green bag)
# ═══════════════════════════════════════════════════════════════

# HSV green range for transparent green bag
GREEN_LOWER = np.array([35, 25, 25], dtype=np.uint8)
GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)
GREEN_AREA_THRESHOLD = 300  # minimum pixels of green in ROI


class FallbackDetector:
    """Traditional CV fallback for objects YOLO struggles with.

    white_plug (white charger on white paper):
      - Edge detection + rectangular contour + two metal pins
    green_bag (transparent green bag):
      - HSV green area measurement in candidate regions
    """

    def __init__(self, save_dir: str = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            (self.save_dir / "images").mkdir(parents=True, exist_ok=True)
            self._meta_path = self.save_dir / "meta.jsonl"

    def detect_white_plug(self, frame: np.ndarray,
                          box_bbox=None) -> Optional[Dict]:
        """Edge + contour based white plug detection.

        Looks for a white-ish rectangular object with two dark metal pins.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Edge detection
        edges = cv2.Canny(enhanced, 30, 100)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # too small
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)

            # White plug: roughly 1.5:1 to 3:1 aspect ratio
            if not (1.2 < aspect < 3.5):
                continue

            # Check if region is mostly white/bright
            roi_gray = gray[y:y + ch, x:x + cw]
            if roi_gray.size == 0:
                continue
            mean_brightness = np.mean(roi_gray)
            if mean_brightness < 130:  # not white enough
                continue

            # Look for metal pins (dark vertical elements) in the region
            roi_edges = edges[y:y + ch, x:x + cw]
            dark_vertical = np.sum(roi_edges) / max(roi_edges.size, 1)

            score = area * (mean_brightness / 255) * (1 + dark_vertical)
            if score > best_score:
                best_score = score
                best = (x, y, x + cw, y + ch, mean_brightness / 255)

        if best:
            x1, y1, x2, y2, conf = best
            return {
                "class": "charger",
                "conf": round(min(conf * 0.7, 0.55), 3),  # cap at 0.55
                "bbox": [x1, y1, x2, y2],
                "method": "edge_fallback",
            }
        return None

    def detect_green_bag(self, frame: np.ndarray,
                         box_bbox=None) -> Optional[Dict]:
        """HSV-based green area detection for transparent green bag."""
        h, w = frame.shape[:2]

        # Search in regions outside the box (initial position)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # Find green blobs
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < GREEN_AREA_THRESHOLD:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)

            # Exclude detections inside the box (those should be YOLO's job)
            if box_bbox:
                bx1, by1, bx2, by2 = map(int, box_bbox)
                center_x = x + cw // 2
                center_y = y + ch // 2
                if bx1 <= center_x <= bx2 and by1 <= center_y <= by2:
                    continue

            if area > best_area:
                best_area = area
                best = (x, y, x + cw, y + ch)

        if best:
            x1, y1, x2, y2 = best
            conf = min(best_area / 3000, 0.6)  # cap at 0.6
            return {
                "class": "green_bag",
                "conf": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "method": "hsv_fallback",
            }
        return None

    def augment_detections(self, frame: np.ndarray,
                           detections: List[Dict],
                           box_bbox=None) -> List[Dict]:
        """Check for missing/low-conf objects and add fallback detections.

        NOTE: green_bag hsv_fallback is NEVER added to display detections.
        It is too unreliable and causes ghost boxes everywhere.
        Only charger edge_fallback is added (white plug on white surface).
        """
        result = list(detections)

        # Check what YOLO found
        has_charger = any(d["class"] == "charger" and d.get("conf", 0) > 0.25
                          for d in detections)
        charger_conf = max((d.get("conf", 0) for d in detections
                           if d["class"] == "charger"), default=0)

        # Missing/low-conf: track for hard sample
        missing = []
        low_conf = []

        if not has_charger:
            missing.append("charger")
            fb = self.detect_white_plug(frame, box_bbox)
            if fb:
                result.append(fb)
        elif charger_conf < 0.35:
            low_conf.append(("charger", charger_conf))
            fb = self.detect_white_plug(frame, box_bbox)
            if fb and fb["conf"] > charger_conf:
                result = [d for d in result if d["class"] != "charger"]
                result.append(fb)

        # green_bag hsv_fallback DISABLED for display — too many false positives
        # Only track whether YOLO found it for hard sample collection
        has_green = any(d["class"] == "green_bag" and d.get("conf", 0) > 0.25
                        for d in detections)
        if not has_green:
            missing.append("green_bag")

        # ── Save hard samples ──
        if (missing or low_conf) and self.save_dir:
            self._save_sample(frame, missing, low_conf)

        return result

    def _save_sample(self, frame: np.ndarray, missing: List[str],
                     low_conf: List[tuple]):
        """Save frame as hard sample for future YOLO retraining."""
        import time as _time
        ts = _time.time()
        fname = f"frame_{ts:.3f}_{'-'.join(missing) if missing else 'lowconf'}.jpg"
        img_path = self.save_dir / "images" / fname
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        with open(self._meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "filename": fname,
                "timestamp": ts,
                "missing_classes": missing,
                "low_conf_classes": [(c, round(conf, 3)) for c, conf in low_conf],
            }, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════
# Stage 2: YOLO detection (GPU)
# ═══════════════════════════════════════════════════════════════

def _iou(box1, box2) -> float:
    """Intersection over Union of two boxes."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1.0)


def _iou_with_roi(bbox, roi) -> float:
    """IoU between a detection bbox and a fixed ROI (x,y,x,y tuple)."""
    x1 = max(bbox[0], roi[0]); y1 = max(bbox[1], roi[1])
    x2 = min(bbox[2], roi[2]); y2 = min(bbox[3], roi[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    a2 = (roi[2] - roi[0]) * (roi[3] - roi[1])
    return inter / max(a1 + a2 - inter, 1.0)


def _filter_charger_box_overlap(detections, box_bbox, containment_threshold=0.65):
    """Remove charger detections that are mostly INSIDE the box bbox.

    Charger sitting ON the box will have edge overlap only (containment ~0.1-0.3).
    A false positive on the box itself has high containment (>0.65).
    We use containment (not IoU) to distinguish these cases.
    """
    filtered = []
    for d in detections:
        if d.cls_name == "charger" and box_bbox:
            dx1 = max(d.bbox[0], box_bbox[0])
            dy1 = max(d.bbox[1], box_bbox[1])
            dx2 = min(d.bbox[2], box_bbox[2])
            dy2 = min(d.bbox[3], box_bbox[3])
            if dx2 > dx1 and dy2 > dy1:
                inter_area = (dx2 - dx1) * (dy2 - dy1)
                charger_area = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
                if charger_area > 0:
                    containment = inter_area / charger_area
                    if containment > containment_threshold:
                        continue  # mostly inside box → false positive
        filtered.append(d)
    return filtered


def _suppress_objects_in_closed_box(det_summary, box_state, box_bbox):
    """When box is closed, remove non-box objects that are inside the box bbox.

    Objects inside a closed box are not visible to the camera.
    Any detection of charger/earphone/green_bag at that point is either:
    1. A false positive on the box surface → suppress
    2. A persisted track of a now-hidden object → suppress
    3. An object genuinely outside the box → keep (won't overlap with box bbox)
    """
    if box_state != "closed" or not box_bbox:
        return det_summary
    bx1, by1, bx2, by2 = box_bbox
    # Add margin: objects slightly outside the box edge are still "on" the box
    margin = 15
    bx1 -= margin
    by1 -= margin
    bx2 += margin
    by2 += margin
    result = []
    for d in det_summary:
        cls = d.get("class", "")
        if cls == "box_closed":
            result.append(d)
            continue
        if cls == "box_open":
            # box_open is impossible when box state is CLOSED — suppress
            continue
        bbox = d.get("bbox", [])
        if len(bbox) != 4:
            result.append(d)
            continue
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        # If object center is inside (expanded) box, suppress it
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            continue
        result.append(d)
    return result


def _find_white_charger_candidates(frame: np.ndarray):
    """Find white/bright regions that could be a charger (no pink paper needed).

    Returns list of (x1, y1, x2, y2, area) sorted by area descending.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold for bright regions (charger body is white)
    _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Find white blobs
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        # Charger is roughly rectangular, 1.5:1 to 4:1
        if 1.3 < aspect < 4.5:
            candidates.append((x, y, x + cw, y + ch, area))
    candidates.sort(key=lambda c: c[4], reverse=True)
    return candidates[:3]  # top 3


def _white_guided_charger_search(frame, yolo_model, w, h, class_names):
    """Run YOLO on white-region ROIs to find charger (replaces pink_guided)."""
    from engine.yolo_detector import Detection
    candidates = _find_white_charger_candidates(frame)
    for (rx1, ry1, rx2, ry2, area) in candidates:
        # Expand ROI slightly
        margin = 20
        rx1 = max(0, rx1 - margin)
        ry1 = max(0, ry1 - margin)
        rx2 = min(w, rx2 + margin)
        ry2 = min(h, ry2 + margin)
        if rx2 - rx1 < 30 or ry2 - ry1 < 30:
            continue
        crop = frame[ry1:ry2, rx1:rx2]
        crop_h, crop_w = crop.shape[:2]
        crop_results = yolo_model.predict(
            crop, verbose=False, conf=0.12, device="cuda",
            imgsz=max(crop_w, crop_h), max_det=3)
        for r in crop_results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    if cls_name == "charger":
                        conf = float(box.conf[0])
                        cx1, cy1, cx2, cy2 = box.xyxy[0].tolist()
                        fx1 = rx1 + cx1; fy1 = ry1 + cy1
                        fx2 = rx1 + cx2; fy2 = ry1 + cy2
                        return Detection(
                            cls_id=3, cls_name="charger",
                            confidence=min(conf * 1.25, 0.92),
                            bbox=(fx1, fy1, fx2, fy2),
                            center=((fx1+fx2)/2, (fy1+fy2)/2),
                            track_id=-1, tracked=False,
                            source="white_guided")
    return None


class YOLOServer:
    """Stage 2: YOLO CUDA detection + fallback for difficult objects."""

    def __init__(self, yolo_path: str, conf: float = 0.3, imgsz: int = 512,
                 save_hard_samples: str = None, use_pink_marker: bool = False):
        from engine.yolo_detector import YOLODetector, Detection
        from engine.detection_stabilizer import DetectionStabilizer
        from engine.box_state_stabilizer import BoxStateStabilizer
        print(f"  YOLO model_conf=0.12 imgsz={imgsz} CLAHE=ON tracker=ByteTrack")
        self.yolo = YOLODetector(yolo_path, conf_thresh=0.12, device="cuda",
                                imgsz=imgsz, use_tracker=True,
                                bbox_ema_alpha=0.35, use_clahe=True)
        self.det_stabilizer = DetectionStabilizer()
        self.box_stabilizer = BoxStateStabilizer(open_thr=0.20, closed_thr=0.35, vote_window=5, vote_need=2)
        self._fallback = FallbackDetector(save_dir=save_hard_samples)
        self._use_pink = use_pink_marker
        if use_pink_marker:
            from engine.pink_marker_detector import PinkMarkerDetector
            _pink_dbg = None
            if save_hard_samples:
                _pink_dbg = str(Path(save_hard_samples).parent / "pink_debug")
            self.pink_detector = PinkMarkerDetector(
                stable_frames=2, debug_dir=_pink_dbg)
            print("  Pink marker ENABLED (legacy)")
        else:
            self.pink_detector = None
            print("  Pink marker DISABLED, using white-guided search")
        self._charger_source = "none"
        self._fps_times = deque(maxlen=30)
        self._count = 0

    def reset(self):
        """Reset YOLO tracking and state (for Stage 2)."""
        self.yolo.reset_tracking()
        self.det_stabilizer.reset()
        self.box_stabilizer.reset()
        if self.pink_detector:
            self.pink_detector.reset()
        self._count = 0
        self._fps_times.clear()
        self._charger_source = "none"
        print("[YOLOServer] State reset")

    def process(self, jpeg_bytes: bytes, frame_id: int,
                timestamp: float) -> Dict:
        t0 = time.time()
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        decode_ms = (time.time() - t0) * 1000

        if frame is None:
            return {"frame_id": frame_id, "server_fps": 0, "error": "decode failed",
                    "mode": "PREVIEW", "workspace_locked": False}

        h, w = frame.shape[:2]

        t1 = time.time()
        detections = self.yolo.detect(frame)
        yolo_ms = (time.time() - t1) * 1000

        # ── Charger detection: YOLO first, then white-guided search ──
        charger_yolo_dets = [d for d in detections if d.cls_name == "charger"]
        charger_yolo_conf = max((d.confidence for d in charger_yolo_dets), default=0.0)
        charger_yolo_ok = len(charger_yolo_dets) > 0 and charger_yolo_conf >= 0.15

        guided_det = None
        if not charger_yolo_ok:
            guided_det = _white_guided_charger_search(
                frame, self.yolo.model, w, h, self.yolo.class_names)

            if guided_det is None and self._use_pink and self.pink_detector:
                from engine.yolo_detector import Detection
                pink = self.pink_detector.detect(frame, roi=None, roi_name="full")
                if pink.present:
                    cx = (pink.bbox[0] + pink.bbox[2]) / 2
                    cy = (pink.bbox[1] + pink.bbox[3]) / 2
                    pmw = pink.bbox[2] - pink.bbox[0]
                    pmh = pink.bbox[3] - pink.bbox[1]
                    rx1 = max(0, int(cx - pmw * 1.5))
                    ry1 = max(0, int(cy - pmh * 0.5))
                    rx2 = min(w, int(cx + pmw * 1.5))
                    ry2 = min(h, int(cy + pmh * 4.0))
                    if rx2 > rx1 + 20 and ry2 > ry1 + 20:
                        crop = frame[ry1:ry2, rx1:rx2]
                        crop_results = self.yolo.model.predict(
                            crop, verbose=False, conf=0.10, device="cuda",
                            imgsz=max(rx2-rx1, ry2-ry1), max_det=3)
                        for r in crop_results:
                            if r.boxes is not None:
                                for box in r.boxes:
                                    cls_id = int(box.cls[0])
                                    cls_name = self.yolo.class_names[cls_id] if cls_id < len(self.yolo.class_names) else str(cls_id)
                                    if cls_name == "charger":
                                        conf = float(box.conf[0])
                                        cx1, cy1, cx2, cy2 = box.xyxy[0].tolist()
                                        fx1 = rx1 + cx1; fy1 = ry1 + cy1
                                        fx2 = rx1 + cx2; fy2 = ry1 + cy2
                                        guided_det = Detection(
                                            cls_id=3, cls_name="charger",
                                            confidence=min(conf * 1.3, 0.95),
                                            bbox=(fx1, fy1, fx2, fy2),
                                            center=((fx1+fx2)/2, (fy1+fy2)/2),
                                            track_id=-1, tracked=False,
                                            source="pink_guided")
                                        break
                                if guided_det:
                                    break
                        if guided_det is None:
                            expanded = self.pink_detector.expand_bbox(pink.bbox, w, h)
                            guided_det = Detection(
                                cls_id=3, cls_name="charger", confidence=0.75,
                                bbox=expanded,
                                center=((expanded[0]+expanded[2])/2,
                                        (expanded[1]+expanded[3])/2),
                                track_id=-1, tracked=False,
                                source="pink_marker")

            if guided_det is not None:
                detections.append(guided_det)
                self._charger_source = guided_det.source
            else:
                self._charger_source = "none"
        else:
            self._charger_source = "yolo"

        if self._count % 30 == 0:
            print(f"[Charger] frame={frame_id} yolo_conf={charger_yolo_conf:.3f} "
                  f"source={self._charger_source}")

        # ── DetectionStabilizer (replaces old hysteresis+persist) ──
        stable = self.det_stabilizer.update(detections)

        # ── BoxStateStabilizer ──
        open_conf = max((d.confidence for d in detections if d.cls_name == "box_open"), default=0.0)
        closed_conf = max((d.confidence for d in detections if d.cls_name == "box_closed"), default=0.0)
        open_bbox = next((d.bbox for d in detections if d.cls_name == "box_open"), None)
        closed_bbox = next((d.bbox for d in detections if d.cls_name == "box_closed"), None)
        self.box_stabilizer.update(open_conf, closed_conf, open_bbox, closed_bbox)
        box_state_str = self.box_stabilizer.state_str
        box_bbox = self.box_stabilizer.bbox

        # Filter charger false positives
        if box_bbox:
            detections = _filter_charger_box_overlap(detections, box_bbox)

        # ── Display: ONLY confirmed detections (NO persist, NO ghost boxes) ──
        det_summary = self.det_stabilizer.get_display_detections()

        # Fallback for difficult objects
        fb_start = time.time()
        det_summary = self._fallback.augment_detections(
            frame, det_summary,
            box_bbox=list(box_bbox) if box_bbox else None)
        fallback_ms = (time.time() - fb_start) * 1000

        # Suppress objects inside closed box
        det_summary = _suppress_objects_in_closed_box(
            det_summary, box_state_str, box_bbox)

        self._count += 1
        self._fps_times.append(t0)
        fps = 0.0
        if len(self._fps_times) >= 2:
            fps = (len(self._fps_times) - 1) / (
                self._fps_times[-1] - self._fps_times[0])

        total_ms = (time.time() - t0) * 1000

        return {
            "frame_id": frame_id,
            "server_fps": round(fps, 1),
            "latency_ms": round(total_ms, 1),
            "decode_ms": round(decode_ms, 1),
            "yolo_ms": round(yolo_ms, 1),
            "fallback_ms": round(fallback_ms, 1),
            "detections": det_summary,
            "box_state": box_state_str,
            "box_bbox": list(box_bbox) if box_bbox else None,
            "charger_source": self._charger_source,
            "charger_yolo_conf": round(charger_yolo_conf, 3),
            "pink_init_present": (guided_det is not None),
            "pink_box_present": False,
            "mode": "PREVIEW",
            "workspace_locked": False,
            "stage": 2,
        }


# ═══════════════════════════════════════════════════════════════
# Stage 3: Full SOP pipeline
# ═══════════════════════════════════════════════════════════════

class SOPServer:
    """Stage 3: Complete pipeline — YOLO + MediaPipe + TCN+BiGRU + FSM.

    RuntimeMode gating:
      - PREVIEW: camera preview + stable boxes only (NO FSM, NO alarms)
      - RUNNING: full SOP detection
    """

    def __init__(self, yolo_path: str, model_path: str,
                 conf: float = 0.3, imgsz: int = 512,
                 save_hard_samples: str = None, use_pink_marker: bool = False):
        from engine.yolo_detector import YOLODetector
        from engine.hand_detector import HandDetector
        from engine.temporal_predictor_v2 import TemporalPredictorV2
        from engine.detection_stabilizer import DetectionStabilizer
        from engine.box_state_stabilizer import BoxStateStabilizer
        from engine.event_detector import EventDetector, EVENT_SEQUENCE, EVENT_TO_STEP_ID
        from engine.object_state_tracker import ObjectStateTracker
        from engine.action_segmenter import ActionSegmenter
        from tools.extract_features_v2_90 import (
            FeatureExtractorV2Standalone, INTERACT_OBJS)

        print(f"[Init] YOLO (CUDA) imgsz={imgsz} CLAHE=ON tracker=ByteTrack model_conf=0.12...")
        self.yolo = YOLODetector(yolo_path, conf_thresh=0.12, device="cuda",
                                imgsz=imgsz, use_tracker=True,
                                bbox_ema_alpha=0.35, use_clahe=True)
        self._base_conf = conf
        self._fallback = FallbackDetector(save_dir=save_hard_samples)
        self._use_pink = use_pink_marker
        if use_pink_marker:
            print("[Init] PinkMarkerDetector ENABLED (legacy)...")
            from engine.pink_marker_detector import PinkMarkerDetector
            _pink_dbg2 = None
            if save_hard_samples:
                _pink_dbg2 = str(Path(save_hard_samples).parent / "pink_debug")
            self.pink_detector = PinkMarkerDetector(
                stable_frames=2, debug_dir=_pink_dbg2)
        else:
            print("[Init] Pink marker DISABLED, white-guided search active")
            self.pink_detector = None
        print("[Init] MediaPipe Hands...")
        self.hand = HandDetector()
        print("[Init] FeatureExtractorV2Standalone...")
        self.extractor = FeatureExtractorV2Standalone(self.yolo, self.hand,
                                                      ema_alpha=0.5,
                                                      hold_frames=5)
        print(f"[Init] TemporalPredictorV2: {model_path}")
        self.predictor = TemporalPredictorV2(model_path, device="cuda",
                                            T=48, stride=4)
        self.fsm = SOPStateMachine(timeout=90.0, min_step_duration=0.2)
        self.STEPS = SOPStep
        self.STEP_NAMES = STEP_NAMES
        self.INTERACT_OBJS = INTERACT_OBJS
        self._phys_enabled = True

        self.EVENT_SEQUENCE = EVENT_SEQUENCE

        # ── Load ROI config ──
        self.roi_cfg = load_roi_config()
        self.BOX_ROI = tuple(self.roi_cfg["box_roi"])
        self.BOX_INNER_ROI = tuple(self.roi_cfg["box_inner_roi"])
        self.INIT_ROI_MAP = {
            "earphone": tuple(self.roi_cfg.get("earphone_init_roi", [100, 200, 350, 450])),
            "charger": tuple(self.roi_cfg.get("charger_init_roi", [600, 100, 900, 350])),
            "green_bag": tuple(self.roi_cfg.get("green_bag_init_roi", [50, 400, 300, 650])),
        }

        # ── NEW modules ──
        self.det_stabilizer = DetectionStabilizer()
        self.box_stabilizer = BoxStateStabilizer(
            open_thr=0.45, closed_thr=0.35, vote_window=5, vote_need=3)
        self.obj_tracker = ObjectStateTracker(
            stable_in_box_min=5, outside_box_min=5, hand_absent_min=5,
            lost_max=90, occluded_max=20)
        self.action_seg = ActionSegmenter()
        _event_dbg = None
        if save_hard_samples:
            _event_dbg = str(Path(save_hard_samples).parent / "event_logic_debug")
        self.event_detector = EventDetector(
            closed_stable_frames=15,
            early_close_confirm_frames=25, event_cooldown=10,
            debug_dir=_event_dbg)
        print("[Init] DetectionStabilizer ready (EMA + spatial consistency)")
        print(f"[Init] BoxStateStabilizer ready (short-window voting: open_thr={self.box_stabilizer.open_thr}, closed_thr={self.box_stabilizer.closed_thr}, window={self.box_stabilizer.vote_window}, need={self.box_stabilizer.vote_need})")
        print(f"[Init] ObjectStateTracker ready (stable={self.obj_tracker.stable_in_box_min}, hand_absent={self.obj_tracker.hand_absent_min})")
        print("[Init] ActionSegmenter ready")
        print("[Init] EventDetector ready (S1=hand-aware state machine, "
              f"S5_closed={self.event_detector.closed_confirm_frames}f, "
              f"S5_hand_absent={self.event_detector._s5_hand_absent_needed}f, "
              f"alarm={self.event_detector.early_close_confirm_frames}f)")
        print(f"[Init] ROI config: box_roi={self.BOX_ROI}")

        # ── RuntimeMode ──
        self.mode: RuntimeMode = RuntimeMode.PREVIEW
        self._workspace_locked: bool = False
        self._workspace_hits: int = 0
        self._workspace_miss: int = 0
        self._start_blocked_reason: Optional[str] = None

        # State
        self._fps_times = deque(maxlen=30)
        self._count = 0
        self._feat_count = 0
        self._last_feat_time = -1.0
        self._feat_interval = 1.0 / FPS_TARGET
        self._prev_fsm_step = -1
        self._fsm_events = []
        self._temporal_trace = []
        self._latest_top3 = []
        self._latest_step_probs = []
        self._latest_event = None
        self._wrong_active_consec: int = 0
        self._s5_complete_frames: int = 0
        self._charger_source = "none"
        self._expected_event: str = self.EVENT_SEQUENCE[0]
        self._accepted_events: List[str] = []
        self._max_fsm_path: List[int] = []  # monotonic — never shrinks
        self._initial_box_state: str = "unknown"
        self._action_result: Dict = {}
        self._last_stable: Dict = {}

    def reset(self):
        """Full reset — back to PREVIEW mode."""
        self.mode = RuntimeMode.PREVIEW
        self._workspace_locked = False
        self._workspace_hits = 0
        self._workspace_miss = 0
        self._start_blocked_reason = None
        self._expected_event = self.EVENT_SEQUENCE[0]
        self._accepted_events.clear()
        self._max_fsm_path.clear()
        self._initial_box_state = "unknown"
        self._action_result.clear()
        self.predictor.reset()
        self.yolo.reset_tracking()
        self.extractor.reset()
        self.hand.palm_velocity.clear()
        self.det_stabilizer.reset()
        self.box_stabilizer.reset()
        self.obj_tracker.reset()
        self.action_seg.reset()
        self.event_detector.reset()
        self.fsm = SOPStateMachine(timeout=90.0, min_step_duration=0.2)
        self._feat_count = 0
        self._last_feat_time = -1.0
        self._prev_fsm_step = -1
        self._fsm_events.clear()
        self._temporal_trace.clear()
        self._latest_top3 = []
        self._latest_step_probs = []
        self._latest_event = None
        self._wrong_active_consec = 0
        self._s5_complete_frames = 0
        self._charger_source = "none"
        if self.pink_detector:
            self.pink_detector.reset()
        print("[SOPServer] Full reset -> PREVIEW mode")

    def _update_workspace_lock(self, stable: Dict) -> bool:
        """Check if workspace is locked (camera aligned, key objects visible).

        Conditions:
          1. box state is known (OPEN or CLOSED, not UNKNOWN)
          2. At least 1 of {earphone, charger, green_bag} confirmed
          3. Stable for 5 consecutive frames
        """
        box_ok = self.box_stabilizer.state_str != "unknown"
        other_ok = sum(1 for obj in ["earphone", "charger", "green_bag"]
                       if stable.get(obj, {}).get("confirmed", False))
        if box_ok and other_ok >= 1:
            self._workspace_hits += 1
            self._workspace_miss = 0
        else:
            self._workspace_hits = 0
            self._workspace_miss += 1
        if self._workspace_hits >= 5:
            self._workspace_locked = True
        elif self._workspace_miss > 15:
            self._workspace_locked = False
        return self._workspace_locked

    def try_start(self) -> Dict:
        """Enter ARMED mode. No pre-conditions — just start."""
        if self.mode == RuntimeMode.RUNNING:
            return {"success": True, "message": "Already running"}
        if self.mode == RuntimeMode.COMPLETE:
            return {"success": False, "reason": "Sequence complete. Press reset."}
        self.mode = RuntimeMode.ARMED
        self._expected_event = self.EVENT_SEQUENCE[0]
        self._accepted_events.clear()
        self._max_fsm_path.clear()
        self._initial_box_state = self.box_stabilizer.state_str
        self.fsm.reset()
        self.obj_tracker.reset()
        self.action_seg.reset()
        self.event_detector.reset()
        self._prev_fsm_step = -1
        self._fsm_events.clear()
        self._temporal_trace.clear()
        self._latest_top3 = []
        self._latest_step_probs = []
        self._latest_event = None
        self._wrong_active_consec = 0
        self._s5_complete_frames = 0
        self._start_blocked_reason = None
        self._action_result.clear()
        print(f"[SOPServer] MODE -> ARMED initial_box={self._initial_box_state} "
              f"expected_event={self._expected_event}")
        return {"success": True, "mode": "ARMED",
                "message": "ARMED - waiting for box_opened",
                "initial_box_state": self._initial_box_state}

    def stop(self):
        self.mode = RuntimeMode.PREVIEW
        self._start_blocked_reason = None
        self._expected_event = self.EVENT_SEQUENCE[0]
        self._accepted_events.clear()
        self._max_fsm_path.clear()
        self._initial_box_state = "unknown"
        self._action_result.clear()
        self.fsm.reset()
        self.obj_tracker.reset()
        self.action_seg.reset()
        self.event_detector.reset()
        self._prev_fsm_step = -1
        self._fsm_events.clear()
        self._latest_top3 = []
        self._latest_step_probs = []
        self._latest_event = None
        self._wrong_active_consec = 0
        self._s5_complete_frames = 0
        print("[SOPServer] MODE -> PREVIEW")

    def process(self, jpeg_bytes: bytes, frame_id: int,
                timestamp: float) -> Dict:
        t0 = time.time()

        # ── Decode JPEG ──
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"frame_id": frame_id, "error": "decode failed",
                    "mode": self.mode.value, "workspace_locked": False}

        h, w = frame.shape[:2]

        # ── YOLO + MediaPipe ──
        detections = self.yolo.detect(frame)
        hands = self.hand.detect(frame)

        # ── Charger detection: YOLO first, then white-guided search ──
        charger_yolo_dets = [d for d in detections if d.cls_name == "charger"]
        charger_yolo_conf = max((d.confidence for d in charger_yolo_dets), default=0.0)
        charger_yolo_ok = len(charger_yolo_dets) > 0 and charger_yolo_conf >= 0.15

        # ── FALLBACKS DISABLED ──
        # white_guided_charger_search, pink_marker, hsv_fallback, edge_fallback
        # are all disabled during YOLO-only debug phase.
        # They bypass DetectionStabilizer and cause inconsistent display state.
        guided_det = None
        pink_init_present = False
        pink_box_present = False
        self._charger_source = "yolo" if charger_yolo_ok else "none"

        # ── Box ROI constraint with strict validation ──
        # box_open/box_closed at a FIXED position. Reject false positives.
        box_rejects = []
        valid_detections = []
        for d in detections:
            if d.cls_name in ("box_open", "box_closed"):
                reason = _validate_box_detection(d.bbox, self.BOX_ROI)
                if reason:
                    box_rejects.append((d.cls_name, reason, d.bbox))
                else:
                    valid_detections.append(d)
            else:
                valid_detections.append(d)
        detections = valid_detections
        if box_rejects and self._count % 30 == 0:
            for cls, reason, bbox in box_rejects:
                print(f"[BoxReject] {cls} rejected={reason} bbox={[int(v) for v in bbox]}", flush=True)

        # ── Box state: derive from max-confidence detection per class ──
        # CRITICAL: conf and bbox must come from the SAME detection.
        # Using max() for conf and next() for bbox separately picks from
        # different detections when multiple box_open/box_closed exist.
        best_open = max((d for d in detections if d.cls_name == "box_open"),
                         key=lambda d: d.confidence, default=None)
        best_closed = max((d for d in detections if d.cls_name == "box_closed"),
                           key=lambda d: d.confidence, default=None)
        open_conf = best_open.confidence if best_open else 0.0
        closed_conf = best_closed.confidence if best_closed else 0.0
        open_bbox = best_open.bbox if best_open else None
        closed_bbox = best_closed.bbox if best_closed else None

        # BoxStateStabilizer's higher-confidence-wins mechanism handles
        # mutual exclusion. Don't second-guess with spatial overlap here.

        # ── BoxStateStabilizer (ONLY source for box state/bbox) ──
        self.box_stabilizer.update(open_conf, closed_conf, open_bbox, closed_bbox)
        box_state_str = self.box_stabilizer.state_str
        box_bbox = self.box_stabilizer.bbox

        # ── DetectionStabilizer (objects ONLY — NOT box_open/box_closed) ──
        # box_open/box_closed are handled EXCLUSIVELY by BoxStateStabilizer
        object_detections = [d for d in detections
                            if d.cls_name not in ("box_open", "box_closed")]

        # ── Focused box-interior crop detection ──
        # When an object is being placed and full-frame YOLO didn't detect it,
        # run a focused higher-resolution pass on just the box area at lower
        # confidence (0.06). Objects inside the box are harder to detect due to
        # different lighting/background, and partial occlusion by box walls.
        _OBJ_FOR_EVENT = {
            "earphone_in_box": "earphone",
            "charger_in_box": "charger",
            "green_bag_in_box": "green_bag",
        }
        if (self.mode == RuntimeMode.RUNNING
                and "box_opened" in self._accepted_events
                and self._expected_event in _OBJ_FOR_EVENT):
            expected_obj = _OBJ_FOR_EVENT[self._expected_event]
            obj_state = self.obj_tracker.get_state(expected_obj)
            if obj_state in (ObjectState.LEFT_INIT, ObjectState.VISIBLE_IN_BOX,
                             ObjectState.OCCLUDED):
                yolo_has_obj = any(
                    d.cls_name == expected_obj and d.confidence >= 0.10
                    for d in object_detections
                )
                if not yolo_has_obj:
                    crop_src = box_bbox if box_bbox else self.BOX_ROI
                    if crop_src:
                        cx1, cy1, cx2, cy2 = crop_src
                        margin = int((cx2 - cx1) * 0.08)
                        crop_roi = (cx1 - margin, cy1 - margin,
                                    cx2 + margin, cy2 + margin)
                        crop_dets = self.yolo.detect_crop(
                            frame, crop_roi, conf_thresh=0.06)
                        crop_matches = [d for d in crop_dets
                                       if d.cls_name == expected_obj]
                        if crop_matches:
                            if self._count % 20 == 0:
                                best = max(crop_matches, key=lambda d: d.confidence)
                                print(f"[FocusedCrop] {expected_obj} found "
                                      f"conf={best.confidence:.3f}", flush=True)
                            object_detections.extend(crop_matches)

        stable = self.det_stabilizer.update(object_detections)
        self._last_stable = stable  # cache for try_start access

        # ── Filter charger false positives (ONLY when box is CLOSED) ──
        # When box is OPEN, charger/green_bag inside the box are genuinely visible.
        # Filtering them would make S2/S3/S4 impossible to complete.
        if box_bbox and box_state_str == "closed":
            detections = _filter_charger_box_overlap(detections, box_bbox)

        # ── WorkspaceLock ──
        self._update_workspace_lock(stable)

        # ── Display detections: objects from DetectionStabilizer + box from BoxStateStabilizer ──
        det_summary = self.det_stabilizer.get_display_detections()

        # ── Box display: ONLY show when state is OPEN or CLOSED, NEVER for UNKNOWN/TRANSITION ──
        # Also require OPEN bbox to spatially overlap with known box ROI — prevents
        # ghost box_open boxes from appearing on blank walls near the paper box.
        if box_state_str == "open" and box_bbox is not None:
            if _iou_with_roi(box_bbox, self.BOX_ROI) > 0.1:
                det_summary.insert(0, {
                    "class": "box_open",
                    "conf": round(self.box_stabilizer.open_conf, 3),
                    "bbox": [int(v) for v in box_bbox],
                })
        elif box_state_str == "closed" and box_bbox is not None:
            det_summary.insert(0, {
                "class": "box_closed",
                "conf": round(self.box_stabilizer.closed_conf, 3),
                "bbox": [int(v) for v in box_bbox],
            })

        # ── FALLBACKS DISABLED during YOLO debug phase ──
        # All fallback detections (white_guided, pink_marker, hsv, edge)
        # bypass the stabilizer pipeline and create inconsistent state.
        # Comment out to re-enable after YOLO base layer is clean.

        # Suppress objects inside closed box
        det_summary = _suppress_objects_in_closed_box(
            det_summary, box_state_str, box_bbox)

        # ── Mode-gated execution ──
        now = time.time()
        do_feat = (now - self._last_feat_time >= self._feat_interval
                   or self._last_feat_time < 0)

        fs_result = None
        current_step_id = 0
        step_name = "IDLE"
        pred_step = -1
        pred_conf = 0.0
        is_complete = False
        alarm = None

        # ── Compute box_inner_roi for ObjectStateTracker ──
        # Prefer the calibrated static ROI in real-camera deployments.
        # Fall back to a dynamic inset only when the current box bbox is valid.
        box_inner_roi = self.BOX_INNER_ROI
        if box_bbox is not None and "box_opened" not in self._accepted_events:
            bx1, by1, bx2, by2 = box_bbox
            bw, bh = bx2 - bx1, by2 - by1
            inset_x = bw * 0.15
            inset_y = bh * 0.15
            box_inner_roi = (bx1 + inset_x, by1 + inset_y, bx2 - inset_x, by2 - inset_y)

        # Always update event evidence (even PREVIEW) so box state history is primed
        box_state_info = self.box_stabilizer.get_state_info()
        self.event_detector.update_evidence(box_state_info)

        if self.mode in (RuntimeMode.ARMED, RuntimeMode.RUNNING):
            # ── hand_in_box: any hand bbox overlaps EXPANDED box ROI ──
            # Expand by 30% margin so hands NEAR the box (approaching/occluding)
            # are also caught, not just hands physically INSIDE the box.
            hand_in_box = False
            hand_blocks_release = False
            box_ref = box_bbox if box_bbox is not None else self.BOX_ROI
            if hands and box_ref:
                bx1, by1, bx2, by2 = box_ref
                bw, bh = bx2 - bx1, by2 - by1
                mx, my = bw * 0.30, bh * 0.30
                ebx1, eby1 = bx1 - mx, by1 - my
                ebx2, eby2 = bx2 + mx, by2 + my
                for h in hands:
                    hx1, hy1, hx2, hy2 = h.bbox
                    ox1 = max(ebx1, hx1); oy1 = max(eby1, hy1)
                    ox2 = min(ebx2, hx2); oy2 = min(eby2, hy2)
                    if ox2 > ox1 and oy2 > oy1:
                        hand_in_box = True
                        break
                rmx, rmy = bw * 0.05, bh * 0.05
                rbx1, rby1 = bx1 - rmx, by1 - rmy
                rbx2, rby2 = bx2 + rmx, by2 + rmy
                for h in hands:
                    hx1, hy1, hx2, hy2 = h.bbox
                    ox1 = max(rbx1, hx1); oy1 = max(rby1, hy1)
                    ox2 = min(rbx2, hx2); oy2 = min(rby2, hy2)
                    if ox2 > ox1 and oy2 > oy1:
                        hand_blocks_release = True
                        break

            # ── ObjectStateTracker: update per-object states (EVERY frame) ──
            for obj_name in ["earphone", "charger", "green_bag"]:
                obj_info = stable.get(obj_name, {})
                detected = obj_info.get("bbox") is not None
                bbox = obj_info.get("bbox")
                conf = obj_info.get("conf", 0.0)
                self.obj_tracker.update(
                    obj_name, detected, bbox, conf,
                    box_inner_roi=box_inner_roi,
                    init_roi=self.INIT_ROI_MAP.get(obj_name),
                    hand_in_box=hand_blocks_release,
                )

            # ── Feature extraction + TemporalPredictor (do_feat ONLY, AUXILIARY) ──
            temporal_aux = None
            if do_feat:
                self._last_feat_time = now
                self._feat_count += 1

                # Build feature_detections from objects + current box state
                feature_detections = [d for d in detections
                                      if d.cls_name not in ("box_open", "box_closed")]
                if feature_detections and box_bbox:
                    self.extractor.compute_interaction(feature_detections, hands, box_bbox, h, w)
                feat = self.extractor.extract(frame, feature_detections, hands,
                                              box_bbox, box_state_str)
                self.predictor.predict(feat)
                mature = self.predictor.get_latest_mature_prediction()

                if mature is not None:
                    pred_step = mature["step"]
                    pred_conf = mature["confidence"]
                    top3 = [(int(i), round(float(p), 4)) for i, p in mature["top3"]]
                    step_probs_arr = [round(float(p), 4) for p in mature["step_probs"]]
                    self._latest_top3 = top3
                    self._latest_step_probs = step_probs_arr
                    temporal_aux = {
                        "step_probs": mature["step_probs"],
                        "confidence": mature["confidence"],
                        "top3": mature["top3"],
                    }

            # ── EventDetector (EVERY frame — NOT gated by do_feat) ──
            box_state_info = self.box_stabilizer.get_state_info()
            self._latest_event = None
            event = self.event_detector.detect(
                box_state=self.box_stabilizer.state,
                box_state_info=box_state_info,
                obj_tracker=self.obj_tracker,
                expected_event=self._expected_event,
                accepted_events=self._accepted_events,
                temporal_aux=temporal_aux,
                hand_in_box=hand_in_box,
            )

            if event is not None:
                self._latest_event = {
                    "name": event.event_name,
                    "confidence": event.confidence,
                    "conditions": event.conditions_met,
                    "rejected": event.rejected_reason,
                    "temporal_boost": event.temporal_conf_boost,
                }

                if self.mode == RuntimeMode.ARMED:
                    if event.event_name == "box_opened":
                        if self._accept_event(event, now):
                            self.mode = RuntimeMode.RUNNING
                            print("[SOPServer] MODE -> RUNNING (box_opened)", flush=True)
                elif self.mode == RuntimeMode.RUNNING:
                    if event.event_name == "early_close_alarm":
                        self.event_detector.mark_event_accepted(event.event_name)
                        self._fsm_events.append({
                            "step": 7, "step_name": "EARLY_CLOSE_ALARM",
                            "event_name": event.event_name,
                            "confidence": round(event.confidence, 4),
                            "is_correct": False, "has_error": True,
                            "error_type": "EARLY_CLOSE",
                            "message": "Box closed but S2/S3/S4 not done!",
                        })
                        self.mode = RuntimeMode.ERROR
                    elif event.event_name == self._expected_event:
                        accepted_now = self._accept_event(event, now)
                        if accepted_now and self.fsm.current_step == self.STEPS.COMPLETE:
                            self.mode = RuntimeMode.COMPLETE
                            print("[SOPServer] COMPLETE!", flush=True)

            # ── ActionSegmenter update (EVERY frame) ──
            obj_summary = self.obj_tracker.get_summary()
            self._action_result = self.action_seg.update(
                expected_event=self._expected_event,
                box_state_str=self.box_stabilizer.state_str,
                box_previous_state_str=self.box_stabilizer.get_state_info()["previous_state"],
                obj_tracker_summary=obj_summary,
                accepted_events=self._accepted_events,
                now=now,
                open_evidence=self.event_detector.open_evidence_frames,
                closed_evidence=self.event_detector.closed_evidence_frames,
            )

            # ── Wrong-order alarm: WRONG_ACTIVE persisting >15 frames → ERROR ──
            if self._action_result.get("action_phase") == "WRONG_ACTIVE":
                self._wrong_active_consec += 1
                if self._wrong_active_consec >= 15:
                    wrong_obj = self._action_result.get("wrong_object", "?")
                    self._fsm_events.append({
                        "step": 7, "step_name": "WRONG_ORDER",
                        "event_name": "wrong_active",
                        "confidence": 0.85,
                        "is_correct": False, "has_error": True,
                        "error_type": "WRONG_ORDER",
                        "message": f"Wrong object in box: {wrong_obj} (expected {self._expected_event})",
                    })
                    self.mode = RuntimeMode.ERROR
                    print(f"[SOPServer] WRONG ORDER: {wrong_obj} entered box before {self._expected_event}", flush=True)
            else:
                self._wrong_active_consec = max(0, self._wrong_active_consec - 1)

            # ── S5→COMPLETE auto-advance ──
            # box_closed event maps to S5_CLOSE. Once the box is stably closed
            # AND the hand has left the box area, auto-advance to COMPLETE.
            if self.fsm.current_step == self.STEPS.S5_CLOSE:
                if self.box_stabilizer.state_str == "closed" and not hand_in_box:
                    self._s5_complete_frames += 1
                elif hand_in_box:
                    self._s5_complete_frames = 0
                if self._s5_complete_frames >= 10:
                    # Auto-advance to COMPLETE
                    self.fsm.current_step = self.STEPS.COMPLETE
                    self.mode = RuntimeMode.COMPLETE
                    self._fsm_events.append({
                        "step": 6, "step_name": "COMPLETE",
                        "event_name": "box_closed_confirmed",
                        "confidence": 0.95,
                        "is_correct": True, "has_error": False,
                        "error_type": "",
                        "message": "S5->COMPLETE: box stably closed, hand absent",
                    })
                    print("[SOPServer] S5->COMPLETE auto-advance (box stably closed, hand absent)",
                          flush=True)

            step_name = self.STEP_NAMES[self.fsm.current_step]
            current_step_id = self.fsm.current_step.value
            is_complete = (self.fsm.current_step == self.STEPS.COMPLETE)

            # Alarm
            errors = [e for e in self._fsm_events if e.get("has_error")]
            if errors:
                alarm = {"type": errors[-1]["error_type"], "message": errors[-1]["message"]}

        elif do_feat:
            # PREVIEW mode: temporal predictions for UI display (warmup only)
            self._last_feat_time = now
            self._feat_count += 1
            pre_feat_dets = [d for d in detections
                            if d.cls_name not in ("box_open", "box_closed")]
            if box_bbox:
                self.extractor.compute_interaction(pre_feat_dets, hands, box_bbox, h, w)
            feat = self.extractor.extract(frame, pre_feat_dets, hands,
                                          box_bbox, box_state_str)
            self.predictor.predict(feat)
            mature = self.predictor.get_latest_mature_prediction()
            if mature is not None:
                pred_step = mature["step"]
                pred_conf = mature["confidence"]
                self._latest_top3 = [(int(i), round(float(p), 4)) for i, p in mature["top3"]]
                self._latest_step_probs = [round(float(p), 4) for p in mature["step_probs"]]

        # ── FPS ──
        self._count += 1
        self._fps_times.append(t0)
        fps = 0.0
        if len(self._fps_times) >= 2:
            fps = (len(self._fps_times) - 1) / (
                self._fps_times[-1] - self._fps_times[0])

        total_ms = (time.time() - t0) * 1000

        # ── Periodic debug ──
        if self._count % 30 == 0:
            conf_cls = self.det_stabilizer.confirmed_classes
            emitted = self.event_detector.emitted_events
            obj_sum = self.obj_tracker.get_summary()
            bsi = self.box_stabilizer.get_state_info()
            ts_info = f" tpred=S{pred_step} tconf={pred_conf:.2f}" if pred_step >= 0 else ""
            oe = self.event_detector.open_evidence_frames
            ce = self.event_detector.closed_evidence_frames
            print(f"[Debug#{self._count}] mode={self.mode.value} box={self.box_stabilizer.state_str} "
                  f"candidate={bsi['candidate_state']} "
                  f"open_raw={open_conf:.2f} closed_raw={closed_conf:.2f} "
                  f"open_evidence={oe}/{self.box_stabilizer.vote_need} "
                  f"closed_evidence={ce}/{self.box_stabilizer.vote_need} "
                  f"votes=O{bsi['vote_open']}C{bsi['vote_closed']}T{bsi['vote_transition']} "
                  f"expected={self._expected_event} confirmed={conf_cls} emitted={emitted} "
                  f"objects={obj_sum}{ts_info} fps={fps:.1f}", flush=True)

        return {
            "frame_id": frame_id,
            "server_fps": round(fps, 1),
            "latency_ms": round(total_ms, 1),
            "detections": det_summary,
            "box_state": box_state_str,
            "box_bbox": (list(box_bbox) if box_bbox else None),
            "box_open_raw_conf": round(open_conf, 3),
            "box_closed_raw_conf": round(closed_conf, 3),
            "box_stabilizer_info": self.box_stabilizer.get_state_info(),
            "hands_detected": len(hands),
            # Mode & workspace
            "mode": self.mode.value,
            "workspace_locked": self._workspace_locked,
            "start_blocked": self._start_blocked_reason,
            # FSM state
            "current_step": step_name,
            "current_step_id": current_step_id,
            "is_complete": is_complete,
            # ARMED mode: don't show TemporalPredictor prediction in S-boxes
            # (model hasn't seen the action sequence yet — predictions are noise)
            "model_pred": -1 if self.mode == RuntimeMode.ARMED else pred_step,
            "confidence": round(pred_conf, 4),
            "top3_probs": getattr(self, '_latest_top3', []),
            "step_probs": getattr(self, '_latest_step_probs', []),
            "alarm": alarm,
            "fsm_path": list(self._max_fsm_path),
            "feature_count": self._feat_count,
            "charger_source": self._charger_source,
            "charger_yolo_conf": round(charger_yolo_conf, 3),
            "pink_init_present": pink_init_present,
            "pink_box_present": pink_box_present,
            # EventDetector info
            "expected_event": self._expected_event,
            "accepted_events": self._accepted_events,
            "event_name": self._latest_event["name"] if self._latest_event else None,
            "event_confidence": self._latest_event["confidence"] if self._latest_event else 0.0,
            "event_conditions": self._latest_event["conditions"] if self._latest_event else [],
            "event_rejected": self._latest_event["rejected"] if self._latest_event else "",
            "event_temporal_boost": self._latest_event["temporal_boost"] if self._latest_event else 0.0,
            # ActionSegmenter info
            "current_action": self._action_result.get("current_action", "none"),
            "action_phase": self._action_result.get("action_phase", "WAITING"),
            "action_duration": self._action_result.get("action_duration", 0.0),
            "wrong_object": self._action_result.get("wrong_object"),
            # Temporal evidence
            "open_evidence": self.event_detector.open_evidence_frames,
            "closed_evidence": self.event_detector.closed_evidence_frames,
            # Object states
            "object_states": self.obj_tracker.get_summary() if self.mode in (RuntimeMode.ARMED, RuntimeMode.RUNNING) else {},
            "stage": 3,
        }

    def _record_fsm_event(self, fs_result, event):
        """Record FSM transition in event log."""
        if fs_result.step_id != self._prev_fsm_step:
            self._fsm_events.append({
                "step": fs_result.step_id,
                "step_name": self.STEP_NAMES.get(fs_result.step_id, "?"),
                "event_name": event.event_name,
                "confidence": round(event.confidence, 4),
                "temporal_boost": round(event.temporal_conf_boost, 4),
                "is_correct": fs_result.is_correct,
                "has_error": fs_result.has_error,
                "error_type": fs_result.error_type,
                "message": fs_result.message,
            })
            self._prev_fsm_step = fs_result.step_id

    def _accept_event(self, event, now: float) -> bool:
        """Accept an event only if FSM validates the transition.

        CRITICAL: _accepted_events and _max_fsm_path are ONLY updated
        when the FSM actually accepts the transition. This prevents
        display/FSM desync that causes S-box regression.
        """
        event_name = event.event_name
        if event_name in self._accepted_events:
            return True

        # Ask FSM first — only accept if FSM says it's correct
        fs_result = self.fsm.validate_event(
            event_name, event.confidence, timestamp=now)
        self._record_fsm_event(fs_result, event)

        if not fs_result.is_correct:
            print(f"[SOPServer] Event REJECTED by FSM: {event_name} -> {fs_result.message}",
                  flush=True)
            return False

        # Confirm with EventDetector (updates ObjectStateTracker for placement events)
        self.event_detector.mark_event_accepted(event_name, obj_tracker=self.obj_tracker)
        self._accepted_events.append(event_name)
        current_path = [EVENT_TO_STEP_ID.get(e, 0) for e in self._accepted_events
                        if EVENT_TO_STEP_ID.get(e, 0) > 0]
        if len(current_path) >= len(self._max_fsm_path):
            self._max_fsm_path = current_path
        self._advance_expected_event()

        # ── Reset next object's tracking state ──
        # When advancing to a new expected event, the next target object
        # must be FRESHLY observed leaving its init position. Any prior
        # left_init_roi latch (from hand occlusion during previous steps)
        # is cleared here to prevent auto-yellow.
        from engine.action_segmenter import OBJECT_FOR_EVENT
        next_obj = OBJECT_FOR_EVENT.get(self._expected_event)
        if next_obj:
            self.obj_tracker.reset_object(next_obj)
            print(f"[SOPServer] Reset tracker for next object: {next_obj}", flush=True)

        print(f"[SOPServer] Event ACCEPTED: {event_name} path={current_path} "
              f"next={self._expected_event}", flush=True)
        return True

    def _advance_expected_event(self):
        """Advance expected_event to the next in sequence."""
        try:
            idx = self.EVENT_SEQUENCE.index(self._expected_event)
            if idx + 1 < len(self.EVENT_SEQUENCE):
                self._expected_event = self.EVENT_SEQUENCE[idx + 1]
        except ValueError:
            pass

# ═══════════════════════════════════════════════════════════════
# WebSocket server
# ═══════════════════════════════════════════════════════════════

async def handle_client(websocket, server, trace_dir=None):
    """Handle one client connection. Drop old frames if busy."""
    global _recv_count, _skip_count, _send_count

    client_addr = websocket.remote_address
    print(f"[Server] 客户端连接: {client_addr}")

    _processing = False
    last_frame_id = -1
    trace_f = None
    fps_samples = []

    if trace_dir:
        trace_path = Path(trace_dir)
        trace_path.mkdir(parents=True, exist_ok=True)
        trace_f = open(str(trace_path / "frames.jsonl"), "w", encoding="utf-8")
        print(f"[Trace] 保存到: {trace_path}")

    try:
        async for message in websocket:
            recv_time = time.time()

            # Skip if still processing previous frame
            if _processing:
                _skip_count += 1
                continue

            _processing = True
            _recv_count += 1

            try:
                # Handle text control messages
                if isinstance(message, str):
                    cmd = message.strip().lower()
                    if cmd == "reset":
                        if hasattr(server, 'reset'):
                            server.reset()
                            print(f"[Server] RESET -> PREVIEW mode")
                            await websocket.send(json.dumps({
                                "control_ack": "reset",
                                "mode": "PREVIEW",
                                "message": "System reset to PREVIEW",
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "control_ack": "reset",
                                "error": "Server does not support reset",
                            }))
                    elif cmd == "start":
                        if hasattr(server, 'try_start'):
                            result = server.try_start()
                            print(f"[Server] START command -> {result}")
                            await websocket.send(json.dumps({
                                "control_ack": "start",
                                **result,
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "control_ack": "start",
                                "error": "Server does not support start",
                            }))
                    elif cmd == "stop":
                        if hasattr(server, 'stop'):
                            server.stop()
                            print(f"[Server] STOP -> PREVIEW mode")
                            await websocket.send(json.dumps({
                                "control_ack": "stop",
                                "mode": "PREVIEW",
                                "message": "Stopped, back to PREVIEW",
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "control_ack": "stop",
                                "error": "Server does not support stop",
                            }))
                    else:
                        print(f"[Server] unknown cmd: {message}")
                    _processing = False
                    continue

                # Parse binary header
                if isinstance(message, bytes) and len(message) >= 12:
                    frame_id = struct.unpack('<I', message[:4])[0]
                    ts = struct.unpack('<d', message[4:12])[0]
                    jpeg_bytes = message[12:]

                    # Skip duplicate frame_id
                    if frame_id <= last_frame_id:
                        _processing = False
                        continue
                    last_frame_id = frame_id

                    # Process
                    result = server.process(jpeg_bytes, frame_id, ts)

                    # Add network info
                    result["recv_to_send_ms"] = round(
                        (time.time() - recv_time) * 1000, 1)

                    # Save trace
                    if trace_f:
                        trace_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fps_samples.append({
                            "frame_id": frame_id,
                            "server_fps": result.get("server_fps", 0),
                            "latency_ms": result.get("latency_ms", 0),
                            "recv_to_send_ms": result.get("recv_to_send_ms", 0),
                        })

                    # Send back
                    await websocket.send(json.dumps(result))
                    _send_count += 1
            except Exception as e:
                print(f"[Server] 处理帧错误: {e}")
                import traceback
                traceback.print_exc()
            finally:
                _processing = False

    except websockets.exceptions.ConnectionClosed:
        print(f"[Server] 客户端断开: {client_addr}")
    except Exception as e:
        print(f"[Server] 连接错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save summary traces on disconnect
        if trace_f:
            trace_f.close()
            # Save temporal_trace and fsm_events if available (Stage 3)
            if hasattr(server, '_temporal_trace') and server._temporal_trace:
                temporal_path = trace_path / "temporal_trace.json"
                with open(str(temporal_path), "w", encoding="utf-8") as ft:
                    json.dump(server._temporal_trace, ft, indent=2, ensure_ascii=False)
                print(f"[Trace] temporal_trace: {len(server._temporal_trace)} 条 → {temporal_path}")
            if hasattr(server, '_fsm_events') and server._fsm_events:
                fsm_path = trace_path / "fsm_trace.json"
                with open(str(fsm_path), "w", encoding="utf-8") as ff:
                    json.dump(server._fsm_events, ff, indent=2, ensure_ascii=False)
                print(f"[Trace] fsm_trace: {len(server._fsm_events)} 条 → {fsm_path}")
            if fps_samples:
                fps_path = trace_path / "fps_log.json"
                with open(str(fps_path), "w", encoding="utf-8") as ffps:
                    json.dump(fps_samples, ffps, indent=2, ensure_ascii=False)
                print(f"[Trace] fps_log: {len(fps_samples)} 条 → {fps_path}")
            print(f"[Trace] frames.jsonl 已保存到 {trace_path}")


async def print_stats():
    """Periodically print server stats."""
    while True:
        await asyncio.sleep(10)
        print(f"[Stats] recv={_recv_count} skip={_skip_count} "
              f"send={_send_count}")


async def main_async(args):
    global _recv_count, _skip_count, _send_count

    print("=" * 60)
    print(f"SOP 推理服务器 — Stage {args.stage}")
    print(f"  监听: ws://{args.host}:{args.port}")

    # ── Select server based on stage ──
    if args.stage == 1:
        print("  模式: Echo only (图像接收验证)")
        server = EchoServer()
    elif args.stage == 2:
        print(f"  模式: YOLO CUDA (imgsz={args.imgsz})")
        yolo_path = args.yolo or str(
            ROOT / "models" / "yolo_final_v1.pt")
        print(f"  YOLO: {yolo_path}")
        server = YOLOServer(yolo_path, conf=args.conf, imgsz=args.imgsz,
                           save_hard_samples=args.save_hard_samples,
                           use_pink_marker=args.use_pink)
    elif args.stage == 3:
        print(f"  模式: Full SOP Pipeline (imgsz={args.imgsz})")
        yolo_path = args.yolo or str(
            ROOT / "models" / "yolo_final_v1.pt")
        model_path = args.model or str(
            ROOT / "models" / "temporal" / "v2_90_tcn_bigru" /
            "checkpoints" / "best.pt")
        print(f"  YOLO: {yolo_path}")
        print(f"  Temporal: {model_path}")
        if args.save_hard_samples:
            print(f"  Hard samples: {args.save_hard_samples}")
        server = SOPServer(yolo_path, model_path,
                          conf=args.conf, imgsz=args.imgsz,
                          save_hard_samples=args.save_hard_samples,
                          use_pink_marker=args.use_pink)
    else:
        print(f"ERROR: Invalid stage {args.stage}")
        return

    print("=" * 60)
    print("\nWaiting for client connection...")

    stats_task = asyncio.create_task(print_stats())

    async def handler(websocket):
        await handle_client(websocket, server, trace_dir=args.save_traces)

    try:
        async with serve(handler, args.host, args.port,
                         max_size=5 * 1024 * 1024):
            await asyncio.Future()  # run forever
    except asyncio.CancelledError:
        pass
    finally:
        stats_task.cancel()
        print(f"\n[Server] 统计: recv={_recv_count} skip={_skip_count} "
              f"send={_send_count}")
        print("[Server] 退出")


def main():
    parser = argparse.ArgumentParser(description="SOP 实时推理服务器")
    parser.add_argument("--stage", type=int, default=1,
                        help="1=echo 2=YOLO 3=full pipeline")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--yolo", default=None, help="YOLO 模型路径")
    parser.add_argument("--model", default=None, help="时序模型路径")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO 置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO 输入尺寸")
    parser.add_argument("--save-hard-samples", default=None,
                        help="保存困难样本目录 (如 data/hard_samples/realcam_v1)")
    parser.add_argument("--save-traces", default=None,
                        help="保存完整trace日志目录 (如 reports/stage3_realcam_smoke/empty_60s)")
    parser.add_argument("--use-pink", action="store_true", default=False,
                        help="启用粉色纸标记检测 (默认关闭，使用白色区域引导搜索)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
