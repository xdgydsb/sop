"""
YOLO物体检测器 + ByteTrack追踪 + 物体轨迹状态

检测5类: box_closed, box_open, earphone, charger, green_bag
ByteTrack提供稳定track ID, 每物体维护轨迹状态:
  - velocity, in_init_region, stable_in_box_frames, touched_by_hand
"""
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque


def _bbox_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1.0)


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]              # (cx, cy)
    track_id: int = -1
    tracked: bool = False
    source: str = "yolo"                     # "yolo" | "pink_marker" | "pink_guided" | "fallback"

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def is_real(self) -> bool:
        """Real detection from current frame (YOLO or pink marker). Not tracker recovery."""
        return self.source in ("yolo", "pink_marker", "pink_guided", "fallback")


@dataclass
class TrackedObject:
    """单物体轨迹状态 — 跨帧维护语义信息"""
    track_id: int
    cls_name: str
    center: Tuple[float, float]              # 当前位置 (cx, cy) 像素坐标
    bbox: Tuple[float, float, float, float]
    confidence: float
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) 像素/帧
    init_center: Optional[Tuple[float, float]] = None  # 初始出现位置
    in_init_region: bool = True               # 是否仍在初始区域
    stable_in_box_frames: int = 0             # 中心点在盒内的连续帧数
    in_box: bool = False                      # 当前是否在盒内
    touched_by_hand: bool = False             # 当前帧是否有手接触
    hand_touch_frames: int = 0                # 连续被手接触的帧数
    frames_since_touch: int = 0               # 手离开后的帧数
    frames_lost: int = 0                      # 丢失帧数
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))  # 最近30帧中心点


class YOLODetector:
    """YOLO + ByteTrack: 检测 + 多目标追踪 + 轨迹状态"""

    def __init__(self, model_path: str, conf_thresh: float = 0.3,
                 iou_thresh: float = 0.45, device: str = "cuda",
                 track_max_lost: int = 90, use_tracker: bool = False,
                 init_region_radius: float = 60.0, imgsz: int = 640,
                 bbox_ema_alpha: float = 0.0, use_clahe: bool = True):
        self.device = device if device else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.use_tracker = use_tracker
        self.track_max_lost = track_max_lost           # max frames before deleting lost track
        self.recover_max_lost = min(track_max_lost + 60, 150)  # max frames for re-association
        self.init_region_radius = init_region_radius
        self.bbox_ema_alpha = bbox_ema_alpha
        self.class_names = ["box_closed", "box_open", "earphone",
                            "charger", "green_bag"]
        self.use_clahe = use_clahe
        if use_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 追踪状态
        self._tracked: Dict[int, TrackedObject] = {}
        self._next_track_id: int = 1000
        self.frame_count = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """检测一帧 + ByteTrack追踪 + 丢失track恢复 + 更新轨迹状态"""
        self.frame_count += 1
        h, w = frame.shape[:2]

        # ── CLAHE lighting normalization ──
        if self.use_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            frame_proc = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        else:
            frame_proc = frame

        # ALWAYS use predict() — ultralytics ByteTrack has an IndexError bug
        # (predictor.trackers is empty) that also infects subsequent predict()
        # calls on the same model instance after track() fails.
        results = self.model.predict(
            frame_proc, verbose=False, conf=self.conf_thresh,
            iou=self.iou_thresh, device=self.device,
            imgsz=self.imgsz,
        )[0]

        raw: List[Detection] = []
        current_detections: List[Tuple] = []

        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                bbox = (x1, y1, x2, y2)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                # 颜色验证 — boost only, NEVER reject (lighting varies)
                try:
                    if cls_name == "green_bag":
                        color_score = self._verify_green(frame, bbox)
                        if color_score > 0.5:
                            conf = max(conf, min(color_score, 0.88))
                    elif cls_name == "charger":
                        color_score = self._verify_white(frame, bbox)
                        if color_score > 0.35:
                            conf = max(conf, min(color_score * 0.9, 0.90))
                    elif cls_name == "earphone":
                        dark_score = self._verify_dark(frame, bbox)
                        if dark_score > 0.4:
                            conf = max(conf, min(dark_score, 0.85))
                except Exception:
                    pass

                current_detections.append((cls_name, bbox, center, conf))

        # ── Custom IoU-based track matching ──
        matched_track_ids: set = set()

        for cls_name, bbox, center, conf in current_detections:
            best_iou = 0.15
            best_tid = -1
            for tid, tobj in self._tracked.items():
                if tid in matched_track_ids:
                    continue
                if tobj.cls_name != cls_name:
                    continue
                iou = _bbox_iou(bbox, tobj.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid >= 0:
                matched_track_ids.add(best_tid)
                self._update_tracked(best_tid, cls_name, bbox, center, conf)
                tobj = self._tracked[best_tid]
                raw.append(Detection(
                    cls_id=self.class_names.index(cls_name) if cls_name in self.class_names else -1,
                    cls_name=cls_name, confidence=conf,
                    bbox=tobj.bbox, center=tobj.center,
                    track_id=best_tid, tracked=True,
                ))
            else:
                # Try to recover a recently-lost track before creating a new one.
                # Hand-occluded objects jump in position (IoU < 0.15) → without
                # recovery they get a new track_id, breaking identity for all
                # downstream modules (DetectionStabilizer, ObjectStateTracker).
                recovered_tid = self._try_recover_lost(cls_name, center)
                if recovered_tid >= 0:
                    matched_track_ids.add(recovered_tid)
                    self._update_tracked(recovered_tid, cls_name, bbox, center, conf)
                    tobj = self._tracked[recovered_tid]
                    raw.append(Detection(
                        cls_id=self.class_names.index(cls_name) if cls_name in self.class_names else -1,
                        cls_name=cls_name, confidence=conf,
                        bbox=tobj.bbox, center=tobj.center,
                        track_id=recovered_tid, tracked=True,
                    ))
                else:
                    tid = self._next_track_id
                    self._next_track_id += 1
                    self._update_tracked(tid, cls_name, bbox, center, conf)
                    tobj = self._tracked[tid]
                    raw.append(Detection(
                        cls_id=self.class_names.index(cls_name) if cls_name in self.class_names else -1,
                        cls_name=cls_name, confidence=conf,
                        bbox=tobj.bbox, center=tobj.center,
                        track_id=tid, tracked=True,
                    ))

        # Mark unmatched tracks as lost
        for tid in list(self._tracked.keys()):
            if tid not in matched_track_ids:
                tobj = self._tracked[tid]
                tobj.frames_lost += 1
                if tobj.frames_lost > self.track_max_lost:
                    del self._tracked[tid]

        # Recover recently-lost tracks via velocity prediction.
        # Extended from 3→15 frames so hand-occluded objects (0.5-0.75 sec)
        # survive the gap between being picked up and appearing in the box.
        # Confidence decays progressively: 0.5× at frame 1 → 0.15× at frame 15.
        for tid, tobj in self._tracked.items():
            if 0 < tobj.frames_lost <= 15 and tobj.center is not None:
                pred_center = (
                    tobj.center[0] + tobj.velocity[0] * tobj.frames_lost,
                    tobj.center[1] + tobj.velocity[1] * tobj.frames_lost,
                )
                bw = tobj.bbox[2] - tobj.bbox[0]
                bh = tobj.bbox[3] - tobj.bbox[1]
                pred_bbox = (
                    pred_center[0] - bw/2, pred_center[1] - bh/2,
                    pred_center[0] + bw/2, pred_center[1] + bh/2,
                )
                # Progressive decay: 0.55 at frame 1 → 0.18 at frame 15
                decay = max(0.18, 0.55 - 0.025 * tobj.frames_lost)
                cls_id = self.class_names.index(tobj.cls_name) if tobj.cls_name in self.class_names else -1
                raw.append(Detection(
                    cls_id=cls_id, cls_name=tobj.cls_name,
                    confidence=max(tobj.confidence * decay, 0.15),
                    bbox=pred_bbox, center=pred_center,
                    track_id=tid, tracked=False, source='track_recover',
                ))

        return raw

    def _try_recover_lost(self, cls_name: str, center: Tuple[float, float]) -> int:
        """Try to match a new detection with a recently lost track of the same class.

        Returns the old track_id if a match is found within proximity, else -1.
        Uses velocity-predicted position for matching — the lost track's last
        known position is extrapolated forward, so hand-occluded objects that
        re-appear ahead of their last known position can still be matched.
        """
        best_tid = -1
        best_dist = float('inf')
        for tid, tobj in self._tracked.items():
            if tobj.frames_lost <= 0 or tobj.frames_lost > self.recover_max_lost:
                continue
            if tobj.cls_name != cls_name:
                continue
            # Predict where the track would be now using velocity
            pred_cx = tobj.center[0] + tobj.velocity[0] * tobj.frames_lost
            pred_cy = tobj.center[1] + tobj.velocity[1] * tobj.frames_lost
            dist = np.sqrt((center[0] - pred_cx) ** 2 +
                          (center[1] - pred_cy) ** 2)
            obj_size = max(tobj.bbox[2] - tobj.bbox[0],
                          tobj.bbox[3] - tobj.bbox[1])
            # Gradually expand search radius as frames_lost increases
            # At 5 frames: ~obj_size * 3, at 30 frames: ~obj_size * 8
            radius_mult = 3.0 + tobj.frames_lost * 0.18
            max_dist = max(obj_size * radius_mult, 180.0)
            if dist < max_dist and dist < best_dist:
                best_dist = dist
                best_tid = tid
        return best_tid

    def _update_tracked(self, track_id: int, cls_name: str,
                        bbox: Tuple, center: Tuple, conf: float):
        """更新单物体轨迹状态"""
        prev = self._tracked.get(track_id)
        vx, vy = 0.0, 0.0

        # ── EMA bbox smoothing ──
        if prev is not None and self.bbox_ema_alpha < 1.0:
            alpha = self.bbox_ema_alpha
            bbox = (
                alpha * bbox[0] + (1 - alpha) * prev.bbox[0],
                alpha * bbox[1] + (1 - alpha) * prev.bbox[1],
                alpha * bbox[2] + (1 - alpha) * prev.bbox[2],
                alpha * bbox[3] + (1 - alpha) * prev.bbox[3],
            )
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            vx = center[0] - prev.center[0]
            vy = center[1] - prev.center[1]
        elif prev is not None:
            vx = center[0] - prev.center[0]
            vy = center[1] - prev.center[1]

        tobj = TrackedObject(
            track_id=track_id,
            cls_name=cls_name,
            center=center,
            bbox=bbox,
            confidence=conf,
            velocity=(vx, vy),
            init_center=prev.init_center if prev else center,
            in_init_region=prev.in_init_region if prev else True,
            stable_in_box_frames=prev.stable_in_box_frames if prev else 0,
            in_box=prev.in_box if prev else False,
            touched_by_hand=False,  # 由 HandDetector 跨模块设置
            hand_touch_frames=prev.hand_touch_frames if prev else 0,
            frames_since_touch=(prev.frames_since_touch + 1) if prev else 0,
            frames_lost=0,
            trajectory=prev.trajectory if prev else deque(maxlen=30),
        )

        # 判断是否离开初始区域
        if tobj.init_center is not None:
            dist_from_init = np.sqrt(
                (center[0] - tobj.init_center[0]) ** 2 +
                (center[1] - tobj.init_center[1]) ** 2
            )
            if dist_from_init > self.init_region_radius:
                tobj.in_init_region = False

        tobj.trajectory.append(center)
        self._tracked[track_id] = tobj

    def update_in_box_status(self, track_id: int, in_box: bool):
        """由外部(PhysicalStateEngine)调用来更新入盒状态"""
        if track_id in self._tracked:
            tobj = self._tracked[track_id]
            tobj.in_box = in_box
            if in_box:
                tobj.stable_in_box_frames += 1
            else:
                tobj.stable_in_box_frames = max(0, tobj.stable_in_box_frames - 1)

    def update_hand_touch(self, track_id: int, touched: bool):
        """由外部(HandDetector)调用来更新手接触状态"""
        if track_id in self._tracked:
            tobj = self._tracked[track_id]
            tobj.touched_by_hand = touched
            if touched:
                tobj.hand_touch_frames += 1
                tobj.frames_since_touch = 0
            else:
                tobj.hand_touch_frames = 0
                tobj.frames_since_touch += 1

    def get_tracked(self, track_id: int) -> Optional[TrackedObject]:
        return self._tracked.get(track_id)

    def get_tracked_by_name(self, cls_name: str) -> List[TrackedObject]:
        """获取指定类别的所有被追踪物体"""
        return [t for t in self._tracked.values()
                if t.cls_name == cls_name and t.frames_lost < 10]

    @property
    def tracked_objects(self) -> Dict[int, TrackedObject]:
        return self._tracked

    def get_box_state(self, detections: List[Detection]) -> Tuple[str, float]:
        """盒子状态: 'open', 'closed', 'unknown'

        Cross-class NMS: when box_open and box_closed overlap on the same
        physical box (IoU ≥ 0.4), suppress the lower-confidence class.
        This prevents YOLO from outputting both classes on one box and
        causing rapid open↔closed oscillation.
        """
        direct = [d for d in detections if not d.tracked]
        box_open = max([d for d in direct if d.cls_name == "box_open"],
                       key=lambda d: d.confidence, default=None)
        box_closed = max([d for d in direct if d.cls_name == "box_closed"],
                         key=lambda d: d.confidence, default=None)

        open_conf = box_open.confidence if box_open else 0
        closed_conf = box_closed.confidence if box_closed else 0

        # ── Cross-class NMS ──
        if box_open is not None and box_closed is not None:
            iou = _bbox_iou(box_open.bbox, box_closed.bbox)
            if iou >= 0.4:
                # Same physical box detected as both open and closed.
                # Suppress the less confident one.
                if closed_conf > open_conf:
                    open_conf = closed_conf * 0.3  # heavily penalize
                else:
                    closed_conf = open_conf * 0.3

        if open_conf > closed_conf and open_conf > 0.2:
            return ("open", open_conf)
        elif closed_conf > 0 and closed_conf >= open_conf:
            return ("closed", closed_conf)
        return ("unknown", 0)

    def get_box_bbox(self, detections: List[Detection]) -> Optional[Tuple]:
        """盒子bbox (优先open)"""
        for name in ["box_open", "box_closed"]:
            d = max([d for d in detections if d.cls_name == name],
                    key=lambda d: d.confidence, default=None)
            if d:
                return d.bbox
        return None

    def get_objects_in_box(self, detections, box_bbox):
        """Return list of object names whose center is inside box_bbox."""
        if box_bbox is None:
            return []
        bx1, by1, bx2, by2 = box_bbox
        objects = []
        for d in detections:
            if d.cls_name in ("box_open", "box_closed"):
                continue
            cx, cy = d.center
            if bx1 <= cx <= bx2 and by1 <= cy <= by2 and d.confidence > 0.2:
                objects.append(d.cls_name)
        return objects

    def compute_in_box_ratio(self, obj_bbox: Tuple, box_bbox: Tuple) -> float:
        """计算物体bbox在盒子bbox内的面积占比 (bbox近似)"""
        ox1, oy1, ox2, oy2 = obj_bbox
        bx1, by1, bx2, by2 = box_bbox
        obj_area = max(1, (ox2 - ox1) * (oy2 - oy1))

        ix1, iy1 = max(ox1, bx1), max(oy1, by1)
        ix2, iy2 = min(ox2, bx2), min(oy2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return min(1.0, inter / obj_area)

    def get_best_detection(self, detections, cls_name):
        """Get highest-confidence detection for a given class name."""
        candidates = [d for d in detections if d.cls_name == cls_name]
        return max(candidates, key=lambda d: d.confidence, default=None)

    def detect_crop(self, frame: np.ndarray,
                    crop_roi: Tuple[float, float, float, float],
                    conf_thresh: float = 0.06) -> List[Detection]:
        """Run YOLO on a cropped region of the frame at lower confidence.

        Used for focused detection when an object is expected to be inside
        or near the box but the full-frame YOLO pass missed it. The crop
        provides higher effective resolution of the target area.

        Args:
            frame: full BGR frame
            crop_roi: (x1, y1, x2, y2) in full-frame coordinates
            conf_thresh: lower than global to catch hard-to-see in-box objects

        Returns:
            List of Detection objects with full-frame coordinates
        """
        h, w = frame.shape[:2]
        cx1, cy1, cx2, cy2 = [max(0, int(v)) for v in crop_roi]
        cx2, cy2 = min(w, cx2), min(h, cy2)
        if cx2 - cx1 < 20 or cy2 - cy1 < 20:
            return []

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return []

        # CLAHE on crop
        if self.use_clahe:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            crop_proc = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        else:
            crop_proc = crop

        # Run at half the global confidence threshold on the zoomed crop
        results = self.model.predict(
            crop_proc, verbose=False,
            conf=max(conf_thresh, 0.04),
            iou=self.iou_thresh, device=self.device,
            imgsz=max(crop_proc.shape[:2]),  # match crop size
        )[0]

        dets = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                # Transform back to full-frame coordinates
                full_bbox = (
                    bx1 + cx1, by1 + cy1,
                    bx2 + cx1, by2 + cy1,
                )
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                center = ((full_bbox[0] + full_bbox[2]) / 2,
                         (full_bbox[1] + full_bbox[3]) / 2)
                # Color verification boost only
                try:
                    if cls_name == "green_bag":
                        cs = self._verify_green(frame, full_bbox)
                        if cs > 0.5:
                            conf = max(conf, min(cs, 0.88))
                    elif cls_name == "charger":
                        cs = self._verify_white(frame, full_bbox)
                        if cs > 0.35:
                            conf = max(conf, min(cs * 0.9, 0.90))
                    elif cls_name == "earphone":
                        ds = self._verify_dark(frame, full_bbox)
                        if ds > 0.4:
                            conf = max(conf, min(ds, 0.85))
                except Exception:
                    pass
                dets.append(Detection(
                    cls_id=cls_id, cls_name=cls_name, confidence=conf,
                    bbox=full_bbox, center=center,
                    track_id=-1, tracked=False, source='crop',
                ))
        return dets

    def reset_tracking(self):
        self._tracked.clear()
        self.frame_count = 0
        # Reset ByteTrack's internal state for fresh tracking session
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            self.model.predictor.trackers = []
            self.model.predictor.track_history = {}

    def _verify_green(self, frame, bbox) -> float:
        """Verify if bbox region contains green (transparent green bag).

        Adaptive: uses frame's mean brightness to adjust HSV thresholds.
        In bright scenes, requires more saturated green. In dark scenes, more lenient.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        # Adaptive saturation threshold based on overall frame brightness
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(frame_gray)
        # Dark scene → lower saturation requirement; bright scene → higher
        sat_min = max(20, min(60, 60 - (mean_brightness - 100) * 0.25))

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([30, int(sat_min), 25]),
                            np.array([95, 255, 240]))
        ratio = np.count_nonzero(green) / green.size

        if ratio < 0.03:
            return 0.0
        return min(1.0, ratio / 0.12)

    def _verify_white(self, frame, bbox) -> float:
        """Verify if bbox region is a white charger.

        Uses HSV white detection + brightness + texture (dark text/lines on charger).
        Adaptive to lighting: adjusts bright threshold based on frame mean.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        # Adaptive brightness threshold
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(frame_gray)
        v_min = int(max(110, 165 - max(0, (mean_brightness - 80) * 0.4)))

        # HSV white detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_hsv = cv2.inRange(hsv, np.array([0, 0, v_min], dtype=np.uint8),
                                np.array([180, 55, 255], dtype=np.uint8))
        hsv_ratio = np.count_nonzero(white_hsv) / white_hsv.size

        # Grayscale bright pixel ratio (adaptive)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright_thr = min(180, max(130, mean_brightness * 1.2))
        bright_px = np.count_nonzero(gray > bright_thr) / gray.size

        # Texture: charger has dark text/lines → some variance
        std = np.std(gray.astype(np.float32))
        has_texture = min(0.25, max(0.0, (std - 12) / 60)) if 12 < std < 75 else 0.0

        # Edge density: charger has rectangular edges + metal pins
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.count_nonzero(edges) / edges.size
        has_structure = min(0.2, edge_density * 2.5) if edge_density > 0.03 else 0.0

        # Combine: white area + brightness + texture + structure
        score = hsv_ratio * 0.4 + bright_px * 0.25 + has_texture + has_structure
        return min(1.0, score / 0.6) if score > 0.20 else 0.0

    def _verify_dark(self, frame, bbox) -> float:
        """Verify if bbox region is a dark/black object (earphone case).

        Checks: low brightness, low saturation, moderate texture.
        Earphone case is matte black/dark gray with subtle texture.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        # Dark region: low value in HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]),
                                np.array([180, 120, 100]))
        dark_ratio = np.count_nonzero(dark_mask) / dark_mask.size

        # Grayscale statistics
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_gray = np.mean(gray)
        std_gray = np.std(gray.astype(np.float32))

        # Earphone is mostly dark (< 80 mean) with some texture (std 10-40)
        darkness_score = max(0.0, 1.0 - mean_gray / 100.0)
        texture_ok = 0.15 if 8 < std_gray < 50 else 0.0

        # Aspect ratio: earphone case is roughly 1:1 to 2:1
        bw, bh = x2 - x1, y2 - y1
        aspect = bw / max(bh, 1)
        shape_ok = 0.1 if 0.6 < aspect < 2.2 else 0.0

        score = dark_ratio * 0.35 + darkness_score * 0.35 + texture_ok + shape_ok
        return min(1.0, score / 0.65) if score > 0.25 else 0.0
