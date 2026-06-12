"""
Pink marker detector — auxiliary charger detection via pink/red visual marker.

The charger has a pink paper strip attached. This detector finds the pink region
and creates a pseudo charger Detection when YOLO misses it.

Key constraints:
- Only searches within specified ROIs (not full frame)
- Uses morphology close to bridge gaps from handwritten text on the marker
- Does NOT do OCR — the handwritten "charger" text is irrelevant
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class MarkerResult:
    present: bool
    bbox: Optional[Tuple[float, float, float, float]]
    area_ratio: float
    confidence: float
    roi_name: str
    area_px: float = 0.0
    debug_info: Optional[Dict] = None


class PinkMarkerDetector:
    """Detect pink/red visual marker within specified ROIs."""

    # ── Calibrated HSV from frame_000315.jpg (2026-05-21) ──
    # Pink paper on white charger, wooden desk background
    HSV_LOWER_1 = np.array([0, 19, 32])
    HSV_UPPER_1 = np.array([30, 255, 255])
    HSV_LOWER_2 = np.array([150, 19, 32])
    HSV_UPPER_2 = np.array([180, 255, 255])

    MIN_AREA_PX = 30
    MIN_AREA_RATIO = 0.0005
    MORPH_CLOSE_KERNEL = 9
    MORPH_OPEN_KERNEL = 3

    # Calibrated: charger is ~1.8x wider, ~3.1x taller than pink paper
    EXPAND_W_RATIO = 1.8
    EXPAND_H_RATIO = 3.1

    def __init__(self, stable_frames: int = 2, debug_dir: str = None,
                 config_path: str = None):
        self._stable_frames = stable_frames
        self._history: List[Optional[Dict]] = []
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._debug_count = 0
        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        # Load calibrated config if available
        if config_path is None:
            # Default: look for calibration config next to this file or in reports/
            candidates = [
                Path(__file__).parent.parent / "reports" / "pink_hsv_config.json",
                Path("reports/pink_hsv_config.json"),
            ]
            for cp in candidates:
                if cp.exists():
                    config_path = str(cp)
                    break

        if config_path:
            self._load_config(config_path)

    def detect(self, frame: np.ndarray,
               roi: Optional[Tuple[int, int, int, int]] = None,
               roi_name: str = "full") -> MarkerResult:
        h, w = frame.shape[:2]

        if roi is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi]
            rx1 = max(0, rx1); ry1 = max(0, ry1)
            rx2 = min(w, rx2); ry2 = min(h, ry2)
            if rx2 <= rx1 + 5 or ry2 <= ry1 + 5:
                return MarkerResult(False, None, 0.0, 0.0, roi_name, 0.0)
            crop = frame[ry1:ry2, rx1:rx2]
            offset_x, offset_y = rx1, ry1
        else:
            crop = frame
            offset_x, offset_y = 0, 0

        ch, cw = crop.shape[:2]
        roi_area = max(1, ch * cw)

        # HSV → pink mask
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.HSV_LOWER_1, self.HSV_UPPER_1)
        mask2 = cv2.inRange(hsv, self.HSV_LOWER_2, self.HSV_UPPER_2)
        mask_raw = cv2.bitwise_or(mask1, mask2)

        # Morphology
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.MORPH_OPEN_KERNEL, self.MORPH_OPEN_KERNEL))
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.MORPH_CLOSE_KERNEL, self.MORPH_CLOSE_KERNEL))
        mask = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ── Debug save (first 10 frames per ROI) ──
        debug_info = None
        if self._debug_dir and self._debug_count < 10:
            self._save_debug(crop, hsv, mask_raw, mask, contours,
                            roi_name, ch, cw)
            self._debug_count += 1
            debug_info = {"debug_frame": self._debug_count}

        if not contours:
            self._update_history(None)
            return MarkerResult(False, None, 0.0, 0.0, roi_name, 0.0, debug_info)

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        area_ratio = area / roi_area

        if area < self.MIN_AREA_PX or area_ratio < self.MIN_AREA_RATIO:
            self._update_history(None)
            return MarkerResult(False, None, area_ratio, 0.0, roi_name,
                              float(area), debug_info)

        # Bbox in full-frame coords
        bx, by, bw, bh = cv2.boundingRect(largest)
        fx1 = offset_x + bx
        fy1 = offset_y + by
        fx2 = fx1 + bw
        fy2 = fy1 + bh
        bbox = (float(fx1), float(fy1), float(fx2), float(fy2))

        confidence = min(1.0, area_ratio * 200)

        result = {"bbox": bbox, "area_ratio": area_ratio,
                  "confidence": confidence, "roi_name": roi_name}
        self._update_history(result)

        stable_count = sum(1 for h in self._history if h is not None)
        present = stable_count >= self._stable_frames
        return MarkerResult(present, bbox, area_ratio, confidence, roi_name,
                          float(area), debug_info)

    def _update_history(self, result: Optional[Dict]):
        self._history.append(result)
        if len(self._history) > max(self._stable_frames, 2):
            self._history.pop(0)

    def _save_debug(self, crop, hsv, mask_raw, mask_morph, contours,
                    roi_name, ch, cw):
        """Save debug images for HSV calibration."""
        tag = f"{roi_name}_{self._debug_count:03d}"
        d = self._debug_dir

        # Crop with overlay
        cv2.imwrite(str(d / f"crop_{tag}.jpg"), crop)

        # HSV visualization (just H and S channels)
        hsv_viz = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cv2.imwrite(str(d / f"hsv_H_{tag}.jpg"), hsv_viz[:, :, 0])  # Hue
        cv2.imwrite(str(d / f"hsv_S_{tag}.jpg"), hsv_viz[:, :, 1])  # Saturation

        # Masks
        cv2.imwrite(str(d / f"mask_raw_{tag}.png"), mask_raw * 255)
        cv2.imwrite(str(d / f"mask_morph_{tag}.png"), mask_morph * 255)

        # Mask overlay on crop
        overlay = crop.copy()
        overlay[mask_morph > 0] = (0, 255, 255)  # yellow highlight
        blended = cv2.addWeighted(crop, 0.6, overlay, 0.4, 0)
        for c in contours:
            cv2.drawContours(blended, [c], -1, (0, 0, 255), 2)
        cv2.imwrite(str(d / f"overlay_{tag}.jpg"), blended)

        # Print HSV stats for the largest pink region
        if contours and len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            mask_contour = np.zeros((ch, cw), dtype=np.uint8)
            cv2.drawContours(mask_contour, [largest], -1, 255, -1)
            hsv_vals = hsv[mask_contour > 0]
            if len(hsv_vals) > 0:
                h_mean = float(np.mean(hsv_vals[:, 0]))
                s_mean = float(np.mean(hsv_vals[:, 1]))
                v_mean = float(np.mean(hsv_vals[:, 2]))
                h_min, s_min, v_min = hsv_vals.min(axis=0).tolist()
                h_max, s_max, v_max = hsv_vals.max(axis=0).tolist()
                print(f"[PinkDebug {tag}] HSV stats in largest contour "
                      f"({len(hsv_vals)} px):")
                print(f"  Mean: H={h_mean:.0f} S={s_mean:.0f} V={v_mean:.0f}")
                print(f"  Range: H=[{h_min:.0f},{h_max:.0f}] "
                      f"S=[{s_min:.0f},{s_max:.0f}] V=[{v_min:.0f},{v_max:.0f}]")
                print(f"  Suggested lower=[{max(0,h_min-5):.0f},{max(0,s_min-10):.0f},{max(0,v_min-20):.0f}]")
                print(f"  Suggested upper=[{min(180,h_max+5):.0f},255,255]")

    def expand_bbox(self, bbox: Tuple[float, float, float, float],
                    frame_w: int, frame_h: int,
                    box_bbox: Optional[Tuple] = None
                    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        mw = max(x2 - x1, 1)
        mh = max(y2 - y1, 1)

        ew = mw * self.EXPAND_W_RATIO
        eh = mh * self.EXPAND_H_RATIO

        ex1 = max(0.0, cx - ew / 2)
        ey1 = max(0.0, cy - eh / 2)
        ex2 = min(float(frame_w), cx + ew / 2)
        ey2 = min(float(frame_h), cy + eh / 2)

        if box_bbox is not None:
            bx1, by1, bx2, by2 = [float(v) for v in box_bbox]
            margin = 10.0
            ex1 = max(ex1, bx1 + margin)
            ey1 = max(ey1, by1 + margin)
            ex2 = min(ex2, bx2 - margin)
            ey2 = min(ey2, by2 - margin)

        return (ex1, ey1, ex2, ey2)

    def reset(self):
        self._history.clear()
        self._debug_count = 0

    def _load_config(self, config_path: str):
        """Load HSV calibration from JSON config file."""
        import json
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            self.HSV_LOWER_1 = np.array(cfg["hsv_lower_1"])
            self.HSV_UPPER_1 = np.array(cfg["hsv_upper_1"])
            self.HSV_LOWER_2 = np.array(cfg["hsv_lower_2"])
            self.HSV_UPPER_2 = np.array(cfg["hsv_upper_2"])
            if "expand_w_ratio" in cfg:
                self.EXPAND_W_RATIO = cfg["expand_w_ratio"]
            if "expand_h_ratio" in cfg:
                self.EXPAND_H_RATIO = cfg["expand_h_ratio"]
            if "min_area_px" in cfg:
                self.MIN_AREA_PX = cfg["min_area_px"]
            if "morph_close_kernel" in cfg:
                self.MORPH_CLOSE_KERNEL = cfg["morph_close_kernel"]
            print(f"[PinkMarker] Loaded config: {config_path}")
            print(f"  HSV1: {self.HSV_LOWER_1} ~ {self.HSV_UPPER_1}")
            print(f"  HSV2: {self.HSV_LOWER_2} ~ {self.HSV_UPPER_2}")
            print(f"  Expand: w={self.EXPAND_W_RATIO}, h={self.EXPAND_H_RATIO}")
        except Exception as e:
            print(f"[PinkMarker] WARNING: Could not load config {config_path}: {e}")
            print(f"[PinkMarker] Using built-in defaults")
