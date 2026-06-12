"""
Interactive ROI calibration tool for SOP system.

Usage:
  python tools/calibrate_rois.py [--camera HIK_CAMERA_INDEX]

Draw 5 ROIs in order:
  1. box_roi — the fixed box position (rejects out-of-area false positives)
  2. box_inner_roi — shrunken box interior for object placement detection
  3. earphone_init_roi — where earphone sits before use
  4. charger_init_roi — where charger sits before use
  5. green_bag_init_roi — where green bag sits before use

Controls:
  - Click and drag to draw ROI
  - SPACE to confirm current ROI and move to next
  - R to redraw current ROI
  - Q to quit
  - S to save current config

Saves to: configs/realcam_sop.yaml
"""
import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml


ROI_NAMES = [
    "box_roi",
    "box_inner_roi",
    "earphone_init_roi",
    "charger_init_roi",
    "green_bag_init_roi",
]

ROI_COLORS = [
    (0, 255, 0),     # green — box_roi
    (0, 200, 200),   # yellow — box_inner_roi
    (255, 100, 0),   # blue — earphone_init_roi
    (0, 100, 255),   # orange — charger_init_roi
    (255, 0, 255),   # magenta — green_bag_init_roi
]

OUTPUT_PATH = ROOT / "configs" / "realcam_sop.yaml"


def load_existing():
    """Load existing config if available."""
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(rois):
    """Save ROIs to config file."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    data["box_roi"] = list(rois.get("box_roi", [300, 150, 950, 650]))
    data["box_inner_roi"] = list(rois.get("box_inner_roi", [380, 200, 870, 600]))
    data["earphone_init_roi"] = list(rois.get("earphone_init_roi", [100, 200, 350, 450]))
    data["charger_init_roi"] = list(rois.get("charger_init_roi", [600, 100, 900, 350]))
    data["green_bag_init_roi"] = list(rois.get("green_bag_init_roi", [50, 400, 300, 650]))
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=True, allow_unicode=True)
    print(f"\n[Calibrate] Saved to {OUTPUT_PATH}")
    print(f"  box_roi:           {data['box_roi']}")
    print(f"  box_inner_roi:     {data['box_inner_roi']}")
    print(f"  earphone_init_roi: {data['earphone_init_roi']}")
    print(f"  charger_init_roi:  {data['charger_init_roi']}")
    print(f"  green_bag_init_roi: {data['green_bag_init_roi']}")


class ROICalibrator:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        self.rois = load_existing()
        self.current_roi_idx = 0
        self.drawing = False
        self.start_pt = None
        self.current_pt = None

    def run(self):
        cv2.namedWindow("ROI Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI Calibration", 1280, 720)

        print("\n" + "=" * 60)
        print("ROI Calibration Tool")
        print("=" * 60)
        print(f"Step 1/5: Draw {ROI_NAMES[0]} (box_roi)")
        print("  Click+Drag = draw, SPACE = confirm, R = redraw, Q = quit")
        print("=" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[Calibrate] Camera read failed")
                break

            display = frame.copy()
            h, w = display.shape[:2]

            # Draw all saved ROIs
            for i, name in enumerate(ROI_NAMES):
                roi = self.rois.get(name)
                if roi and len(roi) == 4:
                    color = ROI_COLORS[i]
                    cv2.rectangle(display,
                                  (int(roi[0]), int(roi[1])),
                                  (int(roi[2]), int(roi[3])),
                                  color, 2)
                    cv2.putText(display, name,
                                (int(roi[0]), int(roi[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw current drawing ROI
            if self.drawing and self.start_pt and self.current_pt:
                color = ROI_COLORS[self.current_roi_idx]
                cv2.rectangle(display, self.start_pt, self.current_pt, color, 2)

            # Status bar
            name = ROI_NAMES[self.current_roi_idx]
            status = f"Step {self.current_roi_idx + 1}/5: {name}"
            cv2.putText(display, status, (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "SPACE=confirm R=redraw S=save Q=quit",
                        (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("ROI Calibration", display)

            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                save_config(self.rois)
            elif key == ord('r'):
                # Clear current ROI
                name = ROI_NAMES[self.current_roi_idx]
                self.rois.pop(name, None)
                self.drawing = False
                self.start_pt = None
                print(f"  Redraw {name}")
            elif key == ord(' '):
                # Confirm current ROI
                name = ROI_NAMES[self.current_roi_idx]
                if name in self.rois:
                    print(f"  Confirmed {name}: {self.rois[name]}")
                else:
                    print(f"  WARNING: {name} not drawn yet!")
                self.current_roi_idx = (self.current_roi_idx + 1) % len(ROI_NAMES)
                self.drawing = False
                self.start_pt = None
                new_name = ROI_NAMES[self.current_roi_idx]
                print(f"  → Step {self.current_roi_idx + 1}/5: Draw {new_name}")

        self.cap.release()
        cv2.destroyAllWindows()

        # Auto-save on exit
        save_config(self.rois)
        print("\nCalibration complete. Config saved to configs/realcam_sop.yaml")

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_pt:
                x1, y1 = self.start_pt
                x2, y2 = x, y
                roi = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                name = ROI_NAMES[self.current_roi_idx]
                self.rois[name] = roi
                print(f"  {name}: {roi}")
            self.start_pt = None
            self.current_pt = None


def main():
    parser = argparse.ArgumentParser(description="SOP ROI Calibration Tool")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    args = parser.parse_args()

    calibrator = ROICalibrator(camera_index=args.camera)
    cv2.setMouseCallback("ROI Calibration", calibrator._mouse_callback)
    calibrator.run()


if __name__ == "__main__":
    main()
