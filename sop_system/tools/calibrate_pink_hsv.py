"""
Pink marker HSV calibration tool.

Usage on server:
  ~/miniconda3/envs/three_env/bin/python tools/calibrate_pink_hsv.py \
    --image path/to/frame.jpg --out-dir reports/pink_calib/

Reads a saved camera frame and runs the pink detector, saving debug images
so you can tune HSV ranges. Also prints pixel-level HSV stats for the
largest pink region found.
"""
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.pink_marker_detector import PinkMarkerDetector


def calibrate(image_path: str, out_dir: str,
              h_low_1: int, s_low: int, v_low: int,
              h_high_1: int, h_low_2: int, h_high_2: int):
    """Run pink detector on a saved image and save debug output."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Cannot read {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"Image: {w}x{h}")

    # Override HSV defaults
    detector = PinkMarkerDetector(stable_frames=1, debug_dir=out_dir)
    detector.HSV_LOWER_1 = np.array([h_low_1, s_low, v_low])
    detector.HSV_UPPER_1 = np.array([h_high_1, 255, 255])
    detector.HSV_LOWER_2 = np.array([h_low_2, s_low, v_low])
    detector.HSV_UPPER_2 = np.array([h_high_2, 255, 255])

    print(f"HSV ranges:")
    print(f"  Range1: H=[{h_low_1},{h_high_1}] S=[{s_low},255] V=[{v_low},255]")
    print(f"  Range2: H=[{h_low_2},{h_high_2}] S=[{s_low},255] V=[{v_low},255]")

    # Run detection (full frame)
    result = detector.detect(frame, roi=None, roi_name="full")
    print(f"\nResult: present={result.present} area={result.area_px:.0f}px "
          f"area_ratio={result.area_ratio:.5f} conf={result.confidence:.2f}")

    if result.present and result.bbox:
        x1, y1, x2, y2 = [int(v) for v in result.bbox]
        print(f"Bbox: ({x1},{y1})-({x2},{y2})  size={x2-x1}x{y2-y1}")

    # Also test with box ROI if specified
    print(f"\nDebug images saved to: {out_dir}")
    print(f"Files: crop_*.jpg, hsv_H_*.jpg, hsv_S_*.jpg, "
          f"mask_raw_*.png, mask_morph_*.png, overlay_*.jpg")
    print(f"\nCheck overlay_*.jpg — yellow = detected pink, red = contour boundary")
    print(f"Check hsv_S_*.jpg — bright = high saturation (colorful), dark = low (washout)")
    print(f"\nIf no pink detected:")
    print(f"  1. Check overlay image for false negatives")
    print(f"  2. If saturation is very low (S < 20): lower --s-low")
    print(f"  3. If hue doesn't match: adjust --h-low-1 / --h-high-1")
    print(f"  4. Try: --s-low 10 --v-low 30 (most lenient)")


def main():
    parser = argparse.ArgumentParser(
        description="Pink marker HSV calibration from saved frame")
    parser.add_argument("--image", required=True,
                        help="Path to saved JPEG frame from camera")
    parser.add_argument("--out-dir", default="reports/pink_calib",
                        help="Output directory for debug images")
    parser.add_argument("--h-low-1", type=int, default=0,
                        help="Hue low for range 1 (warm pink)")
    parser.add_argument("--h-high-1", type=int, default=25,
                        help="Hue high for range 1")
    parser.add_argument("--h-low-2", type=int, default=150,
                        help="Hue low for range 2 (cool pink)")
    parser.add_argument("--h-high-2", type=int, default=180,
                        help="Hue high for range 2")
    parser.add_argument("--s-low", type=int, default=15,
                        help="Saturation low (lower = more lenient)")
    parser.add_argument("--v-low", type=int, default=40,
                        help="Value low (lower = darker ok)")
    args = parser.parse_args()

    calibrate(args.image, args.out_dir,
              args.h_low_1, args.s_low, args.v_low,
              args.h_high_1, args.h_low_2, args.h_high_2)


if __name__ == "__main__":
    main()
