"""
Interactive pink marker + charger calibration tool.

Usage (local Windows):
  python tools/calibrate_pink_interactive.py --image frame_XXXXXX.jpg

Two-step annotation:
  Step 1: Draw rectangle around the PINK PAPER → calibrate HSV
  Step 2: Draw rectangle around the FULL CHARGER → calibrate bbox expansion ratios

Controls:
  Drag mouse = draw rectangle
  SPACE     = confirm current selection
  R         = reset current selection
  ESC       = quit

Output: reports/pink_hsv_config.json
"""
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Mouse state ──
_drawing = False
_start = (-1, -1)
_end = (-1, -1)
_selected = None  # (x1, y1, x2, y2)
_img = None
_img_display = None

# ── Results ──
_pink_rois = []      # [(roi, hsv_stats), ...]
_charger_rois = []   # [(roi, None), ...]
_phase = "pink"      # "pink" or "charger"


def _mouse(event, x, y, flags, param):
    global _drawing, _start, _end, _selected
    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _start = (x, y)
        _end = (x, y)
        _selected = None
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        _end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        _end = (x, y)
        if abs(_start[0] - _end[0]) > 5 and abs(_start[1] - _end[1]) > 5:
            _selected = (
                min(_start[0], _end[0]), min(_start[1], _end[1]),
                max(_start[0], _end[0]), max(_start[1], _end[1]),
            )


def _analyze_pink_roi(x1, y1, x2, y2):
    """Extract HSV stats from pink paper region."""
    crop = _img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].flatten(), hsv[:, :, 1].flatten(), hsv[:, :, 2].flatten()

    return {
        "roi": (x1, y1, x2, y2),
        "w": x2 - x1, "h": y2 - y1,
        "h_p5": int(np.percentile(h, 5)), "h_p95": int(np.percentile(h, 95)),
        "h_min": int(h.min()), "h_max": int(h.max()),
        "s_p5": int(np.percentile(s, 5)), "s_p95": int(np.percentile(s, 95)),
        "s_min": int(s.min()), "s_mean": float(np.mean(s)),
        "v_p5": int(np.percentile(v, 5)), "v_p95": int(np.percentile(v, 95)),
        "v_min": int(v.min()), "v_mean": float(np.mean(v)),
        "pixels": len(h),
    }


def _compute_final_config():
    """Compute optimal HSV + bbox expansion from all samples."""
    if not _pink_rois:
        print("ERROR: No pink paper samples collected!")
        return None

    # ── HSV from pink paper samples ──
    all_h_p5 = [s["h_p5"] for _, s in _pink_rois]
    all_h_p95 = [s["h_p95"] for _, s in _pink_rois]
    all_s_p5 = [s["s_p5"] for _, s in _pink_rois]
    all_v_p5 = [s["v_p5"] for _, s in _pink_rois]

    h_low = max(0, min(all_h_p5) - 10)
    h_high = min(180, max(all_h_p95) + 10)
    s_low = max(5, min(all_s_p5) - 15)
    v_low = max(10, min(all_v_p5) - 40)

    # Determine if hue wraps (red region)
    needs_wrap = (h_low < 5 or h_high > 175)

    if needs_wrap:
        h_low_1 = max(0, h_low)
        h_high_1 = min(30, h_high)
        h_low_2 = max(150, h_low)
        h_high_2 = 180
        config = {
            "hsv_lower_1": [int(h_low_1), int(s_low), int(v_low)],
            "hsv_upper_1": [int(h_high_1), 255, 255],
            "hsv_lower_2": [int(h_low_2), int(s_low), int(v_low)],
            "hsv_upper_2": [int(h_high_2), 255, 255],
        }
    else:
        config = {
            "hsv_lower_1": [int(h_low), int(s_low), int(v_low)],
            "hsv_upper_1": [int(h_high), 255, 255],
            "hsv_lower_2": [180, 255, 255],  # unused
            "hsv_upper_2": [180, 255, 255],
        }

    # ── Bbox expansion from pink→charger size ratio ──
    if _charger_rois:
        ratios_w = []
        ratios_h = []
        # Match each charger roi to nearest pink roi
        for (cr, _), (pr, _) in zip(_charger_rois, _pink_rois):
            pw = pr[2] - pr[0]
            ph = pr[3] - pr[1]
            cw = cr[2] - cr[0]
            ch = cr[3] - cr[1]
            if pw > 0 and ph > 0:
                ratios_w.append(cw / pw)
                ratios_h.append(ch / ph)
        if ratios_w:
            config["expand_w_ratio"] = round(float(np.mean(ratios_w)), 1)
            config["expand_h_ratio"] = round(float(np.mean(ratios_h)), 1)
    else:
        config["expand_w_ratio"] = 2.5
        config["expand_h_ratio"] = 2.2

    # ── Other params ──
    config["min_area_px"] = 30
    config["morph_close_kernel"] = 9
    config["morph_open_kernel"] = 3
    config["stable_frames"] = 2
    config["num_pink_samples"] = len(_pink_rois)
    config["num_charger_samples"] = len(_charger_rois)

    return config


def run(image_path: str):
    global _img, _img_display, _phase, _selected
    global _drawing, _start, _end
    global _pink_rois, _charger_rois

    _img = cv2.imread(image_path)
    if _img is None:
        print(f"ERROR: Cannot read {image_path}")
        return

    h, w = _img.shape[:2]
    print(f"Image: {w}x{h}\n")
    print("=" * 60)
    print("STEP 1: Draw box around PINK PAPER (粉红纸条)")
    print("  Drag mouse → SPACE to confirm → repeat 1-2 times for best results")
    print("  Then press TAB to switch to charger annotation")
    print("=" * 60)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", min(w, 1280), min(h, 900))
    cv2.setMouseCallback("Calibration", _mouse)

    while True:
        _img_display = _img.copy()

        # Phase indicator
        if _phase == "pink":
            phase_text = "PINK PAPER — draw box around pink paper, SPACE=confirm, TAB=next"
            phase_color = (255, 0, 255)
        else:
            phase_text = "CHARGER — draw box around FULL charger, SPACE=confirm, ESC=done"
            phase_color = (0, 255, 255)
        cv2.putText(_img_display, phase_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, phase_color, 2)

        # Draw current selection
        if _drawing:
            cv2.rectangle(_img_display, _start, _end, (0, 0, 255), 1)
        elif _selected and _phase == "pink":
            cv2.rectangle(_img_display, (_selected[0], _selected[1]),
                          (_selected[2], _selected[3]), (255, 0, 255), 2)
        elif _selected and _phase == "charger":
            cv2.rectangle(_img_display, (_selected[0], _selected[1]),
                          (_selected[2], _selected[3]), (0, 255, 255), 2)

        # Draw previous annotations
        for i, (roi, _) in enumerate(_pink_rois):
            cv2.rectangle(_img_display, (roi[0], roi[1]), (roi[2], roi[3]),
                          (255, 0, 200), 1)
            cv2.putText(_img_display, f"P{i+1}", (roi[0], roi[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 200), 1)
        for i, (roi, _) in enumerate(_charger_rois):
            cv2.rectangle(_img_display, (roi[0], roi[1]), (roi[2], roi[3]),
                          (0, 200, 255), 1)
            cv2.putText(_img_display, f"C{i+1}", (roi[0], roi[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

        # Count display
        y_off = 50
        cv2.putText(_img_display, f"Pink samples: {len(_pink_rois)}  "
                    f"Charger samples: {len(_charger_rois)}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Calibration", _img_display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('r'):
            _selected = None
            _drawing = False
        elif key == 9:  # TAB — switch phase
            if _phase == "pink" and _pink_rois:
                _phase = "charger"
                _selected = None
                _drawing = False
                print("\n" + "=" * 60)
                print("STEP 2: Draw box around FULL CHARGER (整个白插头)")
                print("  Draw the bounding box that should CONTAIN the charger")
                print("  This calibrates how much bigger the charger is than the pink paper")
                print("  SPACE to confirm each, ESC when done")
                print("=" * 60)
        elif key == 32 and _selected:  # SPACE
            if _phase == "pink":
                stats = _analyze_pink_roi(*_selected)
                _pink_rois.append((_selected, stats))
                print(f"\n[Pink #{len(_pink_rois)}] ROI=({_selected[0]},{_selected[1]})-"
                      f"({_selected[2]},{_selected[3]})  size={stats['w']}x{stats['h']}")
                print(f"  H p5={stats['h_p5']} p95={stats['h_p95']}  "
                      f"S p5={stats['s_p5']} mean={stats['s_mean']:.0f}  "
                      f"V p5={stats['v_p5']} mean={stats['v_mean']:.0f}")
                _selected = None
                _drawing = False
            elif _phase == "charger":
                _charger_rois.append((_selected, None))
                pw = _selected[2] - _selected[0]
                ph = _selected[3] - _selected[1]
                print(f"\n[Charger #{len(_charger_rois)}] ROI=({_selected[0]},{_selected[1]})-"
                      f"({_selected[2]},{_selected[3]})  size={pw}x{ph}")
                _selected = None
                _drawing = False

    cv2.destroyAllWindows()

    # ── Compute final config ──
    config = _compute_final_config()
    if config is None:
        return

    print(f"\n{'='*60}")
    print("FINAL CALIBRATION RESULT")
    print(f"{'='*60}")
    print(f"\n  HSV_LOWER_1 = np.array([{config['hsv_lower_1'][0]}, "
          f"{config['hsv_lower_1'][1]}, {config['hsv_lower_1'][2]}])")
    print(f"  HSV_UPPER_1 = np.array([{config['hsv_upper_1'][0]}, 255, 255])")
    print(f"  HSV_LOWER_2 = np.array([{config['hsv_lower_2'][0]}, "
          f"{config['hsv_lower_2'][1]}, {config['hsv_lower_2'][2]}])")
    print(f"  HSV_UPPER_2 = np.array([{config['hsv_upper_2'][0]}, 255, 255])")
    print(f"\n  expand_w_ratio = {config['expand_w_ratio']}")
    print(f"  expand_h_ratio = {config['expand_h_ratio']}")

    if _charger_rois and _pink_rois:
        # Show pink→charger spatial relationship
        for i, ((pr, _), (cr, _)) in enumerate(zip(_pink_rois, _charger_rois)):
            pcx = (pr[0] + pr[2]) / 2
            pcy = (pr[1] + pr[3]) / 2
            ccx = (cr[0] + cr[2]) / 2
            ccy = (cr[1] + cr[3]) / 2
            print(f"  Sample {i+1}: pink_center=({pcx:.0f},{pcy:.0f}) "
                  f"charger_center=({ccx:.0f},{ccy:.0f}) "
                  f"offset=({ccx-pcx:.0f},{ccy-pcy:.0f})")

    # Save config
    config_path = ROOT / "reports" / "pink_hsv_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(config_path), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n  Config saved → {config_path}")
    print(f"\n  Send this file to me and I'll update the detector code immediately.")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive pink marker + charger calibration")
    parser.add_argument("--image", required=True,
                        help="Path to saved JPEG frame")
    args = parser.parse_args()
    run(args.image)


if __name__ == "__main__":
    main()
