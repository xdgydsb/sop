"""Hikvision相机预览 & OpenCV回退"""
import cv2
import sys
import time
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from camera.hikvision_camera import HikvisionCamera, is_hikvision_sdk_available

running = True

def on_exit(sig=None, frame=None):
    global running
    running = False

signal.signal(signal.SIGINT, on_exit)
signal.signal(signal.SIGTERM, on_exit)

if __name__ == "__main__":
    cam = None
    method = "NONE"

    # 1. Try Hikvision MVS SDK
    if is_hikvision_sdk_available():
        print("[1/2] Trying Hikvision MVS SDK...")
        cam = HikvisionCamera(camera_index=0)
        if cam.open():
            method = "HIKVISION_MVS"
            print(f"[OK] {method}: {cam._width}x{cam._height}")
        else:
            print("[FAIL] Hikvision SDK open failed")
            cam = None

    # 2. Fallback to OpenCV
    if cam is None:
        print("[2/2] Trying OpenCV...")
        for idx in range(5):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Test read
                ret, frame = cap.read()
                if ret:
                    method = f"OPENCV_DSHOW_{idx}"
                    print(f"[OK] {method}: {frame.shape[1]}x{frame.shape[0]}")
                    cam = cap
                    break
                cap.release()
            else:
                cap.release()
        else:
            print("[FAIL] No camera found!")
            print("\n  1. Check USB connection")
            print("  2. Close MVS client (it holds the camera)")
            print("  3. Replug USB cable")
            sys.exit(1)

    print(f"\nPreview: {method} | Q=Quit\n")

    use_hik = isinstance(cam, HikvisionCamera)
    fps_history = []
    last_t = time.time()

    while running:
        if use_hik:
            ok, frame = cam.read()
        else:
            ok, frame = cam.read()

        if not ok or frame is None:
            print("  Frame lost...")
            time.sleep(0.01)
            continue

        # FPS
        now = time.time()
        fps_history.append(1.0 / max(now - last_t, 0.001))
        last_t = now
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps = sum(fps_history) / len(fps_history)

        # Display
        h, w = frame.shape[:2]
        display = frame.copy()
        cv2.putText(display, f"{method} | {w}x{h} | FPS:{fps:.1f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Q=Quit", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Camera Preview", display)

        if cv2.waitKey(1) in (ord('q'), 27):
            break

    if use_hik:
        cam.release()
    else:
        cam.release()
    cv2.destroyAllWindows()
    print("Done")
