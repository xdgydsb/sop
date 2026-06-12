"""
YOLO实时检测测试 — 使用电脑自带摄像头
验证 yolo_final_v1.pt 对5类物体的检测准确度
"""
import cv2
import torch
import time
import sys
from pathlib import Path
from collections import deque
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

# ── 5类物体 + 颜色 (BGR) ──
CLASS_NAMES = ["box_closed", "box_open", "earphone", "charger", "green_bag"]
CLASS_COLORS = {
    "box_closed": (128, 128, 128),   # 灰
    "box_open":   (0, 220, 220),     # 青
    "earphone":   (100, 100, 255),   # 浅红
    "charger":    (100, 255, 100),   # 浅绿
    "green_bag":  (100, 200, 100),   # 绿
}


def main():
    model_path = Path(__file__).parent / "models" / "yolo_final_v1.pt"
    if not model_path.exists():
        print(f"模型不存在: {model_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"加载YOLO模型: {model_path}")
    model = YOLO(str(model_path))

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头 (index 0)")
        print("尝试 index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("无法打开任何摄像头!")
            sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头: {actual_w}x{actual_h}")

    # ── 状态 ──
    fps_buffer = deque(maxlen=30)
    last_time = time.time()
    frame_count = 0
    save_screenshot = False

    print("=" * 60)
    print("YOLO 实时检测测试")
    print("  5类物体: box_closed | box_open | earphone | charger | green_bag")
    print("  Q=退出  S=截图  Space=暂停")
    print("=" * 60)

    win_name = "YOLO Live Detection - yolo_final_v1"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280 + 320, 720)

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                break
            frame_count += 1
            display = frame.copy()
        else:
            display = display.copy()  # keep last frame

        h, w = display.shape[:2]

        # ── YOLO推理 (暂停时跳过) ──
        if not paused:
            results = model(frame, verbose=False)

            # 统计检测结果
            detected = {}
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                        conf = float(box.conf[0])
                        if conf < 0.3:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        color = CLASS_COLORS.get(cls_name, (128, 128, 128))

                        # 画框
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        # 标签
                        label = f"{cls_name} {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(display, (x1, y1 - label_size[1] - 6),
                                    (x1 + label_size[0] + 4, y1), color, -1)
                        cv2.putText(display, label, (x1 + 2, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        if cls_name not in detected or conf > detected[cls_name]:
                            detected[cls_name] = conf

        # ── FPS ──
        now = time.time()
        fps_buffer.append(1.0 / max(now - last_time, 0.001))
        last_time = now
        fps = np.mean(fps_buffer) if fps_buffer else 0

        # ── 右侧信息面板 ──
        panel_w = 320
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 25)
        font = cv2.FONT_HERSHEY_SIMPLEX

        y = 20
        # 标题
        cv2.putText(panel, "YOLO Detection", (10, y + 15), font, 0.6, (0, 255, 255), 2)
        cv2.putText(panel, "yolo_final_v1.pt", (10, y + 38), font, 0.4, (180, 180, 180), 1)
        y += 55

        # 模型信息
        cv2.putText(panel, f"FPS: {fps:.1f}", (10, y + 15), font, 0.45, (0, 255, 0), 2)
        y += 25
        cv2.putText(panel, f"Frames: {frame_count}", (10, y + 15), font, 0.35, (150, 150, 150), 1)
        y += 25
        cv2.putText(panel, f"Device: {device}", (10, y + 15), font, 0.35, (150, 150, 150), 1)
        y += 35

        # 分隔线
        cv2.line(panel, (10, y), (panel_w - 10, y), (60, 60, 60), 1)
        y += 15

        # 各类检测状态
        cv2.putText(panel, "Detection Status:", (10, y + 15), font, 0.45, (200, 200, 200), 1)
        y += 30

        for cls_name in CLASS_NAMES:
            if not paused:
                conf = detected.get(cls_name, 0)
            else:
                conf = 0

            color = CLASS_COLORS.get(cls_name, (128, 128, 128))
            status_color = (0, 255, 0) if conf >= 0.5 else (
                (0, 200, 255) if conf >= 0.3 else (100, 100, 100))

            # 状态指示灯
            cv2.circle(panel, (18, y + 8), 6, status_color, -1)
            cv2.putText(panel, cls_name, (32, y + 15), font, 0.38, color, 1)

            # 置信度条
            bar_x = 160
            bar_w = int((panel_w - bar_x - 15) * min(conf, 1.0))
            cv2.rectangle(panel, (bar_x, y + 5), (bar_x + 100, y + 14), (40, 40, 40), -1)
            if bar_w > 0:
                cv2.rectangle(panel, (bar_x, y + 5), (bar_x + bar_w, y + 14), status_color, -1)
            cv2.putText(panel, f"{conf:.2f}", (bar_x + 104, y + 14), font, 0.32, status_color, 1)
            y += 22

        y += 10
        cv2.line(panel, (10, y), (panel_w - 10, y), (60, 60, 60), 1)
        y += 15

        # 检测统计
        if not paused:
            detected_count = sum(1 for c in detected.values() if c >= 0.3)
            total_found = sum(1 for c in detected.values() if c >= 0.5)
        else:
            detected_count = 0
            total_found = 0

        cv2.putText(panel, f"Detected (>=0.3): {detected_count}/5", (10, y + 15),
                   font, 0.38, (200, 200, 200), 1)
        y += 20
        cv2.putText(panel, f"Confident (>=0.5): {total_found}/5", (10, y + 15),
                   font, 0.38, (0, 255, 0), 1)
        y += 30

        # 操作提示
        cv2.line(panel, (10, y), (panel_w - 10, y), (60, 60, 60), 1)
        y += 15
        cv2.putText(panel, "Controls:", (10, y + 15), font, 0.4, (180, 180, 180), 1)
        y += 25
        cv2.putText(panel, "  Q - Quit", (10, y + 15), font, 0.35, (128, 128, 128), 1)
        y += 20
        cv2.putText(panel, "  S - Save Screenshot", (10, y + 15), font, 0.35, (128, 128, 128), 1)
        y += 20
        cv2.putText(panel, "  Space - Pause", (10, y + 15), font, 0.35, (128, 128, 128), 1)

        if paused:
            y += 30
            cv2.putText(panel, "PAUSED", (10, y + 20), font, 1.0, (0, 200, 255), 2)

        # 顶部状态条
        if not paused:
            detected_count = sum(1 for c in detected.values() if c >= 0.3)
            if detected_count >= 5:
                top_color = (0, 150, 0)
                top_text = "ALL 5 OBJECTS DETECTED - Ready for SOP"
            elif detected_count >= 3:
                top_color = (0, 150, 200)
                top_text = f"{detected_count}/5 objects detected - Adjust camera view"
            else:
                top_color = (0, 0, 180)
                top_text = f"Only {detected_count}/5 objects - Show objects to camera"
        else:
            top_color = (60, 60, 60)
            top_text = "PAUSED - Press Space to resume"

        bar_h = 35
        top_bar = np.zeros((bar_h, w + panel_w, 3), dtype=np.uint8)
        top_bar[:] = top_color
        cv2.putText(top_bar, top_text, (15, 25), font, 0.6, (255, 255, 255), 2)

        # ── 拼接显示 ──
        display = np.hstack([display, panel])
        display = np.vstack([top_bar, display])

        cv2.imshow(win_name, display)

        # ── 按键 ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"  截图已保存: {filename}")
        elif key == 32:  # Space
            paused = not paused
            if paused:
                print("  暂停")
            else:
                last_time = time.time()
                print("  继续")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n完成. 共处理 {frame_count} 帧.")


if __name__ == "__main__":
    main()
