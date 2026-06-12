"""
用训练好的YOLOv8m自动标注所有视频帧
模型: runs/detect/manual_v1/weights/best.pt
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

VIDEO_DIR = Path("/home/zhaowei/shabi/data")
MODEL_PATH = "/home/zhaowei/shabi/data/runs/detect/manual_v1/weights/best.pt"
OUTPUT_DIR = Path("/home/zhaowei/shabi/data/yolo_auto_v8")
CONF = 0.5
MAX_PER_VIDEO = 30


def _color_check(frame, bbox, cls_id):
    """颜色验证 - charger白色, green_bag绿色"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return True
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return True
    if cls_id == 3:  # charger: white
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        ratio = np.count_nonzero(white) / white.size
        return ratio > 0.05
    elif cls_id == 4:  # green_bag: green
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (30, 50, 30), (95, 255, 255))
        ratio = np.count_nonzero(green) / green.size
        return ratio > 0.03
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(exist_ok=True)

    model = YOLO(MODEL_PATH)

    total_labels = 0
    color_dropped = 0

    for sub in ["yuanzi", "ok", "wr"]:
        vdir = VIDEO_DIR / sub
        if not vdir.exists():
            continue
        videos = sorted(vdir.glob("*.avi"))[:60]
        print(f"\n=== {sub}/ ({len(videos)} videos) ===")

        for vp in videos:
            cap = cv2.VideoCapture(str(vp))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 10:
                cap.release()
                continue

            n_sample = min(MAX_PER_VIDEO, total // 5)
            start, end = int(total * 0.05), int(total * 0.95)
            indices = np.linspace(start, end, max(n_sample, 1), dtype=int)

            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    continue

                results = model(frame, conf=CONF, device=0, verbose=False)[0]
                if results.boxes is None:
                    continue

                boxes = results.boxes
                labels_written = []

                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    if conf < CONF:
                        continue

                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                    # 颜色验证 (charger=3, green_bag=4)
                    if cls in (3, 4):
                        if not _color_check(frame, (x1, y1, x2, y2), cls):
                            color_dropped += 1
                            continue

                    h, w = frame.shape[:2]
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    labels_written.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                if labels_written:
                    name = f"{sub}_{vp.stem}_f{fi}"
                    cv2.imwrite(str(OUTPUT_DIR / "images" / f"{name}.jpg"), frame)
                    with open(OUTPUT_DIR / "labels" / f"{name}.txt", "w") as f:
                        f.write("\n".join(labels_written))
                    total_labels += len(labels_written)

            cap.release()

    print(f"\n=== Done ===")
    print(f"Total labels: {total_labels}")
    print(f"Color-filtered dropped: {color_dropped}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
