"""
基于200张手工标注 → 训练初步YOLO → 自动标注更多帧 → 人工审核

流程:
  1. 用200张手工标注训练YOLOv8n (快速)
  2. 对所有OK/WR/yuanzi视频采样，用模型自动标注
  3. 只保留高置信度+颜色验证通过的标注
  4. 合并手工标注 → 最终YOLO训练集
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import glob, os, shutil

VIDEO_DIR = Path("/home/zhaowei/shabi/data")
MANUAL_DIR = Path("/home/zhaowei/shabi/data/manual_label_frames_yolo")  # 用户标注好后的YOLO格式
OUTPUT_DIR = Path("/home/zhaowei/shabi/data/yolo_dataset_v7")
AUTO_CONF = 0.5  # 自动标注最低置信度


def step1_train_small_yolo(manual_yolo_dir: Path):
    """用200张手工标注训练YOLOv8n ~30 epoch"""
    print("=" * 50)
    print("Step 1: Train YOLOv8n on manual annotations")
    print("=" * 50)

    # 准备data.yaml
    data_yaml = manual_yolo_dir / "data.yaml"
    if not data_yaml.exists():
        # 创建训练/验证划分 (90%/10%)
        img_dir = manual_yolo_dir / "images"
        lbl_dir = manual_yolo_dir / "labels"
        imgs = sorted(glob.glob(str(img_dir / "*.jpg")))
        n_val = max(1, len(imgs) // 10)

        (manual_yolo_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (manual_yolo_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (manual_yolo_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (manual_yolo_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(imgs):
            name = Path(img_path).stem
            split = "val" if i < n_val else "train"
            shutil.copy(img_path, manual_yolo_dir / split / "images" / f"{name}.jpg")
            lbl = lbl_dir / f"{name}.txt"
            if lbl.exists():
                shutil.copy(str(lbl), manual_yolo_dir / split / "labels" / f"{name}.txt")
            else:
                # 创建空标注文件
                (manual_yolo_dir / split / "labels" / f"{name}.txt").touch()

        data_yaml.write_text("""
path: %s
train: train/images
val: val/images
nc: 5
names: ['box_closed', 'box_open', 'earphone', 'charger', 'green_bag']
""" % str(manual_yolo_dir))

    # 训练YOLOv8n (快)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        name="manual_200_round1",
        device=0,
        exist_ok=True,
    )
    return model


def step2_auto_label(model, output_dir: Path, max_per_video: int = 30):
    """用训练好的模型对所有视频自动标注"""
    print("\n" + "=" * 50)
    print("Step 2: Auto-label frames from all videos")
    print("=" * 50)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)

    label_count = 0
    color_verified = 0

    for sub in ["yuanzi", "ok", "wr"]:
        vdir = VIDEO_DIR / sub
        if not vdir.exists():
            continue
        for vp in sorted(vdir.glob("*.avi"))[:60]:
            cap = cv2.VideoCapture(str(vp))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 10:
                cap.release()
                continue

            # 采样: 跳过开头/结尾, 均匀采样
            n_sample = min(max_per_video, total // 5)
            start, end = int(total * 0.05), int(total * 0.95)
            indices = np.linspace(start, end, n_sample, dtype=int)

            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    continue

                results = model(frame, conf=AUTO_CONF, device=0, verbose=False)[0]
                if results.boxes is None:
                    continue

                boxes = results.boxes
                labels_written = []
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    if conf < AUTO_CONF:
                        continue

                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    # 归一化
                    h, w = frame.shape[:2]
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    # 颜色验证
                    if cls in (3, 4):  # charger, green_bag
                        if not _color_check(frame, (x1, y1, x2, y2), cls):
                            continue
                        color_verified += 1

                    labels_written.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                if labels_written:
                    name = f"{sub}_{vp.stem}_f{fi}"
                    cv2.imwrite(str(output_dir / "images" / f"{name}.jpg"), frame)
                    with open(output_dir / "labels" / f"{name}.txt", "w") as f:
                        f.writelines(labels_written)
                    label_count += len(labels_written)

            cap.release()

    print(f"  Auto-labeled: {label_count} instances ({color_verified} color-verified)")
    return label_count


def _color_check(frame, bbox, cls_id):
    """颜色验证 - charger白色, green_bag绿色"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return True  # bbox太小，不做颜色过滤

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-dir", default="/home/zhaowei/shabi/data/manual_label_frames_yolo")
    parser.add_argument("--output-dir", default="/home/zhaowei/shabi/data/yolo_dataset_v7")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only auto-label")
    parser.add_argument("--model-path", default=None, help="Use existing model for auto-label")
    args = parser.parse_args()

    manual_dir = Path(args.manual_dir)
    output_dir = Path(args.output_dir)

    if not args.skip_train:
        model = step1_train_small_yolo(manual_dir)

        weights = sorted(Path("runs/detect/manual_200_round1/weights").glob("best*.pt"))
        if not weights:
            print("ERROR: Training failed, no weights found")
            return
        model_path = str(weights[0])
        print(f"  Best model: {model_path}")
        model = YOLO(model_path)
    elif args.model_path:
        model = YOLO(args.model_path)
    else:
        print("ERROR: --model-path required with --skip-train")
        return

    n_labels = step2_auto_label(model, output_dir)
    print(f"\nDone! {n_labels} auto-labels in {output_dir}")
    print(f"Review the labels, fix errors, then use for final YOLO training.")


if __name__ == "__main__":
    main()
