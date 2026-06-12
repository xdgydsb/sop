"""
准备YOLO训练数据 — 从视频抽帧 + Grounding DINO自动标注 + 手工修正

策略:
1. 从data/yuanzi/1-5.avi各抽取10帧 (不同阶段, 确保覆盖动作全过程)
2. 从data/ok/中抽取代表性的帧 (不同光照、角度)
3. 使用Grounding DINO进行自动预标注 (零样本检测)
4. 输出YOLO格式标注, 供人工检查和修正
5. 生成dataset.yaml配置文件
"""
import cv2
import numpy as np
from pathlib import Path
import random
import yaml
import shutil
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path, step: int = 30,
                   max_frames: int = None, prefix: str = ""):
    """从视频中均匀抽帧"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 跳过开头和结尾 (视频开始和结束可能没有动作)
    start_frame = max(0, int(fps * 0.5))  # 跳过前0.5秒
    end_frame = min(total, int(total - fps * 0.3))  # 跳过后0.3秒

    if max_frames:
        step = max(1, (end_frame - start_frame) // max_frames)

    saved = []
    name = video_path.stem
    for i in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            out_name = f"{prefix}{name}_f{i:06d}.jpg"
            cv2.imwrite(str(output_dir / out_name), frame)
            saved.append(out_name)

    cap.release()
    return saved


def prepare_yuanzi_frames(yuanzi_dir: Path, output_dir: Path, frames_per_video: int = 50):
    """从yuanzi视频抽帧 (每个动作的独立视频, 全部作为训练数据)"""
    print("\n" + "="*60)
    print("从yuanzi视频抽帧 (单动作模板)")
    print("="*60)
    frames_dir = output_dir / "images" / "train"
    frames_dir.mkdir(parents=True, exist_ok=True)
    all_saved = []
    for i in range(1, 6):
        video = yuanzi_dir / f"{i}.avi"
        if video.exists():
            saved = extract_frames(video, frames_dir, max_frames=frames_per_video,
                                   prefix=f"yuanzi{i}_")
            print(f"  {video.name}: {len(saved)} frames")
            all_saved.extend(saved)
    return all_saved


def prepare_ok_frames(ok_dir: Path, output_dir: Path, max_videos: int = 50,
                      frames_per_video: int = 20):
    """从OK视频抽帧 (正确流程)"""
    print("\n" + "="*60)
    print(f"从OK视频抽帧 (正确流程, 最多{max_videos}个)")
    print("="*60)
    frames_dir = output_dir / "images" / "train"
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(ok_dir.glob("*.avi"))[:max_videos]
    all_saved = []
    for video in tqdm(videos, desc="Extracting OK frames"):
        prefix = f"ok_{video.stem[:20]}_"
        saved = extract_frames(video, frames_dir, max_frames=frames_per_video,
                               prefix=prefix)
        all_saved.extend(saved)
    print(f"  Total OK frames: {len(all_saved)}")
    return all_saved


def prepare_wr_frames(wr_dir: Path, output_dir: Path, max_videos: int = 30,
                      frames_per_video: int = 10):
    """从WR视频抽帧 (错误流程, 用于测试)"""
    print("\n" + "="*60)
    print(f"从WR视频抽帧 (错误流程, 最多{max_videos}个)")
    print("="*60)
    frames_dir = output_dir / "images" / "val"
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(wr_dir.glob("*.avi"))[:max_videos]
    all_saved = []
    for video in tqdm(videos, desc="Extracting WR frames"):
        prefix = f"wr_{video.stem[:20]}_"
        saved = extract_frames(video, frames_dir, max_frames=frames_per_video,
                               prefix=prefix)
        all_saved.extend(saved)
    print(f"  Total WR frames: {len(all_saved)}")
    return all_saved


def create_dataset_yaml(output_dir: Path, nc: int, class_names: list):
    """生成YOLO dataset.yaml"""
    yaml_path = output_dir / "dataset.yaml"
    data = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"\nDataset config saved to: {yaml_path}")


def auto_annotate_with_grounding_dino(images_dir: Path, labels_dir: Path,
                                      class_names: list, box_threshold: float = 0.25,
                                      text_threshold: float = 0.25):
    """
    使用Grounding DINO进行自动标注。
    需要安装: pip install transformers accelerate torchvision supervision
    """
    print("\n" + "="*60)
    print("Grounding DINO 自动标注")
    print("="*60)

    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
    except ImportError:
        print("  ERROR: transformers not installed. Install with:")
        print("  pip install transformers accelerate torchvision supervision")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # 加载Grounding DINO
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    labels_dir.mkdir(parents=True, exist_ok=True)
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    # 构建文本提示
    text_prompt = ". ".join([
        "a closed paper box",
        "an opened paper box",
        "a black earphone case",
        "a white charger plug",
        "a small green bag",
    ])

    annotated_count = 0
    for img_path in tqdm(image_files, desc="Auto-annotating"):
        try:
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
            w, h = image.size

            inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(h, w)],
            )

            # 写入YOLO格式标注
            result = results[0]
            boxes = result["boxes"].cpu().numpy()
            labels_list = result["labels"]

            label_path = labels_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                for box, label_str in zip(boxes, labels_list):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    # 映射到class ID
                    cls_id = -1
                    label_lower = label_str.lower()
                    if "closed" in label_lower and "box" in label_lower:
                        cls_id = 0
                    elif "opened" in label_lower and "box" in label_lower:
                        cls_id = 1
                    elif "earphone" in label_lower:
                        cls_id = 2
                    elif "charger" in label_lower:
                        cls_id = 3
                    elif "green" in label_lower or "bag" in label_lower:
                        cls_id = 4

                    if cls_id >= 0:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            annotated_count += 1
        except Exception as e:
            print(f"  Error annotating {img_path.name}: {e}")

    print(f"  Annotated: {annotated_count}/{len(image_files)} frames")
    return annotated_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare YOLO training data")
    parser.add_argument("--data-dir", default="/home/zhaowei/shabi/data",
                       help="Root data directory")
    parser.add_argument("--output-dir", default="/home/zhaowei/shabi/data/yolo_dataset_v3",
                       help="Output directory for YOLO dataset")
    parser.add_argument("--ok-videos", type=int, default=50,
                       help="Max OK videos to extract")
    parser.add_argument("--wr-videos", type=int, default=30,
                       help="Max WR videos to extract")
    parser.add_argument("--yuanzi-frames", type=int, default=50,
                       help="Frames per yuanzi video")
    parser.add_argument("--ok-frames", type=int, default=20,
                       help="Frames per OK video")
    parser.add_argument("--auto-annotate", action="store_true",
                       help="Run Grounding DINO auto-annotation")
    parser.add_argument("--box-threshold", type=float, default=0.25,
                       help="Grounding DINO box threshold")
    parser.set_defaults(auto_annotate=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    class_names = ["box_closed", "box_open", "earphone", "charger", "green_bag"]

    # 1. 抽帧
    prepare_yuanzi_frames(data_dir / "yuanzi", output_dir, args.yuanzi_frames)
    prepare_ok_frames(data_dir / "ok", output_dir, args.ok_videos, args.ok_frames)
    prepare_wr_frames(data_dir / "wr", output_dir, args.wr_videos)

    # 2. 自动标注
    if args.auto_annotate:
        for split in ["train", "val"]:
            images_dir = output_dir / "images" / split
            labels_dir = output_dir / "labels" / split
            if images_dir.exists():
                auto_annotate_with_grounding_dino(
                    images_dir, labels_dir, class_names,
                    box_threshold=args.box_threshold
                )

    # 3. 生成配置文件
    create_dataset_yaml(output_dir, len(class_names), class_names)

    print("\n" + "="*60)
    print("数据准备完成!")
    print(f"输出目录: {output_dir}")
    print(f"下一步: 检查标注质量, 修正错误标注")
    print(f"然后运行: python training/train_yolo.py --data {output_dir}/dataset.yaml")
    print("="*60)


if __name__ == "__main__":
    main()
