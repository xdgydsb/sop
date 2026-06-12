"""
将手动标注的50张图片合并到YOLO数据集
- 手动标注图片 → 全部放入train (高质量ground truth)
- 与v5自动标注数据集合并
"""
import shutil
from pathlib import Path
import numpy as np


def merge_manual_labels(manual_dir, existing_dataset, output_dir, val_ratio=0.1):
    manual_dir = Path(manual_dir)
    existing_dataset = Path(existing_dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train/images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train/labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "val/images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val/labels").mkdir(parents=True, exist_ok=True)

    # 1. 复制已有数据集 (v5)
    print("复制已有数据集...")
    for sub in ["train", "val"]:
        for modality in ["images", "labels"]:
            src = existing_dataset / sub / modality
            if src.exists():
                for f in src.iterdir():
                    dst = output_dir / sub / modality / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
    n_existing_train = len(list((output_dir / "train/images").iterdir()))
    n_existing_val = len(list((output_dir / "val/images").iterdir()))
    print(f"  已有: train={n_existing_train}, val={n_existing_val}")

    # 2. 复制手动标注图片+标签
    print("添加手动标注...")
    manual_images = sorted(manual_dir.glob("*.jpg"))
    manual_labels = sorted(manual_dir.glob("*.txt"))

    # 只保留同时有图片和标签的
    labeled = []
    for img in manual_images:
        lbl = manual_dir / (img.stem + ".txt")
        if lbl.exists():
            labeled.append((img, lbl))

    print(f"  手动标注了 {len(labeled)} 张图片")

    # 随机分一部分到val
    rng = np.random.RandomState(42)
    order = list(range(len(labeled)))
    rng.shuffle(order)
    n_val = max(1, int(len(labeled) * val_ratio))
    val_indices = set(order[:n_val])

    manual_train = 0
    manual_val = 0
    for i, (img, lbl) in enumerate(labeled):
        if i in val_indices:
            dst_img = output_dir / "val/images" / img.name
            dst_lbl = output_dir / "val/labels" / lbl.name
            manual_val += 1
        else:
            dst_img = output_dir / "train/images" / img.name
            dst_lbl = output_dir / "train/labels" / lbl.name
            manual_train += 1
        shutil.copy2(img, dst_img)
        shutil.copy2(lbl, dst_lbl)

    print(f"  手动标注: train={manual_train}, val={manual_val}")

    # 3. 统计
    n_train = len(list((output_dir / "train/images").iterdir()))
    n_val = len(list((output_dir / "val/images").iterdir()))

    # 统计各类别标注数
    class_counts = {i: 0 for i in range(5)}
    class_names = ["box_closed", "box_open", "earphone", "charger", "green_bag"]
    for lbl_dir in [(output_dir / "train/labels"), (output_dir / "val/labels")]:
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id in class_counts:
                            class_counts[cls_id] += 1

    print(f"\n合并完成: train={n_train}, val={n_val}")
    print("各类标注数:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_counts[i]}")

    # 4. 生成 dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("nc: 5\n")
        f.write(f"names: {class_names}\n")
    print(f"\ndataset.yaml → {yaml_path}")
    return output_dir


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-dir", default="/home/zhaowei/shabi/data/manual_label")
    parser.add_argument("--existing-dataset", default="/home/zhaowei/shabi/data/yolo_dataset_v5")
    parser.add_argument("--output-dir", default="/home/zhaowei/shabi/data/yolo_dataset_v6")
    args = parser.parse_args()

    merge_manual_labels(args.manual_dir, args.existing_dataset, args.output_dir)


if __name__ == "__main__":
    main()
