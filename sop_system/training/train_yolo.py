"""
YOLO训练脚本 — 训练7类物体检测模型
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
import torch


def train(data_yaml: str, model_size: str = "m", epochs: int = 100,
          batch: int = 16, imgsz: int = 640, device: str = "0",
          project: str = "runs/train", name: str = "sop_yolo"):
    """训练YOLOv8模型"""
    print("="*60)
    print(f"训练YOLOv8{model_size}: {epochs} epochs, batch={batch}")
    print("="*60)

    model = YOLO(f"yolov8{model_size}.pt")  # 从预训练权重开始

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        patience=20,             # 20 epochs无提升则早停
        save=True,
        save_period=10,          # 每10个epoch保存一次
        val=True,
        plots=True,
        lr0=0.001,              # 初始学习率
        lrf=0.01,               # 最终学习率因子
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,            # 余弦退火
        close_mosaic=15,        # 最后15个epoch关闭mosaic增强
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    # 验证
    print("\n验证最佳模型...")
    best_pt = Path(project) / name / "weights" / "best.pt"
    if best_pt.exists():
        model = YOLO(str(best_pt))
        metrics = model.val()
        print(f"\n验证结果: mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")

    print(f"\n模型保存至: {best_pt}")
    return str(best_pt)


def test_model(model_path: str, test_image: str, device: str = "0"):
    """在单个测试图片上运行模型, 可视化结果"""
    import cv2
    model = YOLO(model_path)
    results = model(test_image, device=device)
    for result in results:
        annotated = result.plot()
        out_path = Path(test_image).parent / f"{Path(test_image).stem}_detected.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"检测结果保存至: {out_path}")

        # 打印检测到的物体
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                print(f"  {cls_name}: {conf:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dataset.yaml path")
    parser.add_argument("--model", default="m", choices=["n", "s", "m", "l"],
                       help="YOLO model size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="sop_yolo")
    parser.add_argument("--test", help="Test image path")
    args = parser.parse_args()

    model_path = train(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
    )

    if args.test:
        test_model(model_path, args.test, args.device)
