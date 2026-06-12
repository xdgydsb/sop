"""
GRU 时序动作识别 — 支持多维度特征 (21/46/130维)

特征维度:
- 21维(指触): 仅指尖-物体距离, YOLO漏检时特征退化严重
- 46维(精简): +物体bbox/IoU/置信度, 但仍缺手部关键点
- 130维(完整): +84维手部关键点坐标, 即使YOLO漏检也有连续手部运动信号 ★推荐

核心思路:
- 130维特征: 手部关键点(84维)提供连续运动信号, 不同动作手部轨迹截然不同
- Clean segment: 只从标注段中间采样, 避开边界歧义
- BiGRU: 捕获双向时序依赖
"""
import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.temporal_lstm import SOPActionGRU, SOPActionLSTM
from config import *


# ---- 模型配置: 按特征维度选择 ----
MODEL_CONFIGS = {
    21:  {"hidden_size": 48,  "num_layers": 1, "dropout": 0.6, "batch_size": 64, "arch": "gru"},
    46:  {"hidden_size": 64,  "num_layers": 1, "dropout": 0.5, "batch_size": 64, "arch": "gru"},
    130: {"hidden_size": 128, "num_layers": 2, "dropout": 0.4, "batch_size": 48, "arch": "gru"},
}

SEQ_LEN = 48
LR = 0.001
EPOCHS = 200
PATIENCE = 40
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.10
GRAD_CLIP = 0.5


def extract_action_segments(y):
    """从标签序列中提取每个动作的连续段 (start, end) → {step: [(start, end), ...]}"""
    segments = {}
    n = len(y)
    i = 0
    while i < n:
        step = int(y[i])
        start = i
        while i < n and int(y[i]) == step:
            i += 1
        end = i
        if end - start >= 10:
            segments.setdefault(step, []).append((start, end))
    return segments


class ActionWindowDataset(Dataset):
    """从标注段中间采样特征窗口, 避开边界, >80%多数标签"""

    def __init__(self, features_dir, video_list, seq_len=SEQ_LEN,
                 margin=6, augment=False):
        self.windows = []
        self.seq_len = seq_len

        for X_name, y_name in video_list:
            X_path = features_dir / X_name
            y_path = features_dir / y_name
            if not X_path.exists() or not y_path.exists():
                continue

            X = np.load(X_path).astype(np.float32)
            y = np.load(y_path).astype(np.int64)

            if len(X) < seq_len:
                continue

            segments = extract_action_segments(y)
            for step, segs in segments.items():
                for seg_start, seg_end in segs:
                    seg_len = seg_end - seg_start
                    inner_start = seg_start + margin
                    inner_end = seg_end - margin

                    if inner_end - inner_start < seq_len:
                        inner_start = seg_start
                        inner_end = seg_end
                    if inner_end - inner_start < seq_len:
                        continue

                    stride = max(seq_len // 6, 2)
                    for w_start in range(inner_start, inner_end - seq_len, stride):
                        w_end = w_start + seq_len
                        window_y = y[w_start:w_end]
                        counts = np.bincount(window_y)
                        majority_label = int(np.argmax(counts))
                        majority_ratio = counts[majority_label] / seq_len

                        if majority_ratio >= 0.8 and majority_label == step:
                            self.windows.append((X[w_start:w_end], majority_label))

            # 数据增强: 随机偏移窗口
            if augment:
                for step, segs in segments.items():
                    for seg_start, seg_end in segs:
                        inner_start = seg_start + margin
                        inner_end = seg_end - margin
                        if inner_end - inner_start < seq_len:
                            inner_start = seg_start
                            inner_end = seg_end
                        if inner_end - inner_start < seq_len:
                            continue

                        stride = max(seq_len // 6, 2)
                        for _ in range(2):
                            for w_start in range(inner_start, inner_end - seq_len, stride):
                                offset = np.random.randint(-margin, margin + 1)
                                w2 = max(seg_start, min(seg_end - seq_len, w_start + offset))
                                w2_end = w2 + seq_len
                                window_y = y[w2:w2_end]
                                counts = np.bincount(window_y)
                                ml = int(np.argmax(counts))
                                if counts[ml] / seq_len >= 0.8 and ml == step:
                                    self.windows.append((X[w2:w2_end], ml))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        return torch.FloatTensor(x.copy()), torch.LongTensor([y])[0]


def validate_features(features_dir, video_list):
    """验证特征质量: 检查每类特征是否有区分度"""
    print("\n[Feature validation]")
    all_class_feats = {}

    for X_name, y_name in video_list[:20]:  # Check first 20 videos
        X_path = features_dir / X_name
        y_path = features_dir / y_name
        if not X_path.exists() or not y_path.exists():
            continue
        X = np.load(X_path).astype(np.float32)
        y = np.load(y_path).astype(np.int64)

        for cls_id in sorted(set(y.astype(int))):
            mask = y == cls_id
            cls_X = X[mask]
            if len(cls_X) < 5:
                continue
            if cls_id not in all_class_feats:
                all_class_feats[cls_id] = []
            all_class_feats[cls_id].append(cls_X.mean(axis=0))

    # Check inter-class feature distance
    if len(all_class_feats) >= 2:
        centroids = {}
        for cls_id, means in all_class_feats.items():
            centroids[cls_id] = np.mean(means, axis=0)

        cls_ids = sorted(centroids.keys())
        print(f"  Classes found: {cls_ids}")
        for i in range(len(cls_ids)):
            for j in range(i + 1, len(cls_ids)):
                a, b = cls_ids[i], cls_ids[j]
                dist = np.linalg.norm(centroids[a] - centroids[b])
                status = "OK" if dist > 2.0 else "WEAK" if dist > 0.5 else "BAD"
                print(f"  Class {a} vs {b}: L2={dist:.3f} [{status}]")

    # Check per-class feature variance (higher = more distinctive signal)
    for cls_id in sorted(all_class_feats.keys()):
        means = all_class_feats[cls_id]
        if len(means) > 1:
            var = np.mean([np.var(m) for m in means])
            print(f"  Class {cls_id}: n={len(means)}, mean_feat_var={var:.4f}")


def train(features_dir, output_dir, epochs=EPOCHS, lr=LR, device="cuda"):
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data, auto-detect input dim ----
    print("=" * 60)
    print("GRU Action Recognition — Multi-Dim Feature Support")
    print("=" * 60)

    print("\n[1/4] Preparing data...")
    video_list = []

    # yuanzi pure actions (gold)
    for i in range(1, 6):
        if (features_dir / f"{i}_X.npy").exists() and (features_dir / f"{i}_y.npy").exists():
            video_list.append((f"{i}_X.npy", f"{i}_y.npy"))
            print(f"  yuanzi/{i}: pure action (label={i})")

    # OK videos
    ok_count = 0
    for x_path in sorted(features_dir.glob("ok_*_X.npy")):
        y_name = x_path.stem.replace("_X", "_y_seg") + ".npy"
        if (features_dir / y_name).exists():
            video_list.append((x_path.name, y_name))
            ok_count += 1
    print(f"  OK videos: {ok_count}")

    # WR videos
    wr_count = 0
    for x_path in sorted(features_dir.glob("wr_*_X.npy")):
        y_name = x_path.stem.replace("_X", "_y_seg") + ".npy"
        if (features_dir / y_name).exists():
            video_list.append((x_path.name, y_name))
            wr_count += 1
    print(f"  WR videos: {wr_count}")
    print(f"  Total: {len(video_list)} videos")

    # Auto-detect input dimension
    sample_X = np.load(features_dir / video_list[0][0])
    input_size = sample_X.shape[1]
    print(f"  Detected input dim: {input_size}")

    cfg = MODEL_CONFIGS.get(input_size, MODEL_CONFIGS[130])
    hidden_size = cfg["hidden_size"]
    num_layers = cfg["num_layers"]
    dropout = cfg["dropout"]
    batch_size = cfg["batch_size"]
    arch = cfg["arch"]
    print(f"  Model: {arch.upper()} hidden={hidden_size}, layers={num_layers}, dropout={dropout}")

    # Split: yuanzi always in train, 85/15 for OK+WR
    n_yuanzi = sum(1 for v in video_list if v[0].startswith(("1_", "2_", "3_", "4_", "5_")))
    n_videos = len(video_list)
    okwr_indices = list(range(n_yuanzi, n_videos))
    np.random.seed(42)
    np.random.shuffle(okwr_indices)
    n_train = int(len(okwr_indices) * 0.85)
    train_idx = okwr_indices[:n_train]
    val_idx = okwr_indices[n_train:]

    train_vids = [video_list[i] for i in range(n_yuanzi)] + [video_list[i] for i in train_idx]
    val_vids = [video_list[i] for i in val_idx]

    # ---- 2. Validate feature quality ----
    print("\n[2/4] Validating feature quality...")
    validate_features(features_dir, video_list)

    # ---- 3. Create datasets ----
    print(f"\n[3/4] Creating clean-segment datasets (seq_len={SEQ_LEN})...")
    train_ds = ActionWindowDataset(features_dir, train_vids, seq_len=SEQ_LEN,
                                   margin=6, augment=True)
    val_ds = ActionWindowDataset(features_dir, val_vids, seq_len=SEQ_LEN,
                                  margin=6, augment=False)

    print(f"  Train: {len(train_ds)} windows from {len(train_vids)} videos")
    print(f"  Val:   {len(val_ds)} windows from {len(val_vids)} videos")

    if len(train_ds) < 100:
        print("  ERROR: Not enough training data!")
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ---- 4. Train ----
    print(f"\n[4/4] Training (epochs={epochs}, lr={lr}, batch={batch_size})...")

    if arch == "lstm":
        model = SOPActionLSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, num_classes=LSTM_NUM_CLASSES,
            dropout=dropout,
        ).to(device)
    else:
        model = SOPActionGRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, num_classes=LSTM_NUM_CLASSES,
            dropout=dropout,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Class weights: 背景和完成稍低
    class_weights = torch.FloatTensor([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    best_path = output_dir / "best_gru.pt"
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(logits, 1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / max(val_total, 1)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_classes": LSTM_NUM_CLASSES,
                "dropout": dropout,
                "val_acc": val_acc,
            }, best_path)
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved: {best_path}")

    # Per-class evaluation
    print("\n--- Per-class evaluation ---")
    checkpoint = torch.load(best_path, map_location="cpu")
    if arch == "lstm":
        eval_model = SOPActionLSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, num_classes=LSTM_NUM_CLASSES,
            dropout=dropout,
        ).to(device)
    else:
        eval_model = SOPActionGRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, num_classes=LSTM_NUM_CLASSES,
            dropout=dropout,
        ).to(device)
    eval_model.load_state_dict(checkpoint["model_state_dict"])
    eval_model.eval()

    class_correct = np.zeros(LSTM_NUM_CLASSES, dtype=int)
    class_total = np.zeros(LSTM_NUM_CLASSES, dtype=int)
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = eval_model(X)
            _, preds = torch.max(logits, 1)
            for i in range(LSTM_NUM_CLASSES):
                mask = (y == i)
                class_correct[i] += (preds[mask] == i).sum().item()
                class_total[i] += mask.sum().item()

    for i in range(LSTM_NUM_CLASSES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"  Class {i} ({STEP_NAMES.get(i, '?')}): {acc:.3f} {bar} ({class_total[i]} windows)")
        else:
            print(f"  Class {i} ({STEP_NAMES.get(i, '?')}): NO DATA")

    return str(best_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="/home/zhaowei/shabi/data/features_v7")
    parser.add_argument("--output-dir", default="/home/zhaowei/shabi/sop_system/runs")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(args.features_dir, args.output_dir, args.epochs,
          lr=args.lr, device=args.device)


if __name__ == "__main__":
    main()
