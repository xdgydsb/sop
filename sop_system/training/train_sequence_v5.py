"""
时序模型训练 — 使用features_v5 (.npz格式)
逐帧序列标注: BiGRU 3层, 256 hidden
"""
import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.temporal_lstm import SOPActionGRU, SOPActionTCN

FEATURES_DIR = Path("/home/zhaowei/shabi/data/features_v5")
OUTPUT_DIR = Path("/home/zhaowei/shabi/data/sop_system/training/runs")

SEQ_LEN = 128
STRIDE = 64
BATCH_SIZE = 24
LR = 5e-4
EPOCHS = 200
PATIENCE = 50
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.02
CLASS_WEIGHTS = torch.FloatTensor([0.4, 0.6, 0.7, 1.8, 3.0, 0.8, 0.1])

STEP_NAMES = {0: "Idle", 1: "S1-Open", 2: "S2-Earphone", 3: "S3-Charger",
              4: "S4-GreenBag", 5: "S5-Close", 6: "Done"}


class SequenceDatasetV5(Dataset):
    """从.npz文件加载 (X, y)"""

    def __init__(self, npz_files, seq_len=SEQ_LEN, stride=STRIDE):
        self.samples = []

        for npz_path in npz_files:
            data = np.load(npz_path)
            X = data['X'].astype(np.float32)
            y = data['y'].astype(np.int64)

            if len(X) < seq_len // 2:
                continue

            n_segments = max(1, (len(X) - seq_len) // stride + 1)
            for i in range(n_segments):
                start = i * stride
                end = min(start + seq_len, len(X))
                if end - start < seq_len // 4:
                    continue
                self.samples.append((X[start:end], y[start:end], npz_path.stem))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, _ = self.samples[idx]
        if len(x) < SEQ_LEN:
            pad = SEQ_LEN - len(x)
            x = np.pad(x, ((0, pad), (0, 0)), mode='edge')
            y = np.pad(y, (0, pad), mode='edge')
        else:
            x = x[:SEQ_LEN]
            y = y[:SEQ_LEN]
        return torch.FloatTensor(x.copy()), torch.LongTensor(y.copy())


def compute_class_distribution(npz_files):
    counts = np.zeros(7, dtype=int)
    for f in npz_files:
        data = np.load(f)
        y = data['y'].astype(np.int64)
        for i in range(7):
            counts[i] += (y == i).sum()
    total = counts.sum()
    print("  Class distribution (frames):")
    for i in range(7):
        pct = counts[i] / max(total, 1) * 100
        bar = "|" * int(pct) + "." * (20 - int(pct))
        print(f"    Class {i} ({STEP_NAMES.get(i, '?')}): {counts[i]:,d} ({pct:.1f}%) {bar}")
    return counts


def main():
    FEATURES_DIR_V5 = FEATURES_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Per-Frame Sequence Labeling — BiGRU on features_v5")
    print("=" * 60)

    # 加载视频列表
    print("\n[1/4] Loading videos...")
    all_npz = sorted(FEATURES_DIR_V5.glob("*.npz"))
    print(f"  Total .npz files: {len(all_npz)}")

    # 分类: yuanzi / OK / WR
    yuanzi_npz = [f for f in all_npz if f.stem.isdigit()]
    ok_npz = [f for f in all_npz if f.stem.startswith("ok_")]
    wr_npz = [f for f in all_npz if f.stem.startswith("wr_")]
    print(f"  yuanzi: {len(yuanzi_npz)}, OK: {len(ok_npz)}, WR: {len(wr_npz)}")

    # 检查特征维度
    sample = np.load(all_npz[0])
    input_size = sample['X'].shape[1]
    print(f"  Input dim: {input_size}")

    # 分层划分: yuanzi全部进train, OK+WR 85/15
    yuanzi_files = sorted(yuanzi_npz)
    okwr_files = sorted(ok_npz + wr_npz)

    np.random.seed(42)
    indices = np.random.permutation(len(okwr_files))
    n_val = max(int(len(indices) * 0.15), 25)
    val_idx = set(indices[:n_val])
    train_files = yuanzi_files + [okwr_files[i] for i in range(len(okwr_files)) if i not in val_idx]
    val_files = [okwr_files[i] for i in range(len(okwr_files)) if i in val_idx]

    # 确保val有所有类别
    val_labels_set = set()
    for f in val_files[:50]:
        y = np.load(f)['y']
        val_labels_set.update(np.unique(y).tolist())
    print(f"  Val classes: {sorted(val_labels_set)}")
    missing = set(range(6)) - val_labels_set
    if missing:
        print(f"  Fixing: moving videos with classes {missing} to val")
        for f in okwr_files[:]:
            if f not in val_files:
                y = np.load(f)['y']
                if missing & set(np.unique(y).tolist()):
                    val_files.append(f)
                    train_files.remove(f)
                    val_labels_set.update(np.unique(y).tolist())
                    missing = set(range(6)) - val_labels_set
                    if not missing:
                        break
    print(f"  Train: {len(train_files)}, Val: {len(val_files)}")

    # 类别分布
    print("\n[2/4] Class distribution...")
    compute_class_distribution(train_files)

    # 数据集
    train_ds = SequenceDatasetV5(train_files)
    val_ds = SequenceDatasetV5(val_files)
    print(f"  Train segments: {len(train_ds)}, Val segments: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 模型
    print("\n[3/4] Building BiGRU model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SOPActionGRU(input_size=input_size, hidden_size=256, num_layers=3,
                         num_classes=7, dropout=0.45).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")

    class_weights = CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING, ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

    # 训练
    print(f"\n[4/4] Training ({EPOCHS} epochs)...")
    best_val_acc = 0
    best_path = OUTPUT_DIR / "best_sequence_v5.pt"
    patience_count = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)  # (B, T, 7)
            loss = criterion(logits.reshape(-1, 7), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()
            pred = logits.argmax(-1)
            mask = y != -1
            train_correct += (pred[mask] == y[mask]).sum().item()
            train_total += mask.sum().item()

        train_acc = train_correct / max(train_total, 1)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        # Per-class accuracy
        class_correct = np.zeros(7, dtype=int)
        class_total = np.zeros(7, dtype=int)

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(-1)
                mask = y != -1
                val_correct += (pred[mask] == y[mask]).sum().item()
                val_total += mask.sum().item()

                for c in range(7):
                    c_mask = (y == c)
                    class_correct[c] += (pred[c_mask] == y[c_mask]).sum().item()
                    class_total[c] += c_mask.sum().item()

        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss/len(train_loader):.4f}, "
                  f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, lr={scheduler.get_last_lr()[0]:.1e}")
            for c in range(7):
                if class_total[c] > 0:
                    ca = class_correct[c] / class_total[c]
                    print(f"    {STEP_NAMES.get(c, '?'):12s}: {ca:.3f} ({class_total[c]} frames)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden_size': 256,
                'num_layers': 3,
                'num_classes': 7,
                'val_acc': val_acc,
            }, best_path)
            patience_count = 0
            print(f"  >>> Best: val_acc={val_acc:.4f}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\nDone! Best val_acc: {best_val_acc:.4f}")
    print(f"Model: {best_path}")


if __name__ == "__main__":
    main()
