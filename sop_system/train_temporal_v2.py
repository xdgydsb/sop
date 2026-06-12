"""
Step 5: Train TCN + BiGRU multi-task temporal model
Input: data/datasets/temporal_v2_90_T48_S8/
Output: models/temporal/v2_90_tcn_bigru/
"""
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from collections import Counter
import time
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / "data" / "datasets" / "temporal_v2_90_T48_S8"
MODEL_DIR = ROOT / "models" / "temporal" / "v2_90_tcn_bigru"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
REPORTS_DIR = ROOT / "reports" / "temporal_v2_90"
LOGS_DIR = ROOT / "logs" / "temporal_v2_90"


class CausalConv1d(nn.Module):
    """Causal 1D convolution with proper padding"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """TCN block: Conv → LayerNorm → ReLU → Dropout"""
    def __init__(self, channels, kernel_size, dilation, dropout=0.3):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        # 1x1 conv for residual if needed
        self.residual = nn.Identity()

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        out = self.conv1(x)  # [B, C, T]
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out + residual)
        out = self.dropout(out)
        return out


class TCNBiGRU(nn.Module):
    """TCN + BiGRU multi-task model for 48-frame windows"""

    def __init__(self, input_dim=90, hidden_dim=128, num_classes_step=6,
                 num_classes_verb=4, num_classes_target=5,
                 dropout=0.3, num_gru_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # TCN blocks
        self.tcn1 = TCNBlock(hidden_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.tcn2 = TCNBlock(hidden_dim, kernel_size=3, dilation=2, dropout=dropout)
        self.tcn3 = TCNBlock(hidden_dim, kernel_size=3, dilation=4, dropout=dropout)

        # BiGRU
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_gru_layers,
                          bidirectional=True, batch_first=True, dropout=dropout if num_gru_layers > 1 else 0)

        # Heads
        self.step_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes_step),
        )
        self.verb_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes_verb),
        )
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes_target),
        )

    def forward(self, x):
        # x: [B, T, D]
        b, t, d = x.shape

        # Project and transpose for TCN
        h = self.input_proj(x)  # [B, T, H]
        h_tcn = h.transpose(1, 2)  # [B, H, T]

        # TCN blocks
        h_tcn = self.tcn1(h_tcn)
        h_tcn = self.tcn2(h_tcn)
        h_tcn = self.tcn3(h_tcn)

        # Back to [B, T, H] for GRU
        h_seq = h_tcn.transpose(1, 2)  # [B, T, H]

        # BiGRU
        gru_out, _ = self.gru(h_seq)  # [B, T, 2*H]

        # Take center frame (T//2) as window-level prediction
        center = gru_out[:, t // 2, :]  # [B, 2*H]

        step_logits = self.step_head(center)
        verb_logits = self.verb_head(center)
        target_logits = self.target_head(center)

        return step_logits, verb_logits, target_logits


class WindowDataset(Dataset):
    """Loads windows from index file, reads features+lazily from npz"""

    def __init__(self, index_path, feature_cache=None):
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.windows = data["windows"]
        self._cache = {} if feature_cache is None else feature_cache
        self._root = ROOT

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        feat_path = self._root / w["feature_path"]

        if feat_path not in self._cache:
            self._cache[feat_path] = np.load(feat_path, allow_pickle=True)["features"]

        features = self._cache[feat_path]
        x = features[w["start_idx"]:w["end_idx"]].astype(np.float32)

        step_y = w["center_step_label"]
        verb_y = w["center_verb_label"]
        target_y = w["center_target_label"]

        return (
            torch.FloatTensor(x),
            torch.LongTensor([step_y])[0],
            torch.LongTensor([verb_y])[0],
            torch.LongTensor([target_y])[0],
        )


def compute_class_weights(windows, num_classes):
    """Compute inverse-frequency class weights"""
    counter = Counter(w["center_step_label"] for w in windows)
    total = sum(counter.values())
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        weights[c] = total / max(counter.get(c, 1), 1)
    weights = weights / weights.mean()  # normalize
    return weights


def train_epoch(model, dataloader, optimizer, class_weights, device):
    model.train()
    total_loss = 0
    step_correct = 0
    verb_correct = 0
    target_correct = 0
    total = 0

    step_w = class_weights.get("step", None)
    if step_w is not None:
        step_w = step_w.to(device)

    for x, step_y, verb_y, target_y in dataloader:
        x = x.to(device)
        step_y = step_y.to(device)
        verb_y = verb_y.to(device)
        target_y = target_y.to(device)

        optimizer.zero_grad()
        step_logits, verb_logits, target_logits = model(x)

        L_step = F.cross_entropy(step_logits, step_y, weight=step_w)
        L_verb = F.cross_entropy(verb_logits, verb_y)
        L_target = F.cross_entropy(target_logits, target_y)
        loss = L_step + 0.3 * L_verb + 0.5 * L_target

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        step_correct += (step_logits.argmax(dim=1) == step_y).sum().item()
        verb_correct += (verb_logits.argmax(dim=1) == verb_y).sum().item()
        target_correct += (target_logits.argmax(dim=1) == target_y).sum().item()
        total += x.size(0)

    return (total_loss / total,
            step_correct / total,
            verb_correct / total,
            target_correct / total)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    step_correct = 0
    verb_correct = 0
    target_correct = 0
    total = 0
    all_step_preds = []
    all_step_targets = []

    for x, step_y, verb_y, target_y in dataloader:
        x = x.to(device)
        step_y = step_y.to(device)
        verb_y = verb_y.to(device)
        target_y = target_y.to(device)

        step_logits, verb_logits, target_logits = model(x)

        L_step = F.cross_entropy(step_logits, step_y)
        L_verb = F.cross_entropy(verb_logits, verb_y)
        L_target = F.cross_entropy(target_logits, target_y)
        loss = L_step + 0.3 * L_verb + 0.5 * L_target

        total_loss += loss.item() * x.size(0)
        step_correct += (step_logits.argmax(dim=1) == step_y).sum().item()
        verb_correct += (verb_logits.argmax(dim=1) == verb_y).sum().item()
        target_correct += (target_logits.argmax(dim=1) == target_y).sum().item()
        total += x.size(0)

        all_step_preds.append(step_logits.argmax(dim=1).cpu())
        all_step_targets.append(step_y.cpu())

    all_preds = torch.cat(all_step_preds).numpy()
    all_targets = torch.cat(all_step_targets).numpy()

    return {
        "loss": total_loss / total,
        "step_acc": step_correct / total,
        "verb_acc": verb_correct / total,
        "target_acc": target_correct / total,
        "preds": all_preds,
        "targets": all_targets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use-class-weights", action="store_true", default=True)
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets
    train_ds = WindowDataset(DATASET_DIR / "train_index.json")
    val_ds = WindowDataset(DATASET_DIR / "val_index.json")
    test_ds = WindowDataset(DATASET_DIR / "test_index.json")

    print(f"Train: {len(train_ds)} windows")
    print(f"Val: {len(val_ds)} windows")
    print(f"Test: {len(test_ds)} windows")

    # Class weights
    step_weights = None
    if args.use_class_weights:
        step_weights = compute_class_weights(train_ds.windows, 6)
        print(f"Step weights: {step_weights.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Build model
    model = TCNBiGRU(
        input_dim=90,
        hidden_dim=args.hidden_dim,
        num_classes_step=6,
        num_classes_verb=4,
        num_classes_target=5,
        dropout=args.dropout,
        num_gru_layers=args.gru_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights_dict = {"step": step_weights}

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    train_log = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_step, train_verb, train_target = train_epoch(
            model, train_loader, optimizer, class_weights_dict, device)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_step_acc": round(train_step, 4),
            "train_verb_acc": round(train_verb, 4),
            "train_target_acc": round(train_target, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_step_acc": round(val_metrics["step_acc"], 4),
            "val_verb_acc": round(val_metrics["verb_acc"], 4),
            "val_target_acc": round(val_metrics["target_acc"], 4),
            "lr": scheduler.get_last_lr()[0],
            "elapsed": round(elapsed, 1),
        }
        train_log.append(log_entry)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train: loss={train_loss:.3f} step={train_step:.3f} "
              f"verb={train_verb:.3f} tgt={train_target:.3f} | "
              f"val: loss={val_metrics['loss']:.3f} step={val_metrics['step_acc']:.3f} "
              f"verb={val_metrics['verb_acc']:.3f} tgt={val_metrics['target_acc']:.3f} | "
              f"{elapsed:.1f}s")

        # Save best
        if val_metrics["step_acc"] > best_val_acc:
            best_val_acc = val_metrics["step_acc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_step_acc": val_metrics["step_acc"],
                "val_verb_acc": val_metrics["verb_acc"],
                "val_target_acc": val_metrics["target_acc"],
                "input_dim": 90,
                "hidden_dim": args.hidden_dim,
                "num_classes_step": 6,
                "num_classes_verb": 4,
                "num_classes_target": 5,
                "dropout": args.dropout,
                "num_gru_layers": args.gru_layers,
            }, CHECKPOINT_DIR / "best.pt")
        else:
            patience_counter += 1

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_dim": 90,
            "hidden_dim": args.hidden_dim,
            "num_classes_step": 6,
            "num_classes_verb": 4,
            "num_classes_target": 5,
            "dropout": args.dropout,
            "num_gru_layers": args.gru_layers,
        }, CHECKPOINT_DIR / "last.pt")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model
    best_ckpt = torch.load(CHECKPOINT_DIR / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # Test evaluation
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest: step={test_metrics['step_acc']:.4f} "
          f"verb={test_metrics['verb_acc']:.4f} "
          f"target={test_metrics['target_acc']:.4f}")

    # Save train config
    config = {
        "model": "TCN_BiGRU",
        "input_dim": 90,
        "hidden_dim": args.hidden_dim,
        "num_classes_step": 6,
        "num_classes_verb": 4,
        "num_classes_target": 5,
        "dropout": args.dropout,
        "gru_layers": args.gru_layers,
        "tcn_blocks": [
            {"kernel": 3, "dilation": 1},
            {"kernel": 3, "dilation": 2},
            {"kernel": 3, "dilation": 4},
        ],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "best_val_step_acc": float(best_val_acc),
        "test_step_acc": float(test_metrics["step_acc"]),
        "test_verb_acc": float(test_metrics["verb_acc"]),
        "test_target_acc": float(test_metrics["target_acc"]),
        "total_params": total_params,
        "class_weights": step_weights.tolist() if step_weights is not None else None,
    }
    with open(MODEL_DIR / "train_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Save label mapping
    label_mapping = {
        "step": {"0": "idle", "1": "open_box", "2": "put_black_case",
                 "3": "put_white_plug", "4": "put_green_bag", "5": "close_box"},
        "verb": {"0": "idle", "1": "open", "2": "put", "3": "close"},
        "target": {"0": "none", "1": "box", "2": "black_case",
                   "3": "white_plug", "4": "green_bag"},
    }
    with open(MODEL_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    # Save feature meta (copy from features dir)
    feat_meta_path = ROOT / "data" / "features" / "v2_90" / "feature_meta_v2_90.json"
    if feat_meta_path.exists():
        with open(feat_meta_path) as f:
            feat_meta = json.load(f)
        with open(MODEL_DIR / "feature_meta.json", "w", encoding="utf-8") as f:
            json.dump(feat_meta, f, indent=2, ensure_ascii=False)

    # Save logs
    with open(LOGS_DIR / "train.log", "w", encoding="utf-8") as f:
        for entry in train_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Val metrics
    with open(REPORTS_DIR / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_step_acc": float(best_val_acc),
            "final_epoch": epoch,
        }, f, indent=2)

    print(f"\nBest epoch: {best_epoch}, val_step_acc: {best_val_acc:.4f}")
    print(f"Model saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
