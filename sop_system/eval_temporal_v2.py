"""
Step 6: 评估时序模型 — confusion matrices, per-class metrics, bad cases
输入: models/temporal/v2_90_tcn_bigru/checkpoints/best.pt
      data/datasets/temporal_v2_90_T48_S8/test_index.json
输出: reports/temporal_v2_90/
"""
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from train_temporal_v2 import TCNBiGRU, WindowDataset

DATASET_DIR = ROOT / "data" / "datasets" / "temporal_v2_90_T48_S8"
MODEL_DIR = ROOT / "models" / "temporal" / "v2_90_tcn_bigru"
REPORTS_DIR = ROOT / "reports" / "temporal_v2_90"

STEP_NAMES = ["idle", "open_box", "put_black_case", "put_white_plug",
              "put_green_bag", "close_box"]
VERB_NAMES = ["idle", "open", "put", "close"]
TARGET_NAMES = ["none", "box", "black_case", "white_plug", "green_bag"]


def plot_confusion(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = MODEL_DIR / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, "
          f"val_step_acc={ckpt.get('val_step_acc', '?')}")

    # Build model
    model = TCNBiGRU(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_classes_step=ckpt["num_classes_step"],
        num_classes_verb=ckpt["num_classes_verb"],
        num_classes_target=ckpt["num_classes_target"],
        dropout=ckpt.get("dropout", 0.3),
        num_gru_layers=ckpt.get("num_gru_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load test dataset
    test_ds = WindowDataset(DATASET_DIR / "test_index.json")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    print(f"Test: {len(test_ds)} windows")

    all_step_preds = []
    all_step_targets = []
    all_verb_preds = []
    all_verb_targets = []
    all_target_preds = []
    all_target_targets = []
    bad_cases = []

    with torch.no_grad():
        for x, step_y, verb_y, target_y in test_loader:
            x = x.to(device)
            step_logits, verb_logits, target_logits = model(x)

            step_pred = step_logits.argmax(dim=1).cpu()
            verb_pred = verb_logits.argmax(dim=1).cpu()
            target_pred = target_logits.argmax(dim=1).cpu()

            step_conf = F.softmax(step_logits, dim=1).max(dim=1).values.cpu()

            for i in range(len(step_pred)):
                all_step_preds.append(step_pred[i].item())
                all_step_targets.append(step_y[i].item())
                all_verb_preds.append(verb_pred[i].item())
                all_verb_targets.append(verb_y[i].item())
                all_target_preds.append(target_pred[i].item())
                all_target_targets.append(target_y[i].item())

                if step_pred[i].item() != step_y[i].item():
                    bad_cases.append({
                        "true_step": step_y[i].item(),
                        "pred_step": step_pred[i].item(),
                        "true_step_name": STEP_NAMES[step_y[i].item()],
                        "pred_step_name": STEP_NAMES[step_pred[i].item()],
                        "true_verb": verb_y[i].item(),
                        "pred_verb": verb_pred[i].item(),
                        "true_target": target_y[i].item(),
                        "pred_target": target_pred[i].item(),
                        "confidence": round(step_conf[i].item(), 4),
                    })

    # Step confusion matrix
    step_cm = confusion_matrix(all_step_targets, all_step_preds,
                               labels=list(range(6)))
    plot_confusion(step_cm, STEP_NAMES,
                   f"Step Confusion Matrix (Acc={np.diag(step_cm).sum()/step_cm.sum():.3f})",
                   REPORTS_DIR / "confusion_step.png")
    print(f"\nStep confusion matrix saved to confusion_step.png")

    # Target confusion matrix
    target_cm = confusion_matrix(all_target_targets, all_target_preds,
                                 labels=list(range(5)))
    plot_confusion(target_cm, TARGET_NAMES,
                   f"Target Confusion Matrix (Acc={np.diag(target_cm).sum()/target_cm.sum():.3f})",
                   REPORTS_DIR / "confusion_target.png")
    print("Target confusion matrix saved to confusion_target.png")

    # Step metrics
    print("\n=== Step Classification Report ===")
    report = classification_report(all_step_targets, all_step_preds,
                                   target_names=STEP_NAMES, digits=4, output_dict=True)
    print(classification_report(all_step_targets, all_step_preds,
                                target_names=STEP_NAMES, digits=4))

    # Verb metrics
    print("\n=== Verb Classification Report ===")
    print(classification_report(all_verb_targets, all_verb_preds,
                                target_names=VERB_NAMES, digits=4))

    # Target metrics
    print("\n=== Target Classification Report ===")
    print(classification_report(all_target_targets, all_target_preds,
                                target_names=TARGET_NAMES, digits=4))

    # S3/S4 confusion analysis
    s3_indices = [i for i, t in enumerate(all_step_targets) if t == 3]
    s4_indices = [i for i, t in enumerate(all_step_targets) if t == 4]

    s3_misclassified_as_s4 = sum(1 for i in s3_indices if all_step_preds[i] == 4)
    s4_misclassified_as_s3 = sum(1 for i in s4_indices if all_step_preds[i] == 3)

    s3_s4_analysis = {
        "S3_total": len(s3_indices),
        "S4_total": len(s4_indices),
        "S3_misclassified_as_S4": s3_misclassified_as_s4,
        "S4_misclassified_as_S3": s4_misclassified_as_s3,
        "S3_S4_confusion_rate": round(
            (s3_misclassified_as_s4 + s4_misclassified_as_s3) /
            max(len(s3_indices) + len(s4_indices), 1), 4),
    }
    with open(REPORTS_DIR / "confusion_s3_s4.json", "w", encoding="utf-8") as f:
        json.dump(s3_s4_analysis, f, indent=2, ensure_ascii=False)
    print(f"\nS3/S4 confusion: S3→S4={s3_misclassified_as_s4}/{len(s3_indices)}, "
          f"S4→S3={s4_misclassified_as_s3}/{len(s4_indices)}")

    # Save per-class metrics
    test_metrics = {
        "step_accuracy": float(np.mean(np.array(all_step_preds) == np.array(all_step_targets))),
        "verb_accuracy": float(np.mean(np.array(all_verb_preds) == np.array(all_verb_targets))),
        "target_accuracy": float(np.mean(np.array(all_target_preds) == np.array(all_target_targets))),
        "step_report": report,
        "n_test_windows": len(all_step_preds),
        "s3_s4_confusion": s3_s4_analysis,
    }
    with open(REPORTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    # Save bad cases (top 200)
    bad_cases = sorted(bad_cases, key=lambda x: x["confidence"], reverse=True)[:200]
    with open(REPORTS_DIR / "bad_cases.json", "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, indent=2, ensure_ascii=False)
    print(f"Bad cases saved: {len(bad_cases)}")

    print(f"\nAll reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
