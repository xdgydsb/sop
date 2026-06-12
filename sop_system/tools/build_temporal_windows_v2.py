"""
Step 4: 按完整视频滑窗生成训练索引
输入: data/features/v2_90/ + data/labels/frame_labels_v2/
输出: data/datasets/temporal_v2_90_T48_S8/

窗口: T=48, stride=8, 按video_name划分train/val/test
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import random

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FEAT_DIR = ROOT / "data" / "features" / "v2_90"
LABEL_DIR = ROOT / "data" / "labels" / "frame_labels_v2"
OUT_DIR = ROOT / "data" / "datasets" / "temporal_v2_90_T48_S8"
SEGMENT_CSV = ROOT / "data" / "annotations" / "segment_annotations_v2.csv"

T = 48
STRIDE = 8
YUANZI_MAX_RATIO = 0.15
IDLE_MAX_RATIO = 1.0  # idle:action ratio cap (1:1)
MAJORITY_THRESH = 0.6
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42


def build_windows_for_video(video_name, video_type):
    """Generate window indices for one video"""
    feat_path = FEAT_DIR / video_type / f"{video_name}.npz"
    label_path = LABEL_DIR / video_type / f"{video_name}.npz"

    if not feat_path.exists() or not label_path.exists():
        return []

    feat = np.load(feat_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)

    N = len(feat["features"])
    step_label = labels["step_label"]
    ignore_mask = labels["ignore_mask"]
    verb_label = labels["verb_label"]
    target_label = labels["target_label"]
    event_label = labels["event_label"]

    windows = []
    for start in range(0, N - T, STRIDE):
        end = start + T
        center = start + T // 2

        # Discard if center has ignore_mask
        if ignore_mask[center] == 1:
            continue

        center_step = int(step_label[center])
        center_verb = int(verb_label[center])
        center_target = int(target_label[center])
        center_event = int(event_label[center])

        # Check label purity in window
        window_steps = step_label[start:end][ignore_mask[start:end] == 0]
        if len(window_steps) == 0:
            continue

        majority_step = Counter(window_steps.tolist()).most_common(1)[0]
        purity = majority_step[1] / len(window_steps)

        # Discard if too mixed
        if purity < MAJORITY_THRESH:
            continue

        windows.append({
            "video_name": video_name,
            "video_type": video_type,
            "feature_path": str(feat_path.relative_to(ROOT)),
            "label_path": str(label_path.relative_to(ROOT)),
            "start_idx": start,
            "end_idx": end,
            "center_idx": center,
            "center_step_label": center_step,
            "center_verb_label": center_verb,
            "center_target_label": center_target,
            "center_event_label": center_event,
        })

    return windows


def split_videos(ok_videos, wr_videos, yuanzi_videos):
    """Split videos into train/val/test, ensuring balanced distribution"""
    random.seed(SEED)

    # Shuffle within each type
    ok_shuffled = sorted(ok_videos)
    random.shuffle(ok_shuffled)
    wr_shuffled = sorted(wr_videos)
    random.shuffle(wr_shuffled)
    yuanzi_shuffled = sorted(yuanzi_videos)
    random.shuffle(yuanzi_shuffled)

    def split_list(lst, train_r, val_r):
        n = len(lst)
        n_train = max(1, int(n * train_r))
        n_val = max(1, int(n * val_r))
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    ok_train, ok_val, ok_test = split_list(ok_shuffled, TRAIN_RATIO, VAL_RATIO)
    wr_train, wr_val, wr_test = split_list(wr_shuffled, TRAIN_RATIO, VAL_RATIO)
    yu_train, yu_val, yu_test = split_list(yuanzi_shuffled, TRAIN_RATIO, VAL_RATIO)

    return {
        "train": ok_train + wr_train + yu_train,
        "val": ok_val + wr_val + yu_val,
        "test": ok_test + wr_test + yu_test,
    }, {
        "ok": {"train": ok_train, "val": ok_val, "test": ok_test},
        "wr": {"train": wr_train, "val": wr_val, "test": wr_test},
        "yuanzi": {"train": yu_train, "val": yu_val, "test": yu_test},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-idle-windows", type=int, default=None,
                        help="Max idle windows (default: match action count)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all video+window data
    all_windows = []
    video_type_counts = defaultdict(int)
    step_counts = Counter()

    for video_type in ["ok", "wr", "yuanzi"]:
        feat_type_dir = FEAT_DIR / video_type
        if not feat_type_dir.exists():
            continue

        for fp in sorted(feat_type_dir.glob("*.npz")):
            video_name = fp.stem
            windows = build_windows_for_video(video_name, video_type)
            for w in windows:
                all_windows.append(w)
                step_counts[w["center_step_label"]] += 1
            video_type_counts[video_type] += 1
            if windows:
                print(f"  {video_name}: {len(windows)} windows")

    print(f"\nTotal windows: {len(all_windows)}")
    print(f"Video counts: {dict(video_type_counts)}")
    print(f"Step distribution: {dict(sorted(step_counts.items()))}")

    # Limit idle windows
    idle_windows = [w for w in all_windows if w["center_step_label"] == 0]
    action_windows = [w for w in all_windows if w["center_step_label"] > 0]
    max_idle = args.max_idle_windows or len(action_windows)

    if len(idle_windows) > max_idle:
        random.seed(SEED)
        idle_windows = random.sample(idle_windows, max_idle)
        all_windows = idle_windows + action_windows
        print(f"  Idle windows limited to {len(idle_windows)} (was {len(idle_windows)})")

    # Split by video
    ok_videos = [v for v in set(w["video_name"] for w in all_windows
                                if w["video_type"] == "ok")]
    wr_videos = [v for v in set(w["video_name"] for w in all_windows
                                if w["video_type"] == "wr")]
    yuanzi_videos = [v for v in set(w["video_name"] for w in all_windows
                                   if w["video_type"] == "yuanzi")]

    splits, split_detail = split_videos(ok_videos, wr_videos, yuanzi_videos)

    # Assign windows to splits
    split_windows = {"train": [], "val": [], "test": []}
    split_step_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    for w in all_windows:
        vname = w["video_name"]
        assigned = None
        for split_name in ["train", "val", "test"]:
            if vname in splits[split_name]:
                assigned = split_name
                break
        if assigned:
            split_windows[assigned].append(w)
            split_step_counts[assigned][w["center_step_label"]] += 1

    # Check for leaks
    train_vids = set(w["video_name"] for w in split_windows["train"])
    val_vids = set(w["video_name"] for w in split_windows["val"])
    test_vids = set(w["video_name"] for w in split_windows["test"])
    leaks = []
    if train_vids & val_vids:
        leaks.append(f"train∩val: {train_vids & val_vids}")
    if train_vids & test_vids:
        leaks.append(f"train∩test: {train_vids & test_vids}")
    if val_vids & test_vids:
        leaks.append(f"val∩test: {val_vids & test_vids}")
    if leaks:
        print(f"\nWARNING: Video leaks detected: {leaks}")
    else:
        print("\nNo video leaks — split is clean")

    # Save split indices
    step_names = {0: "idle", 1: "S1_open", 2: "S2_earphone", 3: "S3_charger",
                  4: "S4_bag", 5: "S5_close", 6: "complete"}

    for split_name in ["train", "val", "test"]:
        out_path = OUT_DIR / f"{split_name}_index.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "windows": split_windows[split_name],
                "count": len(split_windows[split_name]),
                "step_distribution": {step_names.get(k, str(k)): v
                                      for k, v in sorted(split_step_counts[split_name].items())},
            }, f, indent=2, ensure_ascii=False)
        print(f"  {split_name}: {len(split_windows[split_name])} windows "
              f"({len(set(w['video_name'] for w in split_windows[split_name]))} videos)")

    # Save split_by_video
    with open(OUT_DIR / "split_by_video.json", "w", encoding="utf-8") as f:
        json.dump(split_detail, f, indent=2, ensure_ascii=False)

    # Save class distribution
    total_per_step = Counter()
    for w in all_windows:
        total_per_step[w["center_step_label"]] += 1
    class_dist = {
        "total": {step_names.get(k, str(k)): v for k, v in sorted(total_per_step.items())},
        "train": {step_names.get(k, str(k)): v for k, v in sorted(split_step_counts["train"].items())},
        "val": {step_names.get(k, str(k)): v for k, v in sorted(split_step_counts["val"].items())},
        "test": {step_names.get(k, str(k)): v for k, v in sorted(split_step_counts["test"].items())},
    }
    with open(OUT_DIR / "class_distribution.json", "w", encoding="utf-8") as f:
        json.dump(class_dist, f, indent=2, ensure_ascii=False)

    # Save dataset meta
    meta = {
        "window_size": T,
        "stride": STRIDE,
        "feature_dim": 90,
        "feature_version": "v2_90",
        "majority_threshold": MAJORITY_THRESH,
        "idle_max_ratio": IDLE_MAX_RATIO,
        "yuanzi_max_ratio": YUANZI_MAX_RATIO,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "split_seed": SEED,
        "total_windows": len(all_windows),
        "train_windows": len(split_windows["train"]),
        "val_windows": len(split_windows["val"]),
        "test_windows": len(split_windows["test"]),
        "train_videos": len(set(w["video_name"] for w in split_windows["train"])),
        "val_videos": len(set(w["video_name"] for w in split_windows["val"])),
        "test_videos": len(set(w["video_name"] for w in split_windows["test"])),
        "no_video_leaks": len(leaks) == 0,
    }
    with open(OUT_DIR / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nDataset ready at: {OUT_DIR}")


if __name__ == "__main__":
    main()
