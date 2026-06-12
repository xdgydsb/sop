"""
Step 3: 用segment标注 + 特征时间戳 生成frame-level标签
输入: data/annotations/segment_annotations_v2.csv
      data/features/v2_90/{video_type}/{video_name}.npz
输出: data/labels/frame_labels_v2/{video_type}/{video_name}.npz
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FEAT_DIR = ROOT / "data" / "features" / "v2_90"
LABEL_OUT_DIR = ROOT / "data" / "labels" / "frame_labels_v2"
SEGMENT_CSV = ROOT / "data" / "annotations" / "segment_annotations_v2.csv"
META_OUT = LABEL_OUT_DIR / "label_meta_v2.json"
CHECK_DIR = ROOT / "data" / "annotations" / "label_check_reports"

# Label mappings
STEP_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
VERB_MAP = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3}  # idle,open,put,put,put,close
TARGET_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 1}  # none,box,black_case,white_plug,green_bag,box
EVENT_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

BOUNDARY_IGNORE_SEC = 0.3


def build_labels_for_video(video_name, video_type, segments_df):
    """Generate frame-level labels for one video"""
    # Load features to get timestamps
    feat_path = FEAT_DIR / video_type / f"{video_name}.npz"
    if not feat_path.exists():
        return None, f"Features not found: {feat_path}"

    feat = np.load(feat_path, allow_pickle=True)
    timestamps = feat["timestamps"]
    N = len(timestamps)

    # Initialize labels
    step_label = np.zeros(N, dtype=np.int64)
    verb_label = np.zeros(N, dtype=np.int64)
    target_label = np.zeros(N, dtype=np.int64)
    event_label = np.zeros(N, dtype=np.int64)
    ignore_mask = np.zeros(N, dtype=np.int64)

    # Filter segments for this video
    vid_segs = segments_df[segments_df["video_name"] == video_name].sort_values("start_sec")
    if vid_segs.empty:
        return None, f"No segments for {video_name}"

    for _, seg in vid_segs.iterrows():
        step_id = int(seg["step_id"])
        start_s = float(seg["start_sec"])
        end_s = float(seg["end_sec"])

        # Find frames within this time range
        in_segment = (timestamps >= start_s) & (timestamps <= end_s)
        seg_indices = np.where(in_segment)[0]
        if len(seg_indices) == 0:
            continue

        # Boundary ignore: first and last BOUNDARY_IGNORE_SEC of each segment
        boundary_start = start_s + BOUNDARY_IGNORE_SEC
        boundary_end = end_s - BOUNDARY_IGNORE_SEC

        if boundary_end > boundary_start:
            # Strong label region in middle
            strong = (timestamps >= boundary_start) & (timestamps <= boundary_end)
            strong_indices = np.where(strong & in_segment)[0]

            # Boundary regions
            front = (timestamps >= start_s) & (timestamps < boundary_start)
            back = (timestamps > boundary_end) & (timestamps <= end_s)

            # Assign labels
            step_label[strong_indices] = step_id
            verb_label[strong_indices] = VERB_MAP.get(step_id, 0)
            target_label[strong_indices] = TARGET_MAP.get(step_id, 0)
            event_label[strong_indices] = EVENT_MAP.get(step_id, 0)

            # Boundary regions: assign but mask
            for region in [front, back]:
                ri = np.where(region & in_segment)[0]
                step_label[ri] = step_id
                verb_label[ri] = VERB_MAP.get(step_id, 0)
                target_label[ri] = TARGET_MAP.get(step_id, 0)
                event_label[ri] = EVENT_MAP.get(step_id, 0)
                ignore_mask[ri] = 1
        else:
            # Segment too short: assign and mask everything
            step_label[seg_indices] = step_id
            verb_label[seg_indices] = VERB_MAP.get(step_id, 0)
            target_label[seg_indices] = TARGET_MAP.get(step_id, 0)
            event_label[seg_indices] = EVENT_MAP.get(step_id, 0)
            ignore_mask[seg_indices] = 1

    # Save
    out_path = LABEL_OUT_DIR / video_type / f"{video_name}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        step_label=step_label,
        verb_label=verb_label,
        target_label=target_label,
        event_label=event_label,
        ignore_mask=ignore_mask,
        timestamps=timestamps,
        video_name=video_name,
        video_type=video_type,
    )

    # Build check report
    unique_steps = {int(s) for s in np.unique(step_label) if s > 0}
    step_counts = {int(s): int(np.sum((step_label == s) & (ignore_mask == 0)))
                   for s in range(0, 7)}
    seg_summary = [{
        "step_id": int(s["step_id"]),
        "start_sec": float(s["start_sec"]),
        "end_sec": float(s["end_sec"]),
        "strong_frames": int(np.sum((step_label == int(s["step_id"])) & (ignore_mask == 0))),
    } for _, s in vid_segs.iterrows()]

    report = {
        "video_name": video_name,
        "video_type": video_type,
        "n_feature_frames": N,
        "n_label_frames": N,
        "length_match": True,
        "present_steps": sorted(unique_steps),
        "step_frame_counts": step_counts,
        "segments": seg_summary,
    }
    return report, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="Only check 10 random videos")
    args = parser.parse_args()

    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load segments
    df = pd.read_csv(SEGMENT_CSV)
    print(f"Loaded {len(df)} segments from {df['video_name'].nunique()} videos")

    all_reports = {}
    success = 0
    errors = []

    for video_type in ["ok", "wr"]:
        feat_type_dir = FEAT_DIR / video_type
        if not feat_type_dir.exists():
            continue

        feat_files = sorted(feat_type_dir.glob("*.npz"))
        out_type_dir = LABEL_OUT_DIR / video_type
        out_type_dir.mkdir(parents=True, exist_ok=True)

        for fp in feat_files:
            video_name = fp.stem
            report, err = build_labels_for_video(video_name, video_type, df)
            if report:
                all_reports[video_name] = report
                success += 1
            else:
                errors.append((video_name, err))

    # Sample 10 for detailed check
    import random
    sample_videos = random.sample(list(all_reports.keys()), min(10, len(all_reports)))
    for vname in sample_videos:
        report_path = CHECK_DIR / f"{vname}_frame_label_check.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_reports[vname], f, indent=2, ensure_ascii=False)

    # Save meta
    meta = {
        "total_videos": success,
        "label_types": {
            "step_label": "0=idle,1=open_box,2=put_black_case,3=put_white_plug,4=put_green_bag,5=close_box",
            "verb_label": "0=idle,1=open,2=put,3=close",
            "target_label": "0=none,1=box,2=black_case,3=white_plug,4=green_bag",
            "event_label": "0=none,1=box_opened,2=black_case_in_box,3=white_plug_in_box,4=green_bag_in_box,5=box_closed",
        },
        "boundary_ignore_sec": BOUNDARY_IGNORE_SEC,
        "feature_fps": 15.0,
        "errors": errors,
        "sample_checks": sample_videos,
    }
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[Summary] {success} videos processed, {len(errors)} errors")
    print(f"Labels: {LABEL_OUT_DIR}")
    print(f"Meta: {META_OUT}")
    print(f"Sample checks: {CHECK_DIR}")


if __name__ == "__main__":
    main()
