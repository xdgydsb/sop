"""
Step 1: 将手工npy标注（逐帧label 0-5）转换为标准段标注CSV/JSON

输入: data/annotations/manual_npy/{ok,wr}/*.npy
      data/raw_videos/ (用于获取fps和duration)
输出: data/annotations/segment_annotations_v2.csv
      data/annotations/segment_annotations_v2.json
      data/annotations/label_check_reports/{video_name}.json
"""
import numpy as np
import json
import csv
import cv2
import os
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
MANUAL_NPY_DIR = ROOT / "data" / "annotations" / "manual_npy"
RAW_VIDEOS_DIR = ROOT.parent / "data"  # d:/shabi/data/ has ok/ and wr/
OUT_CSV = ROOT / "data" / "annotations" / "segment_annotations_v2.csv"
OUT_JSON = ROOT / "data" / "annotations" / "segment_annotations_v2.json"
CHECK_DIR = ROOT / "data" / "annotations" / "label_check_reports"

# Step mapping: step_id -> {action_label, verb_label, target_label, event_type}
STEP_MAP = {
    0: {"action_label": "idle",        "verb_label": "idle",  "target_label": "none",        "event_type": "none"},
    1: {"action_label": "open_box",    "verb_label": "open",  "target_label": "box",         "event_type": "box_opened"},
    2: {"action_label": "put_black_case","verb_label": "put",  "target_label": "black_case",  "event_type": "black_case_in_box"},
    3: {"action_label": "put_white_plug","verb_label": "put",  "target_label": "white_plug",  "event_type": "white_plug_in_box"},
    4: {"action_label": "put_green_bag", "verb_label": "put",  "target_label": "green_bag",   "event_type": "green_bag_in_box"},
    5: {"action_label": "close_box",   "verb_label": "close", "target_label": "box",         "event_type": "box_closed"},
}


def num_sort_key(name):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else 0


def get_video_paths(video_type):
    """Get sorted list of (video_name, video_path) for a type"""
    vdir = RAW_VIDEOS_DIR / video_type
    if not vdir.exists():
        return []
    videos = sorted(
        [f for f in os.listdir(vdir) if f.endswith('.avi')],
        key=num_sort_key
    )
    return [(Path(v).stem, vdir / v) for v in videos]


def get_npy_paths(video_type):
    """Get sorted list of (video_name, npy_path) for a type"""
    ndir = MANUAL_NPY_DIR / video_type
    if not ndir.exists():
        return []
    npy_files = sorted(
        [f for f in os.listdir(ndir) if f.endswith('.npy')],
        key=num_sort_key
    )
    # File format: ok_123_y_seg.npy -> video_name = ok_123
    return [(f.replace("_y_seg.npy", ""), ndir / f) for f in npy_files]


def extract_segments(y: np.ndarray, fps: float) -> list:
    """从逐帧标签中提取动作段 [start_sec, end_sec]"""
    segments = []
    prev_label = y[0]
    start_frame = 0

    for i in range(1, len(y)):
        if y[i] != prev_label:
            end_frame = i - 1
            if prev_label != 0:  # Skip idle segments
                segments.append({
                    "step_id": int(prev_label),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_sec": start_frame / fps,
                    "end_sec": (end_frame + 1) / fps,
                })
            prev_label = y[i]
            start_frame = i

    # Last segment
    if prev_label != 0 and start_frame < len(y):
        segments.append({
            "step_id": int(prev_label),
            "start_frame": start_frame,
            "end_frame": len(y) - 1,
            "start_sec": start_frame / fps,
            "end_sec": len(y) / fps,
        })

    return segments


def diagnose_error_type(present_steps, expected_order=(1, 2, 3, 4, 5)):
    """根据实际步骤序列诊断错误类型"""
    actual = [s for s in present_steps if s in expected_order]
    if not actual:
        return "none"

    # Check if all expected steps present and in order
    expected_in_actual = all(e in actual for e in expected_order)
    if expected_in_actual and actual == sorted(actual, key=lambda x: expected_order.index(x) if x in expected_order else 999):
        return "none"

    # Check for missing steps
    missing = [e for e in expected_order if e not in actual]
    if missing:
        return "missing_step"

    # Check for wrong order
    for i in range(len(actual) - 1):
        if actual[i] in expected_order and actual[i+1] in expected_order:
            if expected_order.index(actual[i]) > expected_order.index(actual[i+1]):
                return "wrong_order"

    return "other"


def get_error_type(video_type, segments):
    """Determine error_type for the video based on segment sequence"""
    if video_type == "ok":
        return "none"

    present_steps = [s["step_id"] for s in segments]
    return diagnose_error_type(present_steps)


def process_npy(video_name, npy_path, video_path, video_type):
    """Process a single npy file and return segment rows + check report"""
    y = np.load(npy_path)
    n_frames = len(y)

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = video_frames / fps if fps > 0 else n_frames / 25.0
    cap.release()

    # Frame alignment
    if n_frames != video_frames:
        # Try to adjust: sometimes npy has stride=2
        ratio = video_frames / max(n_frames, 1)
        if abs(ratio - 2.0) < 0.1:
            effective_fps = fps / 2.0
        else:
            effective_fps = fps * n_frames / max(video_frames, 1)
        frame_scale = video_frames / max(n_frames, 1)
    else:
        effective_fps = fps
        frame_scale = 1.0

    segments = extract_segments(y, effective_fps)

    # Build rows
    rows = []
    for seg in segments:
        sid = seg["step_id"]
        step_info = STEP_MAP.get(sid, STEP_MAP[0])
        error_type = get_error_type(video_type, [seg])
        rows.append({
            "video_name": video_name,
            "video_type": video_type,
            "step_id": sid,
            "start_sec": round(seg["start_sec"], 3),
            "end_sec": round(seg["end_sec"], 3),
            "action_label": step_info["action_label"],
            "verb_label": step_info["verb_label"],
            "target_label": step_info["target_label"],
            "event_type": step_info["event_type"],
            "error_type": error_type if video_type == "wr" else "none",
            "source_npy": str(npy_path.name),
            "duration_sec": round(seg["end_sec"] - seg["start_sec"], 3),
        })

    # Build check report
    present_steps = sorted(set(s["step_id"] for s in segments))
    missing_steps = [s for s in range(1, 6) if s not in present_steps]
    gaps = []
    for i in range(len(segments) - 1):
        gap = segments[i+1]["start_sec"] - segments[i]["end_sec"]
        if gap > 0.1:
            gaps.append({
                "between": f"S{segments[i]['step_id']}→S{segments[i+1]['step_id']}",
                "gap_sec": round(gap, 3),
            })

    overlaps = []
    for i in range(len(segments) - 1):
        if segments[i]["end_sec"] > segments[i+1]["start_sec"]:
            overlaps.append({
                "segments": f"S{segments[i]['step_id']}→S{segments[i+1]['step_id']}",
                "overlap_sec": round(segments[i]["end_sec"] - segments[i+1]["start_sec"], 3),
            })

    warnings = []
    for seg in segments:
        if seg["end_sec"] <= seg["start_sec"]:
            warnings.append(f"S{seg['step_id']}: end_sec <= start_sec")
        if seg["start_sec"] > duration:
            warnings.append(f"S{seg['step_id']}: start_sec beyond video duration")

    report = {
        "video_name": video_name,
        "video_type": video_type,
        "video_duration_sec": round(duration, 2),
        "fps": round(fps, 2),
        "effective_fps": round(effective_fps, 2),
        "npy_frames": n_frames,
        "video_frames": video_frames,
        "frame_scale": round(frame_scale, 3),
        "annotated_segments": [
            {
                "step_id": s["step_id"],
                "action_label": s["action_label"],
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
                "duration_sec": s["duration_sec"],
            }
            for s in rows
        ],
        "present_steps": present_steps,
        "missing_steps": missing_steps,
        "overlapping_segments": overlaps,
        "gaps_between_segments": gaps,
        "warnings": warnings,
    }

    return rows, report


def main():
    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_reports = {}
    video_types = ["ok", "wr"]

    for vtype in video_types:
        npy_list = get_npy_paths(vtype)
        video_list = get_video_paths(vtype)
        video_dict = dict(video_list)

        print(f"\n{'='*60}")
        print(f"Processing {vtype}: {len(npy_list)} npy files, {len(video_list)} videos")
        print(f"{'='*60}")

        skipped = 0
        for video_name, npy_path in npy_list:
            # Match npy to video
            if video_name not in video_dict:
                print(f"  SKIP {video_name}: no matching video found")
                skipped += 1
                continue

            video_path = video_dict[video_name]
            rows, report = process_npy(video_name, npy_path, video_path, vtype)

            all_rows.extend(rows)
            all_reports[video_name] = report

            # Print summary
            segs_str = "→".join([f"S{r['step_id']}" for r in rows])
            print(f"  {video_name}: {len(rows)} segments [{segs_str}] "
                  f"npy={report['npy_frames']}f vid={report['video_frames']}f "
                  f"fps={report['fps']:.0f}")

            # Save individual check report
            report_path = CHECK_DIR / f"{video_name}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        if skipped:
            print(f"  Skipped {skipped} files (no matching video)")

    # Write CSV
    csv_fields = [
        "video_name", "video_type", "step_id", "start_sec", "end_sec",
        "action_label", "verb_label", "target_label", "event_type",
        "error_type", "source_npy", "duration_sec",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    # Write JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "total_videos": len(all_reports),
            "videos": all_reports,
        }, f, indent=2, ensure_ascii=False)

    # Summary stats
    ok_count = sum(1 for r in all_rows if r["video_type"] == "ok")
    wr_count = sum(1 for r in all_rows if r["video_type"] == "wr")
    ok_videos = len([v for v in all_reports if all_reports[v]["video_type"] == "ok"])
    wr_videos = len([v for v in all_reports if all_reports[v]["video_type"] == "wr"])

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  OK: {ok_videos} videos, {ok_count} segments")
    print(f"  WR: {wr_videos} videos, {wr_count} segments")
    print(f"  Total: {ok_videos + wr_videos} videos, {ok_count + wr_count} segments")
    print(f"  CSV: {OUT_CSV}")
    print(f"  JSON: {OUT_JSON}")
    print(f"  Reports: {CHECK_DIR}/")


if __name__ == "__main__":
    main()
