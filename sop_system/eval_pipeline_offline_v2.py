"""
Step 7: 事件级和FSM级离线评估
输入: 完整视频features, labels, 训练好的模型, EventDetector, FSM
输出: reports/temporal_v2_90/event_sequence_eval.json
      reports/temporal_v2_90/fsm_eval.json
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.sop_fsm import SOPStateMachine, SOPStep
from train_temporal_v2 import TCNBiGRU

FEAT_DIR = ROOT / "data" / "features" / "v2_90"
LABEL_DIR = ROOT / "data" / "labels" / "frame_labels_v2"
MODEL_DIR = ROOT / "models" / "temporal" / "v2_90_tcn_bigru"
REPORTS_DIR = ROOT / "reports" / "temporal_v2_90"
SEGMENT_CSV = ROOT / "data" / "annotations" / "segment_annotations_v2.csv"

T = 48
STRIDE = 4  # finer stride for frame-level prediction
FPS = 15.0


def sliding_window_inference(features, model, device, T=48, stride=4):
    """Run sliding window and produce frame-level step probabilities"""
    N = len(features)
    n_classes = 6
    frame_probs = np.zeros((N, n_classes), dtype=np.float64)
    frame_counts = np.zeros(N, dtype=np.int32)

    model.eval()
    x_tensor = torch.FloatTensor(features)
    half_w = T // 4

    with torch.no_grad():
        for start in range(0, N - T, stride):
            end = start + T
            x = x_tensor[start:end].unsqueeze(0).to(device)
            step_logits, _, _ = model(x)
            probs = torch.softmax(step_logits, dim=1).cpu().numpy()
            # Assign to center region
            assign_start = start + half_w
            assign_end = end - half_w
            for fi in range(assign_start, assign_end):
                if 0 <= fi < N:
                    frame_probs[fi] += probs[0]
                    frame_counts[fi] += 1

    # Handle uncovered frames
    for fi in range(N):
        if frame_counts[fi] == 0:
            for d in range(1, 20):
                if fi + d < N and frame_counts[fi + d] > 0:
                    frame_probs[fi] = frame_probs[fi + d]
                    frame_counts[fi] = 0.5  # lower weight
                    break
                if fi - d >= 0 and frame_counts[fi - d] > 0:
                    frame_probs[fi] = frame_probs[fi - d]
                    frame_counts[fi] = 0.5
                    break

    # Normalize
    frame_probs /= np.maximum(frame_counts[:, None], 0.5)

    # EMA smooth
    alpha = 0.35
    smoothed = np.zeros_like(frame_probs)
    for c in range(n_classes):
        ema = frame_probs[0, c]
        for fi in range(N):
            ema = alpha * frame_probs[fi, c] + (1 - alpha) * ema
            smoothed[fi, c] = ema

    preds = np.argmax(smoothed, axis=1)
    confs = np.max(smoothed, axis=1)
    return preds, confs, smoothed


def simulate_fsm_on_predictions(preds, confs, ground_truth_steps, timeout=90.0):
    """Run FSM on model predictions, compare with ground truth steps.

    Idle (step 0) predictions are treated as neutral — they don't advance FSM
    but also don't cause errors (idle can appear between any steps).
    After S5 (close_box), reaching idle is a natural sequence end.
    """
    fsm = SOPStateMachine(timeout=timeout, min_step_duration=0.3)
    events = []
    prev_step = -1
    steps_seen = set()  # Track which action steps (1-5) were completed

    for fi, (pred, conf) in enumerate(zip(preds, confs)):
        p = int(pred)
        c = float(conf)
        ts = fi / FPS

        # Idle is neutral — record but don't feed to FSM (avoids false WRONG_ORDER)
        if p == 0:
            if fsm.current_step == SOPStep.S5_CLOSE:
                # After S5, seeing idle means sequence naturally ended
                fsm.current_step = SOPStep.COMPLETE
                if prev_step != 6:
                    events.append({
                        "frame": fi, "step": 6, "model_pred": p,
                        "is_correct": True, "has_error": False,
                        "error_type": "", "message": "Sequence complete (idle after S5)",
                    })
                    prev_step = 6
                continue
            # Skip other idle frames — no state change
            continue

        result = fsm.validate(p, c, physical_state_ok=True, timestamp=ts)
        if result.is_correct and p >= 1 and p <= 5:
            steps_seen.add(p)
        if result.step_id != prev_step:
            events.append({
                "frame": fi,
                "step": result.step_id,
                "model_pred": p,
                "is_correct": result.is_correct,
                "has_error": result.has_error,
                "error_type": result.error_type,
                "message": result.message,
            })
            prev_step = result.step_id

    # Post-sequence check: if FSM reached at least S5 via valid progression,
    # consider the sequence completed (may have ended with idle not seen in loop)
    if fsm.current_step == SOPStep.S5_CLOSE:
        fsm.current_step = SOPStep.COMPLETE
        if prev_step != 6:
            events.append({
                "frame": len(preds) - 1, "step": 6, "model_pred": int(preds[-1]),
                "is_correct": True, "has_error": False,
                "error_type": "", "message": "Sequence complete (end of video at S5)",
            })

    gt_steps = sorted(set(int(s) for s in ground_truth_steps if s > 0))
    fsm_path = [e["step"] for e in events if e["step"] > 0]
    fsm_errors = [e for e in events if e["has_error"]]
    final_step = fsm.current_step.value
    fsm_complete = fsm.current_step == SOPStep.COMPLETE
    gt_steps_set = set(gt_steps)
    all_steps_seen = steps_seen == gt_steps_set if gt_steps_set else steps_seen >= {1, 2, 3, 4, 5}

    return {
        "events": events,
        "fsm_path": fsm_path,
        "final_step": final_step,
        "fsm_complete": fsm_complete,
        "fsm_errors": fsm_errors,
        "gt_steps": gt_steps,
        "steps_seen": sorted(steps_seen),
        "all_steps_seen": all_steps_seen,
    }


def evaluate_video(video_name, video_type, model, device, segments_df):
    """Evaluate one video end-to-end"""
    feat_path = FEAT_DIR / video_type / f"{video_name}.npz"
    label_path = LABEL_DIR / video_type / f"{video_name}.npz"

    if not feat_path.exists() or not label_path.exists():
        return None

    feat = np.load(feat_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)

    features = feat["features"]
    step_label = labels["step_label"]
    ignore_mask = labels["ignore_mask"]
    timestamps = feat["timestamps"]

    # Model inference
    preds, confs, probs = sliding_window_inference(
        features, model, device, T=T, stride=STRIDE)

    # Strong label regions (no ignore)
    strong_mask = (ignore_mask == 0)
    strong_preds = preds[strong_mask]
    strong_labels = step_label[strong_mask]

    if len(strong_preds) == 0:
        return None

    # Per-frame accuracy on strong regions
    frame_acc = (strong_preds == strong_labels).mean()

    # FSM simulation
    fsm_result = simulate_fsm_on_predictions(preds, confs, step_label)

    # Event detection evaluation (uses segment CSV as GT)
    event_results = evaluate_events(preds, ignore_mask, confs, segments_df,
                                    video_name, timestamps)

    return {
        "video_name": video_name,
        "video_type": video_type,
        "n_frames": len(features),
        "frame_acc_strong": float(frame_acc),
        "fsm_path": fsm_result["fsm_path"],
        "final_step": fsm_result["final_step"],
        "fsm_complete": fsm_result["fsm_complete"],
        "fsm_errors": [(e["error_type"], e["frame"]) for e in fsm_result["fsm_errors"]],
        "gt_steps": fsm_result["gt_steps"],
        "steps_seen": fsm_result["steps_seen"],
        "event_metrics": event_results,
    }


def evaluate_events(preds, ignore_mask, confs, segments_df, video_name, timestamps):
    """Evaluate event detection using segment timestamps as ground truth.

    Ground truth: segment start_sec from CSV (when each step begins).
    Predictions: model-predicted transitions in strong-label regions.
    A detection is correct if within ±45 frames (~3s at 15fps) of a GT event.
    """
    vid_segs = segments_df[segments_df["video_name"] == video_name].sort_values("start_sec")

    # Ground truth events: start time of each segment (step > 0, not idle)
    gt_events = defaultdict(list)
    for _, seg in vid_segs.iterrows():
        step_id = int(seg["step_id"])
        if step_id > 0:
            # Find closest feature frame to segment start time
            start_s = float(seg["start_sec"])
            gt_frame = int(np.argmin(np.abs(timestamps - start_s)))
            gt_events[step_id].append({
                "frame": gt_frame,
                "start_sec": start_s,
            })

    # Predicted transitions in non-ignore regions
    events_detected = defaultdict(list)
    prev = preds[0]
    for fi in range(1, len(preds)):
        if preds[fi] != prev and ignore_mask[fi] == 0:
            new_step = int(preds[fi])
            if new_step > 0:  # Only count transitions to action steps
                events_detected[new_step].append({
                    "frame": fi, "prev_step": int(prev), "conf": float(confs[fi])
                })
        prev = preds[fi]

    # Compare for each step 1-5
    results = {}
    for step in range(1, 6):
        detected = events_detected.get(step, [])
        gt = gt_events.get(step, [])

        matched = 0
        false_positives = 0
        used_gt = set()

        for det in sorted(detected, key=lambda x: x["frame"]):
            best_match = None
            best_dist = 999
            for gi, g in enumerate(sorted(gt, key=lambda x: x["frame"])):
                if gi in used_gt:
                    continue
                dist = abs(det["frame"] - g["frame"])
                if dist < 45 and dist < best_dist:
                    best_match = gi
                    best_dist = dist
            if best_match is not None:
                matched += 1
                used_gt.add(best_match)
            else:
                false_positives += 1

        results[f"S{step}"] = {
            "gt_count": len(gt),
            "detected_count": len(detected),
            "correct_detections": matched,
            "false_positives": false_positives,
            "missed": max(0, len(gt) - matched),
            "recall": round(matched / max(len(gt), 1), 4),
            "precision": round(matched / max(len(detected), 1), 4),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-type", default="all")
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load segment annotations for event GT
    segments_df = pd.read_csv(SEGMENT_CSV)
    print(f"Loaded {len(segments_df)} segments for event GT")

    # Load model
    ckpt_path = MODEL_DIR / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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

    video_types = ["ok", "wr"] if args.video_type == "all" else [args.video_type]
    all_results = []

    for vtype in video_types:
        feat_type_dir = FEAT_DIR / vtype
        if not feat_type_dir.exists():
            continue

        videos = sorted(feat_type_dir.glob("*.npz"))
        if args.max_videos:
            videos = videos[:args.max_videos]

        print(f"\nEvaluating {vtype}: {len(videos)} videos")

        for vp in videos:
            video_name = vp.stem
            result = evaluate_video(video_name, vtype, model, device, segments_df)
            if result:
                all_results.append(result)
                # OK pass: FSM completed without errors, all steps seen
                # WR detection: FSM errors OR FSM didn't complete
                if vtype == "ok":
                    status = "PASS" if (result["fsm_complete"] and
                                        len(result["fsm_errors"]) == 0) else "FAIL"
                else:
                    status = "OK" if (not result["fsm_complete"] or
                                      len(result["fsm_errors"]) > 0) else "MISS"
                print(f"  {video_name}: acc={result['frame_acc_strong']:.3f} "
                      f"fsm_path={result['fsm_path']} final=S{result['final_step']} "
                      f"errors={len(result['fsm_errors'])} [{status}]")

    if not all_results:
        print("No results")
        return

    # Aggregate event metrics
    event_summary = defaultdict(lambda: {"recall": [], "precision": [], "gt_count": 0})
    for r in all_results:
        for step_key, metrics in r["event_metrics"].items():
            event_summary[step_key]["recall"].append(metrics["recall"])
            event_summary[step_key]["precision"].append(metrics["precision"])
            event_summary[step_key]["gt_count"] += metrics["gt_count"]

    final_event = {}
    for step_key, data in event_summary.items():
        final_event[step_key] = {
            "mean_recall": round(np.mean(data["recall"]), 4),
            "mean_precision": round(np.mean(data["precision"]), 4),
            "total_gt_events": data["gt_count"],
        }

    with open(REPORTS_DIR / "event_sequence_eval.json", "w", encoding="utf-8") as f:
        json.dump(final_event, f, indent=2, ensure_ascii=False)

    # Aggregate FSM metrics
    ok_results = [r for r in all_results if r["video_type"] == "ok"]
    wr_results = [r for r in all_results if r["video_type"] == "wr"]

    # OK pass: FSM completed AND no errors
    ok_pass = sum(1 for r in ok_results if r["fsm_complete"] and len(r["fsm_errors"]) == 0)
    # WR detected: FSM did NOT complete OR has errors
    wr_detected = sum(1 for r in wr_results
                      if not r["fsm_complete"] or len(r["fsm_errors"]) > 0)

    # Error type analysis (across all videos with errors)
    error_types = Counter()
    for r in all_results:
        for err_type, _ in r["fsm_errors"]:
            error_types[err_type] += 1

    # Per-step event pass rate for OK videos
    ok_event_pass = sum(1 for r in ok_results
                        if r["fsm_complete"] and len(r["fsm_errors"]) == 0)

    fsm_eval = {
        "ok_videos": len(ok_results),
        "wr_videos": len(wr_results),
        "ok_pass_rate": round(ok_pass / max(len(ok_results), 1), 4),
        "wr_detection_rate": round(wr_detected / max(len(wr_results), 1), 4),
        "error_type_distribution": dict(error_types),
        "mean_frame_acc": round(np.mean([r["frame_acc_strong"] for r in all_results]), 4),
        "ok_mean_frame_acc": round(np.mean([r["frame_acc_strong"] for r in ok_results]), 4),
        "wr_mean_frame_acc": round(np.mean([r["frame_acc_strong"] for r in wr_results]), 4),
    }

    with open(REPORTS_DIR / "fsm_eval.json", "w", encoding="utf-8") as f:
        json.dump(fsm_eval, f, indent=2, ensure_ascii=False)

    # Print summary
    ok_pass_list = [r["video_name"] for r in ok_results
                    if r["fsm_complete"] and len(r["fsm_errors"]) == 0]
    ok_fail_list = [(r["video_name"], r["fsm_errors"])
                    for r in ok_results
                    if not r["fsm_complete"] or len(r["fsm_errors"]) > 0]

    print(f"\n{'='*60}")
    print(f"PIPELINE EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  OK: {ok_pass}/{len(ok_results)} pass ({fsm_eval['ok_pass_rate']:.1%})")
    print(f"  WR: {wr_detected}/{len(wr_results)} detected ({fsm_eval['wr_detection_rate']:.1%})")
    print(f"  Mean frame acc: {fsm_eval['mean_frame_acc']:.3f} "
          f"(OK={fsm_eval['ok_mean_frame_acc']:.3f}, WR={fsm_eval['wr_mean_frame_acc']:.3f})")
    print(f"\n  Event detection:")
    for step_key, metrics in final_event.items():
        print(f"    {step_key}: recall={metrics['mean_recall']:.3f} "
              f"precision={metrics['mean_precision']:.3f} "
              f"(gt={metrics['total_gt_events']})")
    print(f"\n  Error types (all): {dict(error_types)}")
    if ok_fail_list:
        print(f"\n  Failed OK videos (first 10): {ok_fail_list[:10]}")


if __name__ == "__main__":
    main()
