"""
Phase 3 (方案B): 真正增量流式回放测试 — TemporalPredictorV2 + FSM

从原始 .avi 逐帧跑完整管线，使用修复后的 TemporalPredictorV2 做增量推理:
  逐帧提取 90d 特征 → predictor.predict(feature)
  → get_latest_mature_prediction()
  → 非 None 时喂入 SOPStateMachine
  → 记录完整 trace

关键约束:
  - 使用修复后的 TemporalPredictorV2 (stride=4 窗口调度, 无 fallback)
  - 未成熟帧返回 None → FSM 不消费
  - 不改 FSM, 不改模型, 不改特征
  - 不新增 EventDetector

Usage:
  python tools/replay_realtime_incremental_v2.py --video-type ok --max-videos 20
  python tools/replay_realtime_incremental_v2.py --video-type wr --max-videos 20
  python tools/replay_realtime_incremental_v2.py --video-type yuanzi --max-videos 5
"""
import sys
import json
import argparse
import time
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.yolo_detector import YOLODetector
from engine.hand_detector import HandDetector
from engine.sop_fsm import SOPStateMachine, SOPStep, STEP_NAMES
from engine.temporal_predictor_v2 import TemporalPredictorV2
from tools.extract_features_v2_90 import FeatureExtractorV2Standalone, INTERACT_OBJS

# ── Paths ──
RAW_VIDEOS_DIR = ROOT.parent / "data"
MODEL_DIR = ROOT / "models" / "temporal" / "v2_90_tcn_bigru"
REPORTS_DIR = ROOT / "reports" / "realtime_replay_v2_incremental"

FPS = 15.0
T = 48
STRIDE = 4
N_CLASSES = 6


def load_model_for_predictor(model_path: str) -> str:
    """Validate model checkpoint and return path (predictor loads it internally)"""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    assert ckpt.get("input_dim") == 90, (
        f"WRONG MODEL: input_dim={ckpt.get('input_dim')}, expected 90"
    )
    print(f"[Model] Validated: input_dim=90, hidden_dim={ckpt['hidden_dim']}, "
          f"step_classes={ckpt['num_classes_step']}")
    return model_path


def extract_features_from_video(video_path: Path,
                                 yolo: YOLODetector,
                                 hand_detector: HandDetector,
                                 extractor: FeatureExtractorV2Standalone,
                                 target_fps: float = FPS
                                 ) -> Dict:
    """第一遍: 遍历原始视频，逐帧提取90维特征 (与方案A完全相同)

    Returns dict with:
        features: np.ndarray [N, 90]
        timestamps: np.ndarray [N]
        frame_indices: np.ndarray [N] — original video frame indices
        video_info: dict
        trace: list — per-original-frame YOLO/MediaPipe diagnostics
    """
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    n_target = max(1, int(duration * target_fps))
    target_frame_set = set(
        np.linspace(0, total_frames - 1, n_target, dtype=np.int32).tolist())

    # Reset state
    yolo.reset_tracking()
    extractor.reset()
    hand_detector.palm_velocity.clear()

    feature_list = []
    timestamp_list = []
    frame_idx_list = []
    trace = []
    fc = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fc += 1
        timestamp = (fc - 1) / video_fps

        # YOLO + MediaPipe (every frame for tracking continuity)
        detections = yolo.detect(frame)
        hands = hand_detector.detect(frame)

        box_bbox = yolo.get_box_bbox(detections)
        box_state_str, _ = yolo.get_box_state(detections)

        # Interaction for hand selection
        if box_bbox:
            interaction = extractor.compute_interaction(detections, hands, box_bbox, h, w)
        else:
            interaction = {"hand_obj_iou": np.zeros((2, 5), dtype=np.float32),
                          "hand_obj_dist": np.zeros((2, 5), dtype=np.float32)}

        hand_detector.select_active_hand(hands, target_object_bbox=None, box_bbox=box_bbox)

        # Update tracked object states
        if box_bbox:
            for det in detections:
                if det.track_id >= 0 and det.cls_name in INTERACT_OBJS:
                    cx, cy = det.center
                    bx1, by1, bx2, by2 = box_bbox
                    in_box = (bx1 <= cx <= bx2 and by1 <= cy <= by2)
                    in_box_ratio = yolo.compute_in_box_ratio(det.bbox, box_bbox)
                    yolo.update_in_box_status(det.track_id, in_box or in_box_ratio > 0.3)

        # Extract feature at target frames
        if (fc - 1) in target_frame_set:
            feat = extractor.extract(frame, detections, hands, box_bbox, box_state_str)
            feature_list.append(feat)
            timestamp_list.append(timestamp)
            frame_idx_list.append(fc - 1)

        # Trace entry for diagnostics
        det_dict = {d.cls_name: d for d in detections}
        box_open_conf = det_dict["box_open"].confidence if "box_open" in det_dict else 0.0
        box_closed_conf = det_dict["box_closed"].confidence if "box_closed" in det_dict else 0.0

        trace.append({
            "frame": fc,
            "timestamp": round(timestamp, 3),
            "box_is_open": box_open_conf > 0.3,
            "box_is_closed": box_closed_conf > 0.3 and box_open_conf < 0.3,
            "n_detections": len(detections),
            "has_hand": any(h.bbox is not None for h in hands),
        })

    cap.release()

    features = np.array(feature_list, dtype=np.float32)
    timestamps_arr = np.array(timestamp_list, dtype=np.float32)
    frame_indices_arr = np.array(frame_idx_list, dtype=np.int32)

    return {
        "features": features,
        "timestamps": timestamps_arr,
        "frame_indices": frame_indices_arr,
        "video_info": {
            "video_fps": video_fps,
            "total_frames": total_frames,
            "n_feature_frames": len(feature_list),
            "duration": round(duration, 1),
            "width": w,
            "height": h,
        },
        "trace": trace,
    }


def run_incremental_fsm(predictor: TemporalPredictorV2,
                         features: np.ndarray,
                         timestamps: np.ndarray,
                         frame_indices: np.ndarray,
                         timeout: float = 90.0
                         ) -> Dict:
    """Run incremental inference + FSM on pre-extracted features.

    This is the CORE of the test — true streaming inference:
      for each feature frame:
        predictor.predict(feature)
        mature = predictor.get_latest_mature_prediction()
        if mature is not None:
            feed to FSM (with idle-skip logic matching simulate_fsm_on_predictions)

    Returns:
        event_trace: per-feature-frame FSM state
        fsm_events: state transition events
        temporal_trace: per-mature-frame raw predictions
        mature_predictions: all mature frame predictions with top3
        none_stats: None prediction statistics
    """
    fsm = SOPStateMachine(timeout=timeout, min_step_duration=0.3)
    N = len(features)

    # Trace containers
    event_trace = []        # per-feature-frame state
    fsm_events = []         # state transition events
    temporal_trace = []     # per mature frame predictions
    mature_predictions = [] # detailed mature frame predictions

    # Statistics
    none_prediction_count = 0
    none_consumed_by_fsm_count = 0
    prev_fsm_step = -1

    predictor.reset()

    for fi in range(N):
        feat = features[fi]
        ts = float(timestamps[fi])
        orig_frame = int(frame_indices[fi])

        # ── Feed feature to predictor ──
        window_result = predictor.predict(feat)

        # ── Get latest mature prediction ──
        mature = predictor.get_latest_mature_prediction()

        # Per-frame trace entry
        trace_entry = {
            "feature_idx": fi,
            "original_frame": orig_frame,
            "timestamp": round(ts, 3),
            "mature_frame_idx": predictor.latest_mature_frame,
            "prediction_available": mature is not None,
        }

        if mature is None:
            none_prediction_count += 1
            trace_entry["fsm_step"] = fsm.current_step.value
            trace_entry["fsm_step_name"] = STEP_NAMES[fsm.current_step]
            trace_entry["fsm_updated"] = False
            event_trace.append(trace_entry)
            continue

        # ── Mature prediction available ──
        pred_step = mature["step"]
        pred_conf = mature["confidence"]
        pred_probs = mature["step_probs"]
        top3 = mature["top3"]

        # Record temporal prediction
        temporal_entry = {
            "feature_idx": fi,
            "mature_frame_idx": predictor.latest_mature_frame,
            "original_frame": orig_frame,
            "timestamp": round(ts, 3),
            "step": pred_step,
            "confidence": round(pred_conf, 4),
            "step_probs": [round(float(p), 4) for p in pred_probs],
            "top3": [(int(s), round(float(c), 4)) for s, c in top3],
            "ema_state": [round(float(p), 4) for p in predictor._ema_state] if predictor._ema_state is not None else [],
        }
        temporal_trace.append(temporal_entry)
        mature_predictions.append(temporal_entry)

        # ── Feed to FSM (with idle-skip logic matching simulate_fsm_on_predictions) ──
        fsm_updated = False
        if pred_step == 0:
            # Idle is neutral — skip FSM update
            # Exception: after S5, idle → COMPLETE
            if fsm.current_step == SOPStep.S5_CLOSE:
                fsm.current_step = SOPStep.COMPLETE
                if prev_fsm_step != 6:
                    fsm_events.append({
                        "feature_idx": fi,
                        "original_frame": orig_frame,
                        "timestamp": round(ts, 3),
                        "step": 6,
                        "step_name": STEP_NAMES[SOPStep.COMPLETE],
                        "model_pred": 0,
                        "is_correct": True,
                        "has_error": False,
                        "error_type": "",
                        "message": "Sequence complete (idle after S5)",
                    })
                    prev_fsm_step = 6
                fsm_updated = True
            # else: idle skipped, no FSM update
        else:
            result = fsm.validate(pred_step, pred_conf, physical_state_ok=True, timestamp=ts)
            if result.is_correct and 1 <= pred_step <= 5:
                pass  # steps_seen tracked via FSM internal state

            if result.step_id != prev_fsm_step:
                fsm_events.append({
                    "feature_idx": fi,
                    "original_frame": orig_frame,
                    "timestamp": round(ts, 3),
                    "step": result.step_id,
                    "step_name": STEP_NAMES[SOPStep(result.step_id)],
                    "model_pred": pred_step,
                    "model_confidence": round(pred_conf, 4),
                    "is_correct": result.is_correct,
                    "has_error": result.has_error,
                    "error_type": result.error_type,
                    "message": result.message,
                    "top3": [(int(s), round(float(c), 4)) for s, c in top3],
                })
                prev_fsm_step = result.step_id
            fsm_updated = True

        trace_entry["fsm_step"] = fsm.current_step.value
        trace_entry["fsm_step_name"] = STEP_NAMES[fsm.current_step]
        trace_entry["fsm_updated"] = fsm_updated
        trace_entry["model_pred"] = pred_step
        trace_entry["model_conf"] = round(pred_conf, 4)
        event_trace.append(trace_entry)

    # Post-sequence check (matching simulate_fsm_on_predictions)
    if fsm.current_step == SOPStep.S5_CLOSE:
        fsm.current_step = SOPStep.COMPLETE
        if prev_fsm_step != 6:
            fsm_events.append({
                "feature_idx": N - 1,
                "original_frame": int(frame_indices[-1]),
                "timestamp": round(float(timestamps[-1]), 3),
                "step": 6,
                "step_name": STEP_NAMES[SOPStep.COMPLETE],
                "model_pred": -1,
                "is_correct": True,
                "has_error": False,
                "error_type": "",
                "message": "Sequence complete (end of video at S5)",
            })

    # ── Build FSM path and error summary ──
    fsm_path = [e["step"] for e in fsm_events if e["step"] > 0]
    fsm_errors = [e for e in fsm_events if e["has_error"]]
    first_error = fsm_errors[0] if fsm_errors else None

    # Check if any None was consumed by FSM (should be 0 — we guard against it)
    # none_consumed_by_fsm_count is always 0 by construction above

    return {
        "event_trace": event_trace,
        "fsm_events": fsm_events,
        "temporal_trace": temporal_trace,
        "mature_predictions": mature_predictions,
        "fsm_path": fsm_path,
        "fsm_errors": fsm_errors,
        "first_error": first_error,
        "final_step": fsm.current_step.value,
        "fsm_complete": fsm.current_step == SOPStep.COMPLETE,
        "none_prediction_count": none_prediction_count,
        "none_consumed_by_fsm_count": none_consumed_by_fsm_count,
        "n_mature_predictions": len(mature_predictions),
        "first_mature_frame": predictor.first_mature_frame,
        "latest_mature_frame": predictor.latest_mature_frame,
    }


def process_video(video_path: Path, video_type: str, video_name: str,
                  yolo: YOLODetector, hand_detector: HandDetector,
                  extractor: FeatureExtractorV2Standalone,
                  predictor: TemporalPredictorV2
                  ) -> Dict:
    """完整增量回放: 提取特征 → 增量推理 → FSM"""

    # ── First pass: extract all features from raw video ──
    t0 = time.time()
    feat_data = extract_features_from_video(video_path, yolo, hand_detector, extractor)
    features = feat_data["features"]
    timestamps = feat_data["timestamps"]
    frame_indices = feat_data["frame_indices"]
    extract_time = time.time() - t0

    n_feat = len(features)
    if n_feat < T:
        return {
            "video_name": video_name, "video_type": video_type,
            "error": f"Too few feature frames: {n_feat} < T={T}",
            "status": "SKIP",
        }

    # ── Second pass: incremental inference + FSM ──
    t1 = time.time()
    inc_result = run_incremental_fsm(
        predictor, features, timestamps, frame_indices)
    infer_time = time.time() - t1

    # ── Determine status ──
    fsm_complete = inc_result["fsm_complete"]
    fsm_errors = inc_result["fsm_errors"]
    has_errors = len(fsm_errors) > 0

    if video_type == "ok":
        status = "PASS" if (fsm_complete and not has_errors) else "FAIL"
    elif video_type == "wr":
        status = "DETECTED" if (not fsm_complete or has_errors) else "MISS"
    else:
        status = "OK" if fsm_complete else "INCOMPLETE"

    # ── Build first error details ──
    first_error_info = None
    if inc_result["first_error"]:
        e = inc_result["first_error"]
        first_error_info = {
            "feature_idx": e["feature_idx"],
            "original_frame": e["original_frame"],
            "timestamp": e["timestamp"],
            "error_type": e["error_type"],
            "message": e["message"],
            "model_pred": e["model_pred"],
            "model_confidence": e.get("model_confidence", 0),
            "top3": e.get("top3", []),
        }

    result = {
        "video_name": video_name,
        "video_type": video_type,
        "video_info": feat_data["video_info"],
        "n_feature_frames": n_feat,
        "status": status,
        "fsm_path": inc_result["fsm_path"],
        "final_step": inc_result["final_step"],
        "fsm_complete": fsm_complete,
        "fsm_errors": [{"type": e["error_type"], "feature_idx": e["feature_idx"],
                        "frame": e["original_frame"], "msg": e["message"]}
                       for e in fsm_errors],
        "first_error": first_error_info,
        "none_prediction_count": inc_result["none_prediction_count"],
        "none_consumed_by_fsm_count": inc_result["none_consumed_by_fsm_count"],
        "n_mature_predictions": inc_result["n_mature_predictions"],
        "first_mature_frame": inc_result["first_mature_frame"],
        "latest_mature_frame": inc_result["latest_mature_frame"],
        "timing": {
            "extract_sec": round(extract_time, 2),
            "incremental_infer_fsm_sec": round(infer_time, 3),
        },
        "video_trace": feat_data["trace"],
        # Per-frame traces for saving
        "_event_trace": inc_result["event_trace"],
        "_fsm_events": inc_result["fsm_events"],
        "_temporal_trace": inc_result["temporal_trace"],
        "_mature_predictions": inc_result["mature_predictions"],
        "_predictor_state": {
            "first_mature_frame": inc_result["first_mature_frame"],
            "latest_mature_frame": inc_result["latest_mature_frame"],
            "none_prediction_count": inc_result["none_prediction_count"],
            "none_consumed_by_fsm_count": inc_result["none_consumed_by_fsm_count"],
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--video-type", default="ok",
                        choices=["ok", "wr", "yuanzi"])
    parser.add_argument("--video-name", default=None,
                        help="Single video name (e.g., ok_1)")
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: reports/realtime_replay_v2_incremental)")
    args = parser.parse_args()

    # Paths
    yolo_path = args.yolo or str(ROOT / "models" / "yolo_final_v1.pt")
    model_path = args.model or str(MODEL_DIR / "checkpoints" / "best.pt")
    imgsz = args.imgsz or (640 if args.device == "cuda" else 480)

    # Create output dirs
    out_root = Path(args.output_dir) if args.output_dir else REPORTS_DIR
    ok_dir = out_root / "ok"
    wr_dir = out_root / "wr"
    yuanzi_dir = out_root / "yuanzi"
    for d in [out_root, ok_dir, wr_dir, yuanzi_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3 (方案B): 增量流式回放测试")
    print(f"  YOLO: {yolo_path}")
    print(f"  Model: {model_path}")
    print(f"  Device: {args.device}  Video type: {args.video_type}")
    print(f"  T={T}, stride={STRIDE}")
    print("=" * 60)

    # ── Init components ──
    print("\n[Init] Loading YOLO + MediaPipe...")
    yolo = YOLODetector(yolo_path, conf_thresh=args.conf, device=args.device,
                        imgsz=imgsz)
    hand_detector = HandDetector()
    extractor = FeatureExtractorV2Standalone(yolo, hand_detector,
                                             ema_alpha=0.5, hold_frames=5)

    print("[Init] Loading TemporalPredictorV2 (方案B修复版)...")
    load_model_for_predictor(model_path)
    predictor = TemporalPredictorV2(model_path, device=args.device,
                                     T=T, stride=STRIDE)

    # ── Find videos ──
    vdir = RAW_VIDEOS_DIR / args.video_type
    if not vdir.exists():
        print(f"ERROR: Video dir not found: {vdir}")
        sys.exit(1)

    import re
    if args.video_name:
        vp = vdir / f"{args.video_name}.avi"
        videos = [(args.video_name, vp)] if vp.exists() else []
    else:
        videos = sorted(
            [(f.stem, f) for f in vdir.glob("*.avi")],
            key=lambda x: int(re.search(r'(\d+)', x[0]).group(1))
        )
    if args.max_videos:
        videos = videos[:args.max_videos]

    print(f"\n[Processing] {len(videos)} {args.video_type} videos\n")

    all_results = []
    ok_pass = 0
    ok_total = 0
    wr_detected = 0
    wr_total = 0

    for vi, (vname, vpath) in enumerate(videos):
        t0 = time.time()
        try:
            result = process_video(vpath, args.video_type, vname,
                                   yolo, hand_detector, extractor, predictor)
        except Exception as e:
            print(f"  [{vi+1}/{len(videos)}] {vname}: ERROR {e}")
            import traceback
            traceback.print_exc()
            continue

        if result.get("error"):
            print(f"  [{vi+1}/{len(videos)}] {vname}: SKIP - {result['error']}")
            continue

        elapsed = time.time() - t0

        if result["video_type"] == "ok":
            ok_total += 1
            if result["status"] == "PASS":
                ok_pass += 1
        elif result["video_type"] == "wr":
            wr_total += 1
            if result["status"] == "DETECTED":
                wr_detected += 1

        # Print per-video result
        status_icon = "✓" if result["status"] in ("PASS", "DETECTED", "OK") else "✗"
        none_info = (f"none={result['none_prediction_count']}"
                     if result['none_prediction_count'] > 0 else "")
        print(f"  [{status_icon}] [{vi+1:3d}/{len(videos)}] {vname}: "
              f"fsm_path={result['fsm_path']} "
              f"final=S{result['final_step']} "
              f"mature={result['n_mature_predictions']}/{result['n_feature_frames']} "
              f"first_mat={result['first_mature_frame']} "
              f"[{result['status']}] "
              f"{none_info} "
              f"ext={result['timing']['extract_sec']:.1f}s "
              f"inf={result['timing']['incremental_infer_fsm_sec']:.2f}s")

        if result["fsm_errors"]:
            for e in result["fsm_errors"][:2]:
                print(f"         ERR: {e['type']} @ feat={e['feature_idx']} frame={e['frame']}: {e['msg']}")

        if result["first_error"]:
            fe = result["first_error"]
            print(f"         first_error: {fe['error_type']} pred={fe['model_pred']} "
                  f"conf={fe['model_confidence']:.3f} top3={fe['top3']}")

        # ── Save per-video traces ──
        vtype = result["video_type"]
        trace_dir = {"ok": ok_dir, "wr": wr_dir, "yuanzi": yuanzi_dir}[vtype]

        # event_trace.json
        with open(trace_dir / f"{vname}_event_trace.json", "w", encoding="utf-8") as f:
            json.dump(result.pop("_event_trace"), f, indent=2, ensure_ascii=False)

        # fsm_trace.json
        fsm_trace_out = {
            "video_name": vname,
            "video_type": vtype,
            "fsm_path": result["fsm_path"],
            "fsm_complete": result["fsm_complete"],
            "final_step": result["final_step"],
            "events": result.pop("_fsm_events"),
            "errors": result["fsm_errors"],
        }
        with open(trace_dir / f"{vname}_fsm_trace.json", "w", encoding="utf-8") as f:
            json.dump(fsm_trace_out, f, indent=2, ensure_ascii=False)

        # temporal_trace.json
        with open(trace_dir / f"{vname}_temporal_trace.json", "w", encoding="utf-8") as f:
            json.dump(result.pop("_temporal_trace"), f, indent=2, ensure_ascii=False)

        # prediction_mature_frames.json
        mature_out = {
            "video_name": vname,
            "video_type": vtype,
            "n_mature": result["n_mature_predictions"],
            "predictor_state": result.pop("_predictor_state"),
            "mature_predictions": result.pop("_mature_predictions"),
        }
        with open(trace_dir / f"{vname}_prediction_mature_frames.json", "w", encoding="utf-8") as f:
            json.dump(mature_out, f, indent=2, ensure_ascii=False)

        all_results.append(result)

    if not all_results:
        print("No results to save.")
        return

    # ── Aggregate summary ──
    error_types = defaultdict(int)
    failed_videos = []
    for r in all_results:
        for err in r["fsm_errors"]:
            error_types[err["type"]] += 1
        if r["status"] in ("FAIL", "MISS"):
            failed_videos.append({
                "video_name": r["video_name"],
                "video_type": r["video_type"],
                "fsm_path": r["fsm_path"],
                "fsm_errors": r["fsm_errors"],
                "first_error": r["first_error"],
                "status": r["status"],
                "n_mature": r["n_mature_predictions"],
                "none_count": r["none_prediction_count"],
                "none_consumed": r["none_consumed_by_fsm_count"],
            })

    none_stats = {
        "videos_with_none": sum(1 for r in all_results if r["none_prediction_count"] > 0),
        "total_none_predictions": sum(r["none_prediction_count"] for r in all_results),
        "total_none_consumed_by_fsm": sum(r["none_consumed_by_fsm_count"] for r in all_results),
    }

    summary = {
        "test_date": "2026-05-20",
        "method": "方案B — 增量流式推理 (TemporalPredictorV2 修复版)",
        "model_path": model_path,
        "feature_dim": 90,
        "feature_fps": FPS,
        "window_T": T,
        "stride": STRIDE,
        "predictor_fix": {
            "window_scheduling": "_last_infer_end = start + stride",
            "fallback_behavior": "return None for uncovered frames",
            "ema": "causal, skip uncovered gaps",
        },
        "video_type": args.video_type,
        "total_videos": len(all_results),
        "results": {
            "ok_total": ok_total,
            "ok_pass": ok_pass,
            "ok_pass_rate": round(ok_pass / max(ok_total, 1), 4),
            "wr_total": wr_total,
            "wr_detected": wr_detected,
            "wr_detection_rate": round(wr_detected / max(wr_total, 1), 4),
        },
        "error_type_distribution": dict(error_types),
        "none_prediction_stats": none_stats,
        "failed_videos": failed_videos,
        "per_video_summary": [
            {
                "video_name": r["video_name"],
                "video_type": r["video_type"],
                "status": r["status"],
                "fsm_path": r["fsm_path"],
                "final_step": r["final_step"],
                "n_feature_frames": r["n_feature_frames"],
                "n_mature": r["n_mature_predictions"],
                "first_mature_frame": r["first_mature_frame"],
                "none_count": r["none_prediction_count"],
                "none_consumed": r["none_consumed_by_fsm_count"],
                "n_fsm_errors": len(r["fsm_errors"]),
            }
            for r in all_results
        ],
    }

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print summary ──
    print(f"\n{'=' * 60}")
    print(f"INCREMENTAL REPLAY SUMMARY ({args.video_type}) — 方案B")
    print(f"{'=' * 60}")
    if ok_total > 0:
        print(f"  OK: {ok_pass}/{ok_total} pass ({ok_pass/max(ok_total,1):.1%})")
    if wr_total > 0:
        print(f"  WR: {wr_detected}/{wr_total} detected ({wr_detected/max(wr_total,1):.1%})")
    print(f"  Failed videos: {len(failed_videos)}")
    print(f"  Error types: {dict(error_types)}")
    print(f"  None predictions: {none_stats['total_none_predictions']} total, "
          f"{none_stats['total_none_consumed_by_fsm']} consumed by FSM (must be 0)")
    if failed_videos:
        print(f"\n  FAILED VIDEOS:")
        for fv in failed_videos:
            print(f"    {fv['video_name']}: path={fv['fsm_path']} "
                  f"errors={[e['type'] for e in fv['fsm_errors']]}")
            if fv["first_error"]:
                fe = fv["first_error"]
                print(f"      first_error: {fe['error_type']} pred={fe['model_pred']} "
                      f"conf={fe['model_confidence']:.3f} top3={fe['top3']}")
    print(f"\n  Reports: {out_root}/")

    # ── Acceptance check ──
    ok_rate = ok_pass / max(ok_total, 1)
    wr_rate = wr_detected / max(wr_total, 1)
    none_ok = none_stats["total_none_consumed_by_fsm"] == 0

    print(f"\n  ACCEPTANCE CHECK:")
    print(f"    OK ≥ 95%:  {'✓' if ok_rate >= 0.95 else '✗'} ({ok_rate:.1%})")
    print(f"    WR ≥ 95%:  {'✓' if wr_rate >= 0.95 else '✗'} ({wr_rate:.1%})")
    print(f"    None consumed = 0: {'✓' if none_ok else '✗'} ({none_stats['total_none_consumed_by_fsm']})")

    if ok_rate >= 0.95 and wr_rate >= 0.95 and none_ok:
        print(f"\n  ✓ ACCEPTANCE PASSED — 增量实时推理链路可用")
        print(f"  → 下一步: 跑全量 181 OK + 64 WR")
    else:
        print(f"\n  ⚠ ACCEPTANCE NOT MET — 需检查失败视频 trace")


if __name__ == "__main__":
    main()
