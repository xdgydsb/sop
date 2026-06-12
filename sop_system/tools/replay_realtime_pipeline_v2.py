"""
Phase 3 (方案A): 原始视频回放测试 — 批量推理诊断版

从原始 .avi 逐帧跑完整管线:
  第一遍: 原始帧 → YOLO → MediaPipe → FeatureExtractorV2(90d) → 收集 features[N,90]
  第二遍: features → sliding_window_inference() → preds, confs (与离线eval完全相同)
  第三遍: preds, confs → simulate_fsm_on_predictions() → FSM路径 (与离线eval完全相同)

对比 (如果离线 .npz 存在):
  - replay features vs saved features (mean_abs_diff, max_abs_diff, cosine_sim)
  - replay batch preds vs saved batch preds (frame_agreement, sequence_agreement)
  - FSM path comparison

关键: 方案A 是诊断方案 —
  - 如果 replay features 批量推理接近 98.9% → 特征提取正确，问题在增量推理调度
  - 如果 replay features 批量推理也差 → 特征提取环节有问题
  - 不是最终实时方案

Usage:
  python tools/replay_realtime_pipeline_v2.py --video-type ok --max-videos 20
  python tools/replay_realtime_pipeline_v2.py --video-type wr --max-videos 20
  python tools/replay_realtime_pipeline_v2.py --video-type yuanzi --max-videos 5
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
from engine.sop_fsm import SOPStateMachine, SOPStep
from tools.extract_features_v2_90 import FeatureExtractorV2Standalone, INTERACT_OBJS
from eval_pipeline_offline_v2 import sliding_window_inference, simulate_fsm_on_predictions
from train_temporal_v2 import TCNBiGRU

# ── Paths ──
RAW_VIDEOS_DIR = ROOT.parent / "data"
FEAT_DIR = ROOT / "data" / "features" / "v2_90"
MODEL_DIR = ROOT / "models" / "temporal" / "v2_90_tcn_bigru"
REPORTS_DIR = ROOT / "reports" / "realtime_replay_v2"
REPLAY_FEAT_DIR = REPORTS_DIR / "features"
OK_TRACES_DIR = REPORTS_DIR / "ok"
WR_TRACES_DIR = REPORTS_DIR / "wr"
YUANZI_TRACES_DIR = REPORTS_DIR / "yuanzi"

FPS = 15.0
T = 48
STRIDE = 4


def load_model(model_path: str, device: str):
    """Load TCNBiGRU model from checkpoint"""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    assert ckpt.get("input_dim") == 90, (
        f"WRONG MODEL: input_dim={ckpt.get('input_dim')}, expected 90"
    )
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
    print(f"[Model] Loaded: input_dim={ckpt['input_dim']}, "
          f"hidden_dim={ckpt['hidden_dim']}, params={ckpt.get('total_params', '?')}")
    return model


def compare_features(feat_a: np.ndarray, feat_b: np.ndarray) -> Dict:
    """Compare two feature matrices, return difference metrics"""
    min_len = min(len(feat_a), len(feat_b))
    if min_len == 0:
        return {"error": "empty features", "comparable": False}

    a = feat_a[:min_len]
    b = feat_b[:min_len]

    abs_diff = np.abs(a - b)
    mean_abs = float(np.mean(abs_diff))
    max_abs = float(np.max(abs_diff))

    # Cosine similarity per frame, then average
    cos_sims = []
    for i in range(min_len):
        na = np.linalg.norm(a[i])
        nb = np.linalg.norm(b[i])
        if na > 1e-9 and nb > 1e-9:
            cos_sims.append(float(np.dot(a[i], b[i]) / (na * nb)))
    mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

    # Per-dimension mean abs diff
    dim_diff = np.mean(abs_diff, axis=0)
    worst_dims = np.argsort(-dim_diff)[:5].tolist()
    worst_dim_vals = [float(dim_diff[d]) for d in worst_dims]

    return {
        "comparable": True,
        "min_len": min_len,
        "len_a": len(feat_a),
        "len_b": len(feat_b),
        "mean_abs_diff": round(mean_abs, 6),
        "max_abs_diff": round(max_abs, 6),
        "mean_cosine_sim": round(mean_cos, 6),
        "worst_5_dims": worst_dims,
        "worst_5_dim_diff": [round(v, 6) for v in worst_dim_vals],
    }


def compare_predictions(preds_a: np.ndarray, preds_b: np.ndarray) -> Dict:
    """Compare two prediction arrays"""
    min_len = min(len(preds_a), len(preds_b))
    if min_len == 0:
        return {"error": "empty predictions", "comparable": False}

    a = preds_a[:min_len]
    b = preds_b[:min_len]

    frame_agree = float(np.mean(a == b))
    diff_frames = int(np.sum(a != b))
    total_frames = min_len

    # Sequence agreement: find transitions (where pred changes)
    def transitions(p):
        return [(i, int(p[i - 1]), int(p[i]))
                for i in range(1, len(p)) if p[i] != p[i - 1]]

    trans_a = transitions(a)
    trans_b = transitions(b)

    return {
        "comparable": True,
        "min_len": min_len,
        "len_a": len(preds_a),
        "len_b": len(preds_b),
        "frame_agreement": round(frame_agree, 4),
        "diff_frames": diff_frames,
        "total_frames": total_frames,
        "transitions_a": trans_a[:20],
        "transitions_b": trans_b[:20],
    }


def extract_features_from_video(video_path: Path,
                                 yolo: YOLODetector,
                                 hand_detector: HandDetector,
                                 extractor: FeatureExtractorV2Standalone,
                                 target_fps: float = FPS
                                 ) -> Dict:
    """第一遍: 遍历原始视频，收集全部90维特征

    Returns dict with:
        features: np.ndarray [N, 90]
        timestamps: np.ndarray [N]
        frame_indices: np.ndarray [N]
        video_info: dict
        trace: list (per-frame YOLO/MediaPipe state for diagnostics)
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

        # Update tracked object states (in-box, hand touch)
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


def process_video(video_path: Path, video_type: str, video_name: str,
                  yolo: YOLODetector, hand_detector: HandDetector,
                  extractor: FeatureExtractorV2Standalone,
                  model, device: str,
                  segments_df=None
                  ) -> Dict:
    """完整诊断流程: 提取特征 → 批量推理 → FSM → 对比

    Returns dict with 'feat_data' key containing the extracted features,
    so caller can save/compare without re-extracting.
    """

    # ── First pass: extract all features from raw video ──
    t0 = time.time()
    feat_data = extract_features_from_video(video_path, yolo, hand_detector, extractor)
    features = feat_data["features"]
    extract_time = time.time() - t0

    n_feat = len(features)
    if n_feat < T:
        return {
            "video_name": video_name, "video_type": video_type,
            "error": f"Too few feature frames: {n_feat} < T={T}",
            "status": "SKIP",
        }

    # ── Second pass: batch sliding window inference (SAME as offline eval) ──
    t1 = time.time()
    preds, confs, probs = sliding_window_inference(
        features, model, device, T=T, stride=STRIDE)
    infer_time = time.time() - t1

    # ── Third pass: FSM (SAME as offline eval) ──
    t2 = time.time()
    # Get ground truth steps from segments if available
    gt_steps = []
    if segments_df is not None:
        vid_segs = segments_df[segments_df["video_name"] == video_name]
        gt_steps = sorted(vid_segs["step_id"].unique().tolist())

    fsm_result = simulate_fsm_on_predictions(preds, confs, gt_steps)
    fsm_time = time.time() - t2

    # ── Determine status ──
    fsm_complete = fsm_result["fsm_complete"]
    fsm_errors = fsm_result["fsm_errors"]
    has_errors = len(fsm_errors) > 0

    if video_type == "ok":
        status = "PASS" if (fsm_complete and not has_errors) else "FAIL"
    elif video_type == "wr":
        status = "DETECTED" if (not fsm_complete or has_errors) else "MISS"
    else:
        status = "OK" if fsm_complete else "INCOMPLETE"

    # ── Build result ──
    result = {
        "video_name": video_name,
        "video_type": video_type,
        "video_info": feat_data["video_info"],
        "n_feature_frames": n_feat,
        "status": status,
        "fsm_path": fsm_result["fsm_path"],
        "final_step": fsm_result["final_step"],
        "fsm_complete": fsm_complete,
        "fsm_errors": [{"type": e["error_type"], "frame": e["frame"], "msg": e["message"]}
                       for e in fsm_errors],
        "steps_seen": fsm_result["steps_seen"],
        "gt_steps": gt_steps,
        "timing": {
            "extract_sec": round(extract_time, 2),
            "infer_sec": round(infer_time, 3),
            "fsm_sec": round(fsm_time, 4),
        },
        "model_preds": [int(p) for p in preds],
        "model_confs": [round(float(c), 3) for c in confs],
        "trace": feat_data["trace"],
        "fsm_events": fsm_result["events"],
        "feat_data": feat_data,  # returned so caller can save without re-extracting
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
    parser.add_argument("--feat-dir", default=None,
                        help="Path to saved .npz features for comparison "
                             "(default: data/features/v2_90 under ROOT)")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip comparison with saved features")
    args = parser.parse_args()

    # Paths
    yolo_path = args.yolo or str(ROOT / "models" / "yolo_final_v1.pt")
    model_path = args.model or str(MODEL_DIR / "checkpoints" / "best.pt")
    feat_dir = Path(args.feat_dir) if args.feat_dir else FEAT_DIR
    imgsz = args.imgsz or (640 if args.device == "cuda" else 480)

    # Create output dirs
    for d in [REPORTS_DIR, REPLAY_FEAT_DIR, OK_TRACES_DIR, WR_TRACES_DIR, YUANZI_TRACES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3 (方案A): 批量推理诊断回放")
    print(f"  YOLO: {yolo_path}")
    print(f"  Model: {model_path}")
    print(f"  Device: {args.device}  Video type: {args.video_type}")
    print("=" * 60)

    # ── Init components ──
    print("\n[Init] Loading YOLO + MediaPipe...")
    yolo = YOLODetector(yolo_path, conf_thresh=args.conf, device=args.device,
                        imgsz=imgsz)
    hand_detector = HandDetector()
    extractor = FeatureExtractorV2Standalone(yolo, hand_detector,
                                             ema_alpha=0.5, hold_frames=5)

    print("[Init] Loading temporal model...")
    model = load_model(model_path, args.device)

    # Load segment annotations for GT comparison
    segments_df = None
    segment_csv = ROOT / "data" / "annotations" / "segment_annotations_v2.csv"
    if segment_csv.exists():
        import pandas as pd
        segments_df = pd.read_csv(segment_csv)
        print(f"[Init] Loaded {len(segments_df)} segment annotations")

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
    feature_comparisons = []
    prediction_comparisons = []
    ok_pass = 0
    wr_detected = 0
    total_ok = 0
    total_wr = 0

    for vi, (vname, vpath) in enumerate(videos):
        t0 = time.time()
        try:
            result = process_video(vpath, args.video_type, vname,
                                   yolo, hand_detector, extractor,
                                   model, args.device, segments_df)
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
            total_ok += 1
            if result["status"] == "PASS":
                ok_pass += 1
        elif result["video_type"] == "wr":
            total_wr += 1
            if result["status"] == "DETECTED":
                wr_detected += 1

        status_icon = "PASS" if result["status"] in ("PASS", "DETECTED", "OK") else "FAIL"
        print(f"  [{status_icon}] [{vi+1:3d}/{len(videos)}] {vname}: "
              f"{result['n_feature_frames']}f "
              f"fsm_path={result['fsm_path']} "
              f"errors={len(result['fsm_errors'])} "
              f"final=S{result['final_step']} "
              f"[{result['status']}] "
              f"ext={result['timing']['extract_sec']:.1f}s "
              f"inf={result['timing']['infer_sec']:.2f}s")

        # ── Save replay features (from the same extraction, no re-extract) ──
        feat_data = result.pop("feat_data")
        feat_out = REPLAY_FEAT_DIR / f"{vname}.npz"
        np.savez_compressed(
            feat_out,
            features=feat_data["features"],
            timestamps=feat_data["timestamps"],
            frame_indices=feat_data["frame_indices"],
            video_name=vname,
            video_type=args.video_type,
        )

        # ── Compare with saved features ──
        saved_feat_path = feat_dir / args.video_type / f"{vname}.npz"
        if not args.no_compare and saved_feat_path.exists():
            saved = np.load(saved_feat_path, allow_pickle=True)
            if "features" in saved:
                saved_feats = saved["features"]

                # Feature comparison
                feat_cmp = compare_features(feat_data["features"], saved_feats)
                feat_cmp["video_name"] = vname
                feature_comparisons.append(feat_cmp)

                # Prediction comparison: batch inference on replay features vs saved features
                r_preds, r_confs, _ = sliding_window_inference(
                    feat_data["features"], model, args.device, T=T, stride=STRIDE)
                s_preds, s_confs, _ = sliding_window_inference(
                    saved_feats, model, args.device, T=T, stride=STRIDE)
                pred_cmp = compare_predictions(r_preds, s_preds)
                pred_cmp["video_name"] = vname

                # FSM comparison
                r_fsm = simulate_fsm_on_predictions(r_preds, r_confs, [])
                s_fsm = simulate_fsm_on_predictions(s_preds, s_confs, [])
                pred_cmp["fsm_path_replay"] = r_fsm["fsm_path"]
                pred_cmp["fsm_path_saved"] = s_fsm["fsm_path"]
                pred_cmp["fsm_match"] = r_fsm["fsm_path"] == s_fsm["fsm_path"]
                prediction_comparisons.append(pred_cmp)

                cos = feat_cmp.get("mean_cosine_sim", 0)
                agree = pred_cmp["frame_agreement"]
                fsm_match = "✓" if pred_cmp["fsm_match"] else "✗"
                print(f"         feat_cos={cos:.4f}  pred_agree={agree:.3f}  fsm_match={fsm_match}")

        # Save per-video trace
        vtype = result["video_type"]
        out_dir = {"ok": OK_TRACES_DIR, "wr": WR_TRACES_DIR, "yuanzi": YUANZI_TRACES_DIR}[vtype]
        trace_path = out_dir / f"{vname}_trace.json"
        trace_data = {k: v for k, v in result.items() if k not in ("trace", "model_preds", "model_confs", "fsm_events")}
        trace_data["model_preds"] = result["model_preds"]
        trace_data["model_confs"] = result["model_confs"]
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

        all_results.append(result)

    if not all_results:
        print("No results to save.")
        return

    # ── Save comparison reports ──
    if feature_comparisons:
        feat_summary = {
            "n_compared": len(feature_comparisons),
            "mean_cosine_sim": round(np.mean([c["mean_cosine_sim"] for c in feature_comparisons]), 6),
            "mean_abs_diff": round(np.mean([c["mean_abs_diff"] for c in feature_comparisons]), 6),
            "max_abs_diff_max": round(max(c["max_abs_diff"] for c in feature_comparisons), 6),
            "per_video": feature_comparisons,
        }
        with open(REPORTS_DIR / "feature_compare.json", "w", encoding="utf-8") as f:
            json.dump(feat_summary, f, indent=2, ensure_ascii=False)
        print(f"\nFeature comparison: {len(feature_comparisons)} videos, "
              f"mean_cos={feat_summary['mean_cosine_sim']:.4f}")

    if prediction_comparisons:
        pred_summary = {
            "n_compared": len(prediction_comparisons),
            "mean_frame_agreement": round(np.mean([c["frame_agreement"] for c in prediction_comparisons]), 4),
            "fsm_match_rate": round(np.mean([1.0 if c["fsm_match"] else 0.0 for c in prediction_comparisons]), 4),
            "per_video": prediction_comparisons,
        }
        with open(REPORTS_DIR / "pred_compare.json", "w", encoding="utf-8") as f:
            json.dump(pred_summary, f, indent=2, ensure_ascii=False)
        print(f"Pred comparison: {len(prediction_comparisons)} videos, "
              f"frame_agree={pred_summary['mean_frame_agreement']:.3f}, "
              f"fsm_match={pred_summary['fsm_match_rate']:.2%}")

    # ── Aggregate summary ──
    error_types = defaultdict(int)
    for r in all_results:
        for err in r["fsm_errors"]:
            error_types[err["type"]] += 1

    summary = {
        "replay_date": "2026-05-20",
        "method": "方案A — 批量推理诊断",
        "video_type": args.video_type,
        "total_videos": len(all_results),
        "model_path": model_path,
        "feature_dim": 90,
        "feature_fps": FPS,
        "window_T": T,
        "stride": STRIDE,
        "results": {
            "ok_total": total_ok,
            "ok_pass": ok_pass,
            "ok_pass_rate": round(ok_pass / max(total_ok, 1), 4),
            "wr_total": total_wr,
            "wr_detected": wr_detected,
            "wr_detection_rate": round(wr_detected / max(total_wr, 1), 4),
        },
        "error_type_distribution": dict(error_types),
        "bad_cases": [
            {
                "video_name": r["video_name"],
                "video_type": r["video_type"],
                "fsm_path": r["fsm_path"],
                "fsm_errors": r["fsm_errors"],
                "status": r["status"],
            }
            for r in all_results if r["status"] in ("FAIL", "MISS")
        ],
        "feature_compare_summary": (
            {"n_compared": len(feature_comparisons)}
            if feature_comparisons else None
        ),
        "prediction_compare_summary": (
            {"n_compared": len(prediction_comparisons)}
            if prediction_comparisons else None
        ),
    }

    with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print summary ──
    print(f"\n{'=' * 60}")
    print(f"REPLAY SUMMARY ({args.video_type}) — 方案A 批量推理")
    print(f"{'=' * 60}")
    if total_ok > 0:
        print(f"  OK: {ok_pass}/{total_ok} pass ({ok_pass/max(total_ok,1):.1%})")
    if total_wr > 0:
        print(f"  WR: {wr_detected}/{total_wr} detected ({wr_detected/max(total_wr,1):.1%})")
    print(f"  Bad cases: {len(summary['bad_cases'])}")
    print(f"  Error types: {dict(error_types)}")
    print(f"\n  Reports: {REPORTS_DIR}/")
    print(f"  Replay features: {REPLAY_FEAT_DIR}/")
    print(f"  Traces: {OK_TRACES_DIR}/, {WR_TRACES_DIR}/")

    # Diagnostic conclusion
    if total_ok > 0:
        pass_rate = ok_pass / max(total_ok, 1)
        if pass_rate >= 0.90:
            print(f"\n  DIAGNOSIS: Replay batch inference matches offline (≥90%).")
            print(f"  → Problem confirmed in TemporalPredictorV2 incremental scheduling.")
            print(f"  → Proceed to 方案B: fix incremental predictor.")
        else:
            print(f"\n  DIAGNOSIS: Replay batch inference also below 90% ({pass_rate:.1%}).")
            print(f"  → Feature extraction from raw video may differ from offline .npz.")
            print(f"  → Check feature_compare.json for per-dimension differences.")


if __name__ == "__main__":
    main()
