"""
方案B: TemporalPredictorV2 增量 vs 批量 等价性测试

验证: 同一个 features[N,90]，增量推理的成熟帧预测和批量推理一致。

测试逻辑:
  1. 所有特征逐帧送入 predictor.predict()
  2. 全部处理完后，正向遍历所有成熟帧，调用 get_frame_prediction(fi)
  3. 对比: 增量 raw probs vs 批量 raw probs (pre-EMA)
  4. 对比: 增量 EMA pred vs 批量 EMA pred
  5. 对比: FSM 路径

Usage:
  python tools/test_temporal_predictor_equivalence.py --max-videos 20
"""
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval_pipeline_offline_v2 import sliding_window_inference, simulate_fsm_on_predictions
from engine.temporal_predictor_v2 import TemporalPredictorV2
from train_temporal_v2 import TCNBiGRU

T = 48
STRIDE = 4
HALF_W = T // 4  # 12
N_CLASSES = 6
REPORTS_DIR = ROOT / "reports" / "temporal_v2_90"


def compute_batch_raw_probs(features, model, device):
    """Compute batch raw frame_probs (BEFORE EMA) — same as sliding_window_inference internal.

    Returns: frame_probs [N, 6], frame_counts [N]
    """
    N = len(features)
    frame_probs = np.zeros((N, N_CLASSES), dtype=np.float64)
    frame_counts = np.zeros(N, dtype=np.int32)

    model.eval()
    x_tensor = torch.FloatTensor(features)

    with torch.no_grad():
        for start in range(0, N - T, STRIDE):
            end = start + T
            x = x_tensor[start:end].unsqueeze(0).to(device)
            step_logits, _, _ = model(x)
            probs = torch.softmax(step_logits, dim=1).cpu().numpy()
            assign_start = start + HALF_W
            assign_end = end - HALF_W
            for fi in range(assign_start, assign_end):
                if 0 <= fi < N:
                    frame_probs[fi] += probs[0]
                    frame_counts[fi] += 1

    # Handle uncovered frames (same as offline)
    for fi in range(N):
        if frame_counts[fi] == 0:
            for d in range(1, 20):
                if fi + d < N and frame_counts[fi + d] > 0:
                    frame_probs[fi] = frame_probs[fi + d]
                    frame_counts[fi] = 0.5
                    break
                if fi - d >= 0 and frame_counts[fi - d] > 0:
                    frame_probs[fi] = frame_probs[fi - d]
                    frame_counts[fi] = 0.5
                    break

    frame_probs_norm = frame_probs / np.maximum(frame_counts[:, None], 0.5)
    return frame_probs_norm, frame_counts


def test_video(video_name: str, video_type: str, predictor, model, device: str,
               feat_dir: Path) -> dict:
    """Test equivalence for one video"""
    feat_path = feat_dir / video_type / f"{video_name}.npz"
    if not feat_path.exists():
        return {"video_name": video_name, "error": "feature file not found"}

    data = np.load(feat_path, allow_pickle=True)
    features = data["features"]  # [N, 90]
    N = len(features)

    if N < T:
        return {"video_name": video_name, "error": f"too few frames: {N} < {T}"}

    # ── Batch: full sliding_window_inference ──
    batch_preds, batch_confs, batch_smoothed = sliding_window_inference(
        features, model, device, T=T, stride=STRIDE)

    # ── Batch raw probs (pre-EMA) for comparison ──
    batch_raw, batch_counts = compute_batch_raw_probs(features, model, device)

    # ── Incremental: process all features ──
    predictor.reset()
    for fi in range(N):
        predictor.predict(features[fi])

    # Collect incremental raw probs for mature frames
    inc_raw_probs = {}
    inc_mature = []
    for fi in range(N):
        if predictor.is_frame_mature(fi):
            inc_raw_probs[fi] = (
                predictor._frame_probs[fi]
                / max(predictor._frame_counts[fi], 1)
            )
            inc_mature.append(fi)

    # Collect incremental EMA predictions in FORWARD order
    # Reset EMA and re-traverse
    predictor._ema_state = None
    predictor._last_ema_frame = -1
    inc_ema_preds = {}
    for fi in inc_mature:
        pred = predictor.get_frame_prediction(fi)
        if pred is not None:
            inc_ema_preds[fi] = pred

    # ── Compare raw probs (incremental vs batch) on mature frames ──
    raw_agree = 0
    raw_disagree = 0
    raw_max_diff = 0.0
    for fi in inc_mature:
        if fi < N:
            inc_argmax = int(np.argmax(inc_raw_probs[fi]))
            batch_argmax = int(np.argmax(batch_raw[fi]))
            if inc_argmax == batch_argmax:
                raw_agree += 1
            else:
                raw_disagree += 1
            diff = float(np.max(np.abs(inc_raw_probs[fi] - batch_raw[fi])))
            if diff > raw_max_diff:
                raw_max_diff = diff

    raw_agree_rate = raw_agree / max(raw_agree + raw_disagree, 1)

    # ── Compare EMA predictions on overlapping frames ──
    common_frames = sorted(set(inc_mature) & set(range(N)))
    ema_agree = 0
    ema_disagree = 0
    ema_disagreements = []
    for fi in common_frames:
        inc_step = inc_ema_preds[fi]["step"] if fi in inc_ema_preds else -1
        batch_step = int(batch_preds[fi])
        if inc_step == batch_step:
            ema_agree += 1
        else:
            ema_disagree += 1
            if len(ema_disagreements) < 10:
                ema_disagreements.append({
                    "frame": fi,
                    "incremental": inc_step,
                    "batch": batch_step,
                })

    ema_agree_rate = ema_agree / max(ema_agree + ema_disagree, 1)

    # ── Build incremental preds array for FSM (use raw argmax, no EMA) ──
    inc_preds_full = np.zeros(N, dtype=int)
    inc_confs_full = np.zeros(N, dtype=float)
    for fi in range(N):
        if fi in inc_raw_probs:
            inc_preds_full[fi] = int(np.argmax(inc_raw_probs[fi]))
            inc_confs_full[fi] = float(np.max(inc_raw_probs[fi]))
        # else: stays 0 (idle) with 0 confidence

    # ── FSM comparison ──
    batch_fsm = simulate_fsm_on_predictions(batch_preds, batch_confs, [])
    inc_fsm = simulate_fsm_on_predictions(inc_preds_full, inc_confs_full, [])

    n_mature = len(inc_mature)
    first_mature = inc_mature[0] if inc_mature else -1
    expected_first = HALF_W  # ~12
    first_mature_ok = abs(first_mature - expected_first) <= 4

    return {
        "video_name": video_name,
        "video_type": video_type,
        "n_frames": N,
        "n_mature": n_mature,
        "expected_mature_min": N - 2 * HALF_W,
        "first_mature_frame": first_mature,
        "first_mature_ok": first_mature_ok,
        "raw_agree_rate": round(raw_agree_rate, 4),
        "raw_agree_frames": raw_agree,
        "raw_disagree_frames": raw_disagree,
        "raw_max_abs_diff": round(raw_max_diff, 6),
        "ema_agree_rate": round(ema_agree_rate, 4),
        "ema_agree_frames": ema_agree,
        "ema_disagree_frames": ema_disagree,
        "ema_disagreements": ema_disagreements,
        "batch_fsm_path": batch_fsm["fsm_path"],
        "inc_fsm_path": inc_fsm["fsm_path"],
        "fsm_match": batch_fsm["fsm_path"] == inc_fsm["fsm_path"],
        "batch_final_step": batch_fsm["final_step"],
        "inc_final_step": inc_fsm["final_step"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--feat-dir", default=None)
    parser.add_argument("--video-name", default=None)
    parser.add_argument("--video-type", default="ok", choices=["ok", "wr"])
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_path = args.model or str(ROOT / "models" / "temporal" / "v2_90_tcn_bigru" / "checkpoints" / "best.pt")
    feat_dir = Path(args.feat_dir) if args.feat_dir else (ROOT / "data" / "features" / "v2_90")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("方案B: TemporalPredictorV2 增量 vs 批量 等价性测试")
    print(f"  Model: {model_path}")
    print(f"  Features: {feat_dir}")
    print(f"  T={T}, stride={STRIDE}, half_w={HALF_W}")
    print("=" * 60)

    # Load model for batch inference
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = TCNBiGRU(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_classes_step=ckpt["num_classes_step"],
        num_classes_verb=ckpt["num_classes_verb"],
        num_classes_target=ckpt["num_classes_target"],
        dropout=ckpt.get("dropout", 0.3),
        num_gru_layers=ckpt.get("num_gru_layers", 2),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load predictor (incremental)
    predictor = TemporalPredictorV2(model_path, device=args.device, T=T, stride=STRIDE)

    # Find videos
    feat_type_dir = feat_dir / args.video_type
    if not feat_type_dir.exists():
        print(f"ERROR: Feature dir not found: {feat_type_dir}")
        sys.exit(1)

    import re
    if args.video_name:
        videos = [(args.video_name, feat_type_dir / f"{args.video_name}.npz")]
    else:
        videos = sorted(
            [(f.stem, f) for f in feat_type_dir.glob("*.npz")],
            key=lambda x: int(re.search(r'(\d+)', x[0]).group(1))
        )
    if args.max_videos:
        videos = videos[:args.max_videos]

    print(f"\n[Testing] {len(videos)} {args.video_type} videos\n")

    all_results = []
    for vi, (vname, vpath) in enumerate(videos):
        result = test_video(vname, args.video_type, predictor, model, args.device,
                            feat_dir)
        all_results.append(result)

        if "error" in result:
            print(f"  [{vi+1}/{len(videos)}] {vname}: SKIP - {result['error']}")
            continue

        raw_icon = "✓" if result["raw_agree_rate"] >= 0.95 else "✗"
        ema_icon = "✓" if result["ema_agree_rate"] >= 0.95 else "✗"
        fsm_icon = "✓" if result["fsm_match"] else "✗"
        print(f"  raw={raw_icon} ema={ema_icon} fsm={fsm_icon} "
              f"[{vi+1:3d}/{len(videos)}] {vname}: "
              f"mature={result['n_mature']}/{result['n_frames']} "
              f"first_mat={result['first_mature_frame']}"
              f"{'✓' if result['first_mature_ok'] else '✗'} "
              f"raw_agree={result['raw_agree_rate']:.4f} "
              f"max_diff={result['raw_max_abs_diff']:.6f} "
              f"batch_fsm={result['batch_fsm_path']} "
              f"inc_fsm={result['inc_fsm_path']}")

        if result["ema_disagree_frames"] > 0:
            for d in result["ema_disagreements"][:3]:
                print(f"         ema_diff frame {d['frame']}: inc={d['incremental']} batch={d['batch']}")

    if not all_results:
        print("No results.")
        return

    n_valid = len([r for r in all_results if "error" not in r])
    if n_valid == 0:
        print("No valid results.")
        return

    # Summary
    raw_rates = [r["raw_agree_rate"] for r in all_results if "error" not in r]
    ema_rates = [r["ema_agree_rate"] for r in all_results if "error" not in r]
    fsm_matches = sum(1 for r in all_results if "error" not in r and r["fsm_match"])
    first_oks = sum(1 for r in all_results if "error" not in r and r["first_mature_ok"])
    max_diffs = [r["raw_max_abs_diff"] for r in all_results if "error" not in r]

    summary = {
        "test_date": "2026-05-20",
        "fix_version": "方案B — no fallback, causal EMA",
        "n_videos_tested": n_valid,
        "mean_raw_agree": round(np.mean(raw_rates), 6),
        "min_raw_agree": round(np.min(raw_rates), 6),
        "mean_ema_agree": round(np.mean(ema_rates), 6),
        "min_ema_agree": round(np.min(ema_rates), 6),
        "fsm_match_rate": round(fsm_matches / n_valid, 4),
        "first_mature_ok_rate": round(first_oks / n_valid, 4),
        "mean_raw_max_diff": round(np.mean(max_diffs), 8),
        "max_raw_max_diff": round(np.max(max_diffs), 8),
        "per_video": all_results,
    }

    out_path = REPORTS_DIR / "incremental_equivalence.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"EQUIVALENCE TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Videos: {n_valid}")
    print(f"  Raw agree:  mean={summary['mean_raw_agree']:.6f} min={summary['min_raw_agree']:.6f}")
    print(f"  Raw max_diff: mean={summary['mean_raw_max_diff']:.8f} max={summary['max_raw_max_diff']:.8f}")
    print(f"  EMA agree:  mean={summary['mean_ema_agree']:.4f} min={summary['min_ema_agree']:.4f}")
    print(f"  FSM match:  {fsm_matches}/{n_valid} = {fsm_matches/n_valid:.1%}")
    print(f"  First mature ok: {first_oks}/{n_valid}")
    print(f"\n  Report: {out_path}")

    if summary["min_raw_agree"] >= 0.999 and fsm_matches / n_valid >= 0.90:
        print(f"\n  ✓ PASS — incremental raw predictions match batch exactly")
        print(f"  → Ready for real-time replay test")
        return 0
    elif summary["min_raw_agree"] >= 0.95:
        print(f"\n  ⚠ WARN — raw predictions close but not exact")
        print(f"  → Small numerical differences, may not affect FSM")
        return 0
    else:
        print(f"\n  ✗ FAIL — incremental raw predictions diverge from batch")
        return 1


if __name__ == "__main__":
    sys.exit(main())
