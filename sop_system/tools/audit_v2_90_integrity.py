"""
v2_90 可信性审查脚本
检查: 数据泄漏、评估独立性、模型/特征一致性
输出: reports/temporal_v2_90/audit_integrity_v2.json
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FEAT_DIR = ROOT / "data" / "features" / "v2_90"
LABEL_DIR = ROOT / "data" / "labels" / "frame_labels_v2"
DATASET_DIR = ROOT / "data" / "datasets" / "temporal_v2_90_T48_S8"
MODEL_DIR = ROOT / "models" / "temporal" / "v2_90_tcn_bigru"
REPORTS_DIR = ROOT / "reports" / "temporal_v2_90"
SEGMENT_CSV = ROOT / "data" / "annotations" / "segment_annotations_v2.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def audit_split_integrity():
    """Check 1: train/val/test split by video_name, no leakage"""
    results = {"check": "split_integrity", "passed": True, "details": {}}

    split_videos = {}
    for split_name in ["train", "val", "test"]:
        index_path = DATASET_DIR / f"{split_name}_index.json"
        if not index_path.exists():
            results["passed"] = False
            results["details"][split_name] = "MISSING index file"
            continue

        with open(index_path) as f:
            data = json.load(f)
        windows = data.get("windows", data) if isinstance(data, dict) else data

        videos = set()
        ok_cnt = wr_cnt = 0
        for item in windows:
            vtype = item["video_type"]
            vname = item["video_name"]
            videos.add(f"{vtype}/{vname}")
            if vtype == "ok":
                ok_cnt += 1
            else:
                wr_cnt += 1

        split_videos[split_name] = videos
        results["details"][split_name] = {
            "n_windows": len(windows),
            "n_videos": len(videos),
            "ok_windows": ok_cnt,
            "wr_windows": wr_cnt,
        }

    # Cross-check for leakage
    train_v = split_videos.get("train", set())
    val_v = split_videos.get("val", set())
    test_v = split_videos.get("test", set())

    train_val_leak = train_v & val_v
    train_test_leak = train_v & test_v
    val_test_leak = val_v & test_v

    results["details"]["cross_split_leakage"] = {
        "train_val": len(train_val_leak),
        "train_test": len(train_test_leak),
        "val_test": len(val_test_leak),
    }

    if train_val_leak or train_test_leak or val_test_leak:
        results["passed"] = False
        results["details"]["leaked_videos"] = list(
            train_val_leak | train_test_leak | val_test_leak)

    total_unique = len(train_v | val_v | test_v)
    results["details"]["total_unique_videos"] = total_unique
    results["details"]["expected"] = 244
    if total_unique < 240:
        results["passed"] = False

    return results


def audit_feature_label_independence():
    """Check 2: Features extracted without label leakage"""
    results = {"check": "feature_label_independence", "passed": True, "details": {}}

    # Check a sample of feature npz files — they should only contain
    # features, timestamps, frame_indices + metadata, NOT step labels
    ok_feats = sorted((FEAT_DIR / "ok").glob("*.npz"))
    wr_feats = sorted((FEAT_DIR / "wr").glob("*.npz"))

    forbidden_keys = ["step_label", "verb_label", "target_label", "event_label", "ignore_mask"]

    for feat_type, feat_list in [("ok", ok_feats[:3]), ("wr", wr_feats[:3])]:
        for fp in feat_list:
            try:
                data = np.load(fp, allow_pickle=True)
                keys = list(data.keys())
                leaked = [k for k in forbidden_keys if k in keys]
                if leaked:
                    results["passed"] = False
                    results["details"][fp.stem] = f"CONTAINS LABEL KEYS: {leaked}"
                # Check feature dimension
                if "features" in data:
                    dim = data["features"].shape[-1]
                    if dim != 90:
                        results["passed"] = False
                        results["details"][f"{fp.stem}_dim"] = f"WRONG DIM: {dim}"
            except Exception as e:
                results["details"][fp.stem] = f"READ ERROR: {e}"

    results["details"]["ok_features"] = len(ok_feats)
    results["details"]["wr_features"] = len(wr_feats)
    results["details"]["total_features"] = len(ok_feats) + len(wr_feats)

    # Check labels exist
    ok_labels = sorted((LABEL_DIR / "ok").glob("*.npz"))
    wr_labels = sorted((LABEL_DIR / "wr").glob("*.npz"))
    results["details"]["ok_labels"] = len(ok_labels)
    results["details"]["wr_labels"] = len(wr_labels)
    results["details"]["total_labels"] = len(ok_labels) + len(wr_labels)

    # wr_60 should be MISSING
    wr_label_names = {f.stem for f in wr_labels}
    wr_feat_names = {f.stem for f in wr_feats}
    feats_without_labels = wr_feat_names - wr_label_names
    if feats_without_labels:
        results["details"]["features_without_labels"] = sorted(feats_without_labels)

    return results


def audit_model_checkpoint():
    """Check 3: Model checkpoint is 90-dim v2 model"""
    results = {"check": "model_checkpoint", "passed": True, "details": {}}

    ckpt_path = MODEL_DIR / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        results["passed"] = False
        results["details"]["error"] = f"MISSING: {ckpt_path}"
        return results

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        results["passed"] = False
        results["details"]["error"] = f"LOAD ERROR: {e}"
        return results

    results["details"]["input_dim"] = ckpt.get("input_dim", "MISSING")
    results["details"]["hidden_dim"] = ckpt.get("hidden_dim", "MISSING")
    results["details"]["num_classes_step"] = ckpt.get("num_classes_step", "MISSING")
    results["details"]["model"] = ckpt.get("model", ckpt.get("arch", "UNKNOWN"))
    results["details"]["total_params"] = ckpt.get("total_params", "UNKNOWN")
    results["details"]["best_epoch"] = ckpt.get("best_epoch", "UNKNOWN")

    # Verify input_dim == 90
    if ckpt.get("input_dim") != 90:
        results["passed"] = False
        results["details"]["input_dim_error"] = (
            f"Expected 90, got {ckpt.get('input_dim')}")

    # Verify not loading old 130-dim model
    if ckpt.get("input_size") == 130 or ckpt.get("input_dim") == 130:
        results["passed"] = False
        results["details"]["input_dim_error"] = "Would load OLD 130-dim model!"

    return results


def audit_fsm_evaluation_independence():
    """Check 4: FSM evaluation uses model predictions, NOT GT labels"""
    results = {"check": "fsm_evaluation_independence", "passed": True, "details": {}}

    # Read eval_pipeline_offline_v2.py and verify it:
    # 1. Calls sliding_window_inference on model
    # 2. Passes preds (not labels) to simulate_fsm_on_predictions
    eval_script = ROOT / "eval_pipeline_offline_v2.py"
    if not eval_script.exists():
        # Try sop_system
        eval_script = ROOT.parent / "sop_system" / "eval_pipeline_offline_v2.py"

    if eval_script.exists():
        content = eval_script.read_text(encoding="utf-8")
        checks = {
            "uses_model_inference": "sliding_window_inference(features, model" in content,
            "fsm_uses_preds_not_labels": "simulate_fsm_on_predictions(preds, confs" in content,
            "event_gt_from_segments": "segments_df" in content,
        }
        results["details"]["code_checks"] = checks
        if not checks["uses_model_inference"]:
            results["passed"] = False
        if not checks["fsm_uses_preds_not_labels"]:
            results["passed"] = False
            results["details"]["warning"] = "FSM might use ground truth labels directly!"
    else:
        results["details"]["warning"] = "eval_pipeline_offline_v2.py not found for audit"

    return results


def audit_data_counts():
    """Check 5: Data counts match expectations"""
    results = {"check": "data_counts", "passed": True, "details": {}}

    # Feature counts
    ok_feats = len(list((FEAT_DIR / "ok").glob("*.npz")))
    wr_feats = len(list((FEAT_DIR / "wr").glob("*.npz")))
    results["details"]["features"] = {"ok": ok_feats, "wr": wr_feats, "total": ok_feats + wr_feats}

    # Label counts
    ok_labels = len(list((LABEL_DIR / "ok").glob("*.npz")))
    wr_labels = len(list((LABEL_DIR / "wr").glob("*.npz")))
    results["details"]["labels"] = {"ok": ok_labels, "wr": wr_labels, "total": ok_labels + wr_labels}

    # Segment counts
    if SEGMENT_CSV.exists():
        import pandas as pd
        df = pd.read_csv(SEGMENT_CSV)
        results["details"]["segments"] = len(df)
        results["details"]["segments_by_step"] = df["step_id"].value_counts().to_dict()

    # Window counts
    for split_name in ["train", "val", "test"]:
        index_path = DATASET_DIR / f"{split_name}_index.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            windows = data.get("windows", data) if isinstance(data, dict) else data
            results["details"][f"{split_name}_windows"] = len(windows)

    return results


def main():
    print("=" * 60)
    print("v2_90 Integrity Audit")
    print("=" * 60)

    all_checks = {}

    # Run all checks
    checks = [
        ("split_integrity", audit_split_integrity),
        ("feature_label_independence", audit_feature_label_independence),
        ("model_checkpoint", audit_model_checkpoint),
        ("fsm_evaluation_independence", audit_fsm_evaluation_independence),
        ("data_counts", audit_data_counts),
    ]

    all_passed = True
    for name, check_fn in checks:
        print(f"\n[{name}] Running...")
        result = check_fn()
        all_checks[name] = result
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {status}")
        if not result["passed"]:
            all_passed = False
            for k, v in result.get("details", {}).items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sv, bool) and not sv:
                            print(f"  ISSUE: {k}.{sk} = {sv}")
                elif isinstance(v, str) and ("ERROR" in v or "MISSING" in v or "WRONG" in v):
                    print(f"  ISSUE: {k} = {v}")

    # Summary
    audit_report = {
        "audit_date": "2025-05-20",
        "audit_version": "v2_90",
        "overall_pass": all_passed,
        "checks": all_checks,
    }

    output_path = REPORTS_DIR / "audit_integrity_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'ISSUES FOUND'}")
    print(f"Report: {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
