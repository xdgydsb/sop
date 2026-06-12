"""
批量视觉管线测试 — YOLO+MediaPipe+PhysicalState+EventDetector+FSM
无头模式，输出每视频的详细事件日志和最终判定

Usage:
  python batch_pipeline_test.py                     # 测试所有下载的视频
  python batch_pipeline_test.py --single ok_20      # 单视频测试
  python batch_pipeline_test.py --max-frames 150     # 每视频最多帧数
"""
import sys
import time
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import torch

from engine.yolo_detector import YOLODetector
from engine.hand_detector import HandDetector
from engine.physical_state import PhysicalStateEngine
from engine.sop_fsm import SOPStateMachine, SOPStep, STEP_NAMES
from engine.temporal_lstm import SOPActionGRU, FeatureExtractor

TEST_VIDEOS_DIR = Path(__file__).parent / "data" / "test_videos"
MODELS_DIR = Path(__file__).parent / "models"

STEP_TARGET_OBJECT = {2: "earphone", 3: "charger", 4: "green_bag"}

# 期望的物品顺序
OBJECT_ORDER = ["earphone", "charger", "green_bag"]


class BatchPipelineTester:
    def __init__(self, yolo_path: str, model_path: str,
                 device: str = "cpu", conf: float = 0.35, imgsz: int = 320):
        self.device = device

        print(f"[1/5] YOLO (imgsz={imgsz})...")
        self.yolo = YOLODetector(yolo_path, conf_thresh=conf, device=device, imgsz=imgsz)

        print("[2/5] MediaPipe...")
        self.hand_detector = HandDetector()

        print("[3/5] Model...")
        self.temporal_model, self.input_size = self._load_model(model_path)
        self.feature_extractor = FeatureExtractor(self.yolo, self.hand_detector)

        print("[4/5] PhysicalState...")
        self.physical = PhysicalStateEngine(confirm_frames=6)

        print("[5/5] FSM...")
        self.fsm = SOPStateMachine(timeout=90.0, min_step_duration=0.3)

        # State
        self.feature_buffer = []
        self.frame_idx = 0
        self.model_interval = 4
        self._model_history = []
        self._running_max = 0
        self._ema_model_probs = np.zeros(7, dtype=np.float32)

    def _load_model(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        model = SOPActionGRU(
            input_size=ckpt["input_size"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            num_classes=ckpt["num_classes"],
            dropout=0.4,
        ).to(self.device)
        model.load_state_dict(sd)
        model.eval()
        return model, ckpt["input_size"]

    def reset(self):
        self.yolo.reset_tracking()
        self.physical.reset()
        self.fsm.reset()
        self.feature_buffer.clear()
        self.frame_idx = 0
        self._model_history.clear()
        self._running_max = 0
        self._ema_model_probs = np.zeros(7, dtype=np.float32)

    def process_frame(self, frame: np.ndarray) -> Dict:
        """处理单帧，返回详细状态字典"""
        h, w = frame.shape[:2]
        self.frame_idx += 1

        # 1. YOLO
        detections = self.yolo.detect(frame)

        # 2. MediaPipe
        hands = self.hand_detector.detect(frame)

        # 3. Box state
        box_state, box_conf = self.yolo.get_box_state(detections)
        box_bbox = self.yolo.get_box_bbox(detections)

        # 4. Interaction
        interaction = self.hand_detector.compute_interaction(
            hands, detections, box_bbox, (h, w))
        hand_near_box = any(d < 0.15 for d in interaction.get("hand_box_dist", [999, 999]))

        # 5. Update trajectory states
        self._update_trajectory_states(detections, box_bbox, interaction)

        # 6. Physical state
        phys_result = self.physical.update(
            detections, box_state, box_bbox, hand_near_box, len(hands) > 0,
            hand_obj_iou=interaction.get("hand_obj_iou"),
            holding_objects=[],
            hand_box_dist=interaction.get("hand_box_dist"),
            tracked_objects=self.yolo.tracked_objects)

        # 7. Active hand + Feature
        current_step = self.fsm.current_step.value
        target_obj = STEP_TARGET_OBJECT.get(current_step + 1, None) if 0 <= current_step < 5 else None
        target_bbox = None
        if target_obj:
            target_det = next((d for d in detections if d.cls_name == target_obj), None)
            if target_det:
                target_bbox = target_det.bbox
        self.hand_detector.select_active_hand(hands, target_bbox, box_bbox)

        try:
            feat = self.feature_extractor.extract(
                frame, detections, hands, interaction, box_bbox, box_state)
        except Exception:
            feat = np.zeros(self.input_size, dtype=np.float32)
        self.feature_buffer.append(feat)
        if len(self.feature_buffer) > 160:
            self.feature_buffer.pop(0)

        # 8. Model inference
        model_step, model_conf, model_top3 = 0, 0.0, []
        if self.frame_idx % self.model_interval == 0 and len(self.feature_buffer) >= 16:
            model_step, model_conf, model_top3 = self._run_model()

        # 9. Event detection
        detected_step, detected_conf, phys_ok = self._detect_event(
            phys_result, model_step, model_conf, model_top3, box_bbox)

        # 10. FSM
        fsm_result = self.fsm.validate(
            detected_step, detected_conf, phys_ok, time.time())

        # Collect per-object tracked state
        obj_track_state = {}
        for obj_name in OBJECT_ORDER:
            tracked_list = self.yolo.get_tracked_by_name(obj_name)
            if tracked_list:
                t = tracked_list[0]
                obj_track_state[obj_name] = {
                    "visible": True,
                    "in_box": t.in_box,
                    "stable_in_box_frames": t.stable_in_box_frames,
                    "in_init_region": t.in_init_region,
                    "touched_by_hand": t.touched_by_hand,
                    "hand_touch_frames": t.hand_touch_frames,
                    "velocity": list(t.velocity),
                }
            else:
                obj_track_state[obj_name] = {"visible": False}

        return {
            "frame": self.frame_idx,
            "box_state": box_state,
            "box_is_open": phys_result.box_is_open,
            "box_is_closed": phys_result.box_is_closed,
            "box_state_conf": phys_result.box_state_conf,
            "phys_step": phys_result.current_phys_step,
            "objects_placed": list(phys_result.objects_placed),
            "visible_objects": phys_result.visible_objects,
            "wrong_placement": phys_result.wrong_placement,
            "wrong_placement_frames": phys_result.wrong_placement_frames,
            "placement_stages": {k: v.name for k, v in phys_result.placement_stages.items()},
            "stable_frames": dict(phys_result.stable_frames),
            "model_step": model_step,
            "model_conf": model_conf,
            "model_top3": [(s, round(c, 3)) for s, c in model_top3],
            "detected_step": detected_step,
            "detected_conf": detected_conf,
            "phys_ok": phys_ok,
            "fsm_step": fsm_result.step_id,
            "fsm_is_correct": fsm_result.is_correct,
            "fsm_has_error": fsm_result.has_error,
            "fsm_error_type": fsm_result.error_type,
            "fsm_message": fsm_result.message,
            "obj_track_state": obj_track_state,
            "hands_detected": phys_result.hands_detected,
            "hand_near_box": phys_result.hand_near_box,
        }

    def _update_trajectory_states(self, detections, box_bbox, interaction):
        if box_bbox is None:
            return
        hand_obj_iou = interaction.get("hand_obj_iou", np.zeros((2, 5), dtype=np.float32))
        obj_name_to_idx = {"earphone": 2, "charger": 3, "green_bag": 4}

        for det in detections:
            if det.track_id < 0 or det.cls_name in ("box_open", "box_closed"):
                continue
            cx, cy = det.center
            bx1, by1, bx2, by2 = box_bbox
            in_box = (bx1 <= cx <= bx2 and by1 <= cy <= by2)
            in_box_ratio = self.yolo.compute_in_box_ratio(det.bbox, box_bbox)
            self.yolo.update_in_box_status(det.track_id, in_box or in_box_ratio > 0.3)

            touched = False
            oi = obj_name_to_idx.get(det.cls_name, -1)
            if oi >= 0:
                for hi in range(min(2, hand_obj_iou.shape[0])):
                    if hand_obj_iou[hi, oi] > 0.04:
                        touched = True
                        break
            self.yolo.update_hand_touch(det.track_id, touched)

    def _run_model(self):
        features = np.stack(list(self.feature_buffer), axis=0)
        feat_dim = features.shape[-1]
        if feat_dim != self.input_size:
            return 0, 0.0, []

        seq_len = min(128, len(features))
        seq = features[-seq_len:]
        x = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.temporal_model(x)
            probs = torch.softmax(logits[0], dim=-1)
            last_probs = probs[-8:] if probs.shape[0] >= 8 else probs
            avg_probs = last_probs.mean(dim=0).cpu().numpy()

        alpha = 0.4
        self._ema_model_probs = alpha * avg_probs + (1 - alpha) * self._ema_model_probs

        pred = int(np.argmax(self._ema_model_probs))
        conf = float(self._ema_model_probs[pred])
        top3 = sorted([(i, float(self._ema_model_probs[i])) for i in range(7)
                      if self._ema_model_probs[i] > 0.02], key=lambda x: -x[1])[:3]
        return pred, conf, top3

    def _detect_event(self, phys_result, model_step, model_conf, model_top3, box_bbox):
        from collections import Counter
        self._model_history.append(model_step)
        if len(self._model_history) > 12:
            self._model_history.pop(0)
        smoothed_step = Counter(self._model_history).most_common(1)[0][0] if self._model_history else 0
        if smoothed_step > self._running_max:
            self._running_max = smoothed_step

        phys_step = phys_result.current_phys_step
        phys_ok = not phys_result.is_error

        # Rule 0: Wrong object
        if phys_result.wrong_placement and phys_result.wrong_placement_frames >= 3:
            obj_to_step = {"earphone": 2, "charger": 3, "green_bag": 4}
            wrong_step = obj_to_step.get(phys_result.wrong_placement, 0)
            if wrong_step > 0:
                return (wrong_step, 0.6, False)

        # Rule 1: Box closed → S5 (only if objects have been placed, avoid initial closed state)
        if phys_result.box_is_closed:
            n_placed = len(phys_result.objects_placed)
            # Must have at least 1 object placed to avoid false S5 at video start
            if n_placed >= 1 or phys_step >= 4:
                return (5, max(model_conf if model_step == 5 else 0.7, phys_result.box_state_conf), True)
            # Box closed but nothing placed → still idle, wait for box to open
            if phys_step == 0:
                return (0, 0.5, True)

        # Rule 2: Box open + idle → S1
        if phys_result.box_is_open and phys_step <= 1:
            if model_step == 1 and model_conf > 0.4:
                return (1, model_conf, True)
            if phys_step == 1:
                return (1, 0.7, True)
            # Box just opened, but model hasn't confirmed yet
            if phys_step == 0 and model_step == 0:
                return (1, 0.55, True)  # Force S1 when box opens

        # Rule 3: S2-S4 during box open
        if phys_result.box_is_open and 1 <= phys_step <= 4:
            target_obj = STEP_TARGET_OBJECT.get(phys_step + 1, None)
            trajectory_ok = True
            if target_obj:
                tracked_list = self.yolo.get_tracked_by_name(target_obj)
                if tracked_list:
                    tobj = tracked_list[0]
                    trajectory_ok = (not tobj.in_init_region) or (tobj.stable_in_box_frames >= 3)

            model_ready = (self._running_max >= phys_step + 1 and model_conf > 0.3)
            model_hints_target = any(s == phys_step + 1 and c > 0.15 for s, c in model_top3)

            if model_ready or model_hints_target:
                return (phys_step + 1, max(model_conf, 0.5), phys_ok and trajectory_ok)
            if phys_step > self._running_max:
                return (phys_step, 0.6, phys_ok)

        # Rule 4: Default
        step = max(phys_step, self._running_max)
        step = max(0, min(5, step))
        return (step, model_conf if model_step == step else 0.4, phys_ok)


def test_video(video_path: str, tester: BatchPipelineTester,
               max_frames: int = None, stride: int = 2,
               verbose: bool = True) -> Dict:
    """测试单个视频，返回完整结果"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tester.reset()

    name = Path(video_path).stem
    cat = "ok" if "ok" in str(video_path) else ("wr" if "wr" in str(video_path) else "yuanzi")

    frames_data = []
    events = []  # step changes
    prev_fsm_step = -1
    fc = 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {name}  |  {cat}  |  {total_frames}f  |  {fps:.0f}fps  |  stride={stride}")
        print(f"{'='*70}")

    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fc += 1
        if fc % stride != 0:
            continue
        if max_frames and fc > max_frames:
            break

        data = tester.process_frame(frame)
        frames_data.append(data)

        # Track FSM step changes
        if data["fsm_step"] != prev_fsm_step:
            prev_fsm_step = data["fsm_step"]
            event = {
                "frame": fc,
                "step": data["fsm_step"],
                "detected_step": data["detected_step"],
                "model_step": data["model_step"],
                "model_conf": data["model_conf"],
                "box_state": data["box_state"],
                "phys_step": data["phys_step"],
                "objects_placed": data["objects_placed"],
                "is_correct": data["fsm_is_correct"],
                "error_type": data["fsm_error_type"],
                "message": data["fsm_message"],
            }
            events.append(event)
            if verbose:
                status = "OK" if data["fsm_is_correct"] else f"ERR:{data['fsm_error_type']}"
                print(f"  [f{fc:4d}] →S{data['fsm_step']} {status} "
                      f"model=S{data['model_step']}:{data['model_conf']:.2f} "
                      f"box={data['box_state']} phys={data['phys_step']} "
                      f"placed={data['objects_placed']} "
                      f"stages={data['placement_stages']}")

    cap.release()
    elapsed = time.time() - t0

    # Final FSM state
    final_step = tester.fsm.current_step.value
    final_is_error = tester.fsm.current_step == SOPStep.ERROR
    fsm_path = [h["step"] for h in tester.fsm.step_history]

    # Physical event summary
    phys_events = []
    prev_placed = []
    for fd in frames_data:
        if fd["objects_placed"] != prev_placed:
            phys_events.append({
                "frame": fd["frame"],
                "objects_placed": list(fd["objects_placed"]),
            })
            prev_placed = list(fd["objects_placed"])

    # Determine verdict
    if cat == "ok":
        verdict = "PASS" if final_step >= 5 and not final_is_error else "FAIL"
    else:
        verdict = "PASS" if final_is_error else "MISS"

    result = {
        "name": name, "category": cat,
        "total_frames": total_frames, "processed_frames": fc,
        "fps": fps, "elapsed": elapsed,
        "final_step": final_step,
        "final_is_error": final_is_error,
        "fsm_path": fsm_path,
        "fsm_errors": [e for e in events if e["error_type"]],
        "events": events,
        "phys_events": phys_events,
        "verdict": verdict,
        "last_box_state": frames_data[-1]["box_state"] if frames_data else "?",
        "last_objects_placed": frames_data[-1]["objects_placed"] if frames_data else [],
        "last_placement_stages": frames_data[-1]["placement_stages"] if frames_data else {},
    }

    if verbose:
        print(f"\n  Result: {verdict}  "
              f"final=S{final_step}  path={fsm_path}  "
              f"errors={len(result['fsm_errors'])}  "
              f"elapsed={elapsed:.1f}s ({fc/elapsed:.1f}fps)")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=str, help="Test single video by name (e.g. ok_20)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--yolo", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    yolo_path = args.yolo or str(MODELS_DIR / "yolo_final_v1.pt")
    model_path = args.model or str(MODELS_DIR / "best_sequence_v5.pt")

    print("=" * 70)
    print("Batch Pipeline Test — YOLO+MediaPipe+PhysicalState+EventDetector+FSM")
    print(f"  Device: {args.device}  Max frames: {args.max_frames or 'unlimited'}")
    print("=" * 70)

    tester = BatchPipelineTester(yolo_path, model_path, device=args.device, conf=args.conf, imgsz=320)

    # Collect videos
    if args.single:
        # Find the video file
        for sub in ["ok", "wr", "yuanzi"]:
            p = TEST_VIDEOS_DIR / sub / f"{args.single}.avi"
            if p.exists():
                videos = [p]
                break
        else:
            print(f"Video not found: {args.single}")
            return
    else:
        videos = []
        for sub in ["yuanzi", "ok", "wr"]:
            vdir = TEST_VIDEOS_DIR / sub
            if vdir.exists():
                videos.extend(sorted(vdir.glob("*.avi")))

    print(f"\n[Videos] {len(videos)} to test")

    all_results = []
    ok_pass = ok_fail = 0
    wr_pass = wr_miss = 0

    for vp in videos:
        result = test_video(vp, tester, max_frames=args.max_frames, stride=2)
        all_results.append(result)

        if result["category"] == "ok":
            if result["verdict"] == "PASS":
                ok_pass += 1
            else:
                ok_fail += 1
        elif result["category"] == "wr":
            if result["verdict"] == "PASS":
                wr_pass += 1
            else:
                wr_miss += 1

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  OK: {ok_pass} PASS, {ok_fail} FAIL  ({ok_pass/max(ok_pass+ok_fail,1):.1%})")
    print(f"  WR: {wr_pass} DETECTED, {wr_miss} MISSED  ({wr_pass/max(wr_pass+wr_miss,1):.1%})")

    # Per-video details
    print(f"\n{'Name':<15} {'Cat':<8} {'Frames':>6} {'Verdict':>8} {'Final':>6} {'FSMErrs':>7} {'Placed':<20}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['name']:<15} {r['category']:<8} {r['processed_frames']:>6} "
              f"{r['verdict']:>8} S{r['final_step']:<5} {len(r['fsm_errors']):>7} "
              f"{str(r['last_objects_placed']):<20}")

    # Save detailed results
    out_path = Path(__file__).parent / "batch_test_results.json"
    # Convert to serializable
    serializable = []
    for r in all_results:
        r2 = dict(r)
        r2.pop("events", None)  # too verbose for JSON
        r2.pop("phys_events", None)
        serializable.append(r2)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
