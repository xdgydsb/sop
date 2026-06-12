"""测试完整pipeline — YOLO+手部+交互+物理+时序+融合"""
import sys, cv2, numpy as np, torch
from pathlib import Path
from collections import deque
sys.path.insert(0, str(Path(__file__).parent))

from engine.yolo_detector import YOLODetector
from engine.hand_detector import HandDetector
from engine.physical_state import PhysicalStateEngine
from engine.fusion import FusionEngine
from engine.temporal_lstm import SOPActionGRU, FeatureExtractor

YOLO_PATH = "models/yolo_final_v1.pt"
MODEL_PATH = "training/runs/best_sequence_v5.pt"
STEP_NAMES = {0: "Idle", 1: "S1-Open", 2: "S2-Earphone", 3: "S3-Charger",
              4: "S4-GreenBag", 5: "S5-Close", 6: "Done"}


def load_temporal_model(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    input_size = ckpt.get("input_size", 130)
    hidden_size = ckpt.get("hidden_size", 256)
    num_layers = ckpt.get("num_layers", 3)
    num_classes = ckpt.get("num_classes", 7)

    model = SOPActionGRU(input_size=input_size, hidden_size=hidden_size,
                         num_layers=num_layers, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Temporal model loaded: {input_size}d, val_acc={ckpt.get('val_acc', 0):.3f}")
    return model, input_size


def test(video_path):
    print(f"Testing: {video_path}")
    device = "cuda"
    yolo = YOLODetector(YOLO_PATH, conf_thresh=0.3, device=device)
    hand = HandDetector()
    phys = PhysicalStateEngine()
    fusion = FusionEngine(confirm_count=2, model_window=8)
    feat_extractor = FeatureExtractor(yolo, hand)

    model, input_size = load_temporal_model(MODEL_PATH, device)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total}")

    step_transitions = []
    prev_step = 0
    fi = 0
    feature_buffer = deque(maxlen=128)
    model_probs_history = deque(maxlen=12)  # 平滑模型预测

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % 2 != 0:
            fi += 1
            continue

        h, w = frame.shape[:2]
        detections = yolo.detect(frame)
        hands_list = hand.detect(frame)
        box_state, _ = yolo.get_box_state(detections)
        box_bbox = yolo.get_box_bbox(detections)

        interaction = hand.compute_interaction(hands_list, detections, box_bbox, (h, w))
        hand_near_box = any(d < 0.25 for d in interaction["hand_box_dist"] if d < 999)

        phys_result = phys.update(
            detections, box_state, box_bbox, hand_near_box, len(hands_list) > 0,
            hand_obj_iou=interaction["hand_obj_iou"],
            hand_box_dist=interaction["hand_box_dist"])

        feat = feat_extractor.extract(frame, detections, hands_list, interaction, box_bbox, box_state)
        feature_buffer.append(feat)

        # 时序模型推理：每8帧跑一次（更频繁），用全buffer
        model_step, model_conf, model_top3 = 0, 0.0, []
        if fi % 6 == 0 and len(feature_buffer) >= 12:
            feats = np.stack(list(feature_buffer), axis=0)
            x = torch.FloatTensor(feats).unsqueeze(0).to(device)  # (1, T, D)
            with torch.no_grad():
                logits = model(x)  # (1, T, 7)
                probs = torch.softmax(logits[0], dim=-1)  # (T, 7)

                # 取最后8帧的平均概率（减少噪声）
                last_probs = probs[-8:] if probs.shape[0] >= 8 else probs
                avg_probs = last_probs.mean(dim=0)

                top3_probs, top3_idx = torch.topk(avg_probs, 3)
                model_step = top3_idx[0].item()
                model_conf = top3_probs[0].item()
                model_top3 = [(idx.item(), prob.item()) for idx, prob in zip(top3_idx, top3_probs)]

        # 物理步骤（不依赖模型）
        phys_step = phys_result.current_phys_step

        # 融合
        fusion_result = fusion.update(phys_result, model_step, model_conf, model_top3, fi / 30.0)

        if fusion_result.step != prev_step:
            step_transitions.append({
                "frame": fi, "from": prev_step, "to": fusion_result.step,
                "name": fusion_result.step_name,
                "placed": phys_result.placed_this_frame,
                "model_step": fusion_result.model_step, "model_conf": fusion_result.model_conf,
                "model_top3": model_top3,
                "objects_placed": list(phys_result.objects_placed),
                "box_state": box_state,
            })
            prev_step = fusion_result.step

            top3_str = " ".join(["S%d(%.2f)" % (s, c) for s, c in model_top3[:3]])
            print("  [%4d] S%d (%s) model=(%s) box=%s phys=%d placed=%s" % (
                fi, fusion_result.step, fusion_result.step_name,
                top3_str, box_state, phys_step, phys_result.objects_placed))

        # 每30帧打印模型预测（调试用）
        if fi % 30 == 0 and model_top3:
            top3_str = " ".join(["S%d(%.2f)" % (s, c) for s, c in model_top3[:3]])
            print("  [%4d] model_pred=%s box=%s placed=%s" % (
                fi, top3_str, box_state, phys_result.objects_placed))

        fi += 1

    cap.release()

    print("\n=== Step Transitions ===")
    for t in step_transitions:
        extra = " [PLACED: %s]" % t["placed"] if t["placed"] else ""
        print("  f%5d: S%d -> S%d (%s)%s" % (t["frame"], t["from"], t["to"], t["name"], extra))

    steps_seen = set(t["to"] for t in step_transitions)
    print("\nUnique steps: %s" % sorted(steps_seen))
    if len(steps_seen) >= 5:
        print("*** SUCCESS: Full SOP sequence! ***")
    elif len(steps_seen) >= 3:
        print("Partial: %d/5 steps" % len(steps_seen))
    else:
        print("Needs work: %d/5 steps" % len(steps_seen))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/home/zhaowei/shabi/data/ok/ok_100.avi")
    args = parser.parse_args()
    test(args.video)
