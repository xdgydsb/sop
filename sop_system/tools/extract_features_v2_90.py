"""
Step 2: 完整视频逐帧提取90维特征 (15fps)
输入: data/raw_videos/{ok,wr,yuanzi}/
输出: data/features/v2_90/{video_type}/{video_name}.npz
"""
import sys
import time
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import deque

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.yolo_detector import YOLODetector, Detection, TrackedObject
from engine.hand_detector import HandDetector


# ── 90-dim feature layout ──
# [  0- 41] Active hand keypoints (21×2)           42
# [ 42- 49] Hand global state                      8
# [ 50- 56] Box state                              7
# [ 57- 83] Object interaction (3 obj × 9)         27
# [ 84- 89] Object velocity (3 obj × 2)            6
# Total: 90

OBJ_NAMES = ["box_closed", "box_open", "earphone", "charger", "green_bag"]
INTERACT_OBJS = ["earphone", "charger", "green_bag"]
OBJ_TO_IDX = {"earphone": 2, "charger": 3, "green_bag": 4}

RAW_VIDEOS_DIR = ROOT.parent / "data"  # d:/shabi/data/
FEAT_OUT_DIR = ROOT / "data" / "features" / "v2_90"
MODELS_DIR = ROOT / "models"


class FeatureExtractorV2Standalone:
    """Standalone 90-dim feature extractor with EMA and hold-last-value"""

    def __init__(self, yolo_detector, hand_detector,
                 ema_alpha: float = 0.5, hold_frames: int = 5):
        self.yolo = yolo_detector
        self.hand = hand_detector
        self.ema_alpha = ema_alpha
        self.hold_frames = hold_frames
        self._ema_state = None      # EMA of 90-dim feature
        self._last_valid = None     # Last valid feature for hold
        self._lost_count = 0

    def compute_interaction(self, detections, hands, box_bbox, h, w):
        """Compute hand-object interaction matrices"""
        det_dict = {d.cls_name: d for d in detections}
        hand_obj_iou = np.zeros((2, 5), dtype=np.float32)
        hand_obj_dist = np.zeros((2, 5), dtype=np.float32)

        for hi, hand in enumerate(hands[:2]):
            if hand.bbox is None:
                continue
            hx1, hy1, hx2, hy2 = hand.bbox
            h_area = max(1, (hx2 - hx1) * (hy2 - hy1))
            hand_cx = (hx1 + hx2) / 2
            hand_cy = (hy1 + hy2) / 2

            for oi, oname in enumerate(OBJ_NAMES):
                d = det_dict.get(oname)
                if d is None:
                    continue
                ox, oy = d.center
                dist = np.sqrt((hand_cx - ox) ** 2 + (hand_cy - oy) ** 2) / w
                hand_obj_dist[hi, oi] = 1.0 / (1.0 + dist * 10)

                # IoU between hand bbox and object bbox
                ox1, oy1, ox2, oy2 = d.bbox
                ix1, iy1 = max(hx1, ox1), max(hy1, oy1)
                ix2, iy2 = min(hx2, ox2), min(hy2, oy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = h_area + max(1, (ox2 - ox1) * (oy2 - oy1)) - inter
                    hand_obj_iou[hi, oi] = inter / max(union, 1)

        return {"hand_obj_iou": hand_obj_iou, "hand_obj_dist": hand_obj_dist}

    def extract(self, frame, detections, hands, box_bbox, box_state) -> np.ndarray:
        """Extract 90-dim feature for one frame"""
        h, w = frame.shape[:2]
        det_dict = {d.cls_name: d for d in detections}

        interaction = self.compute_interaction(detections, hands, box_bbox, h, w)
        hand_obj_iou = interaction["hand_obj_iou"]
        hand_obj_dist = interaction["hand_obj_dist"]

        # ── 1. Active hand keypoints (42 dims) ──
        active_hand = self.hand.get_active_hand(hands)
        hand_feat = np.zeros(42, dtype=np.float32)

        if active_hand is not None and active_hand.landmarks is not None:
            palm_indices = [0, 5, 9, 13, 17]
            valid_marks = active_hand.landmarks
            if len(valid_marks) >= 21:
                palm_x = float(np.mean([valid_marks[i, 0] for i in palm_indices]))
                palm_y = float(np.mean([valid_marks[i, 1] for i in palm_indices]))
                hand_scale = max(0.01, active_hand.openness * 0.5 + 0.1)

                for li in range(21):
                    hand_feat[li * 2] = (valid_marks[li, 0] - palm_x) / hand_scale
                    hand_feat[li * 2 + 1] = (valid_marks[li, 1] - palm_y) / hand_scale

        # ── 2. Hand global state (8 dims) ──
        hand_global = np.zeros(8, dtype=np.float32)
        palm_cx_px, palm_cy_px = w / 2, h / 2
        if active_hand is not None and active_hand.landmarks is not None:
            palm_indices = [0, 5, 9, 13, 17]
            if len(active_hand.landmarks) >= 21:
                palm_x = float(np.mean([active_hand.landmarks[i, 0] for i in palm_indices]))
                palm_y = float(np.mean([active_hand.landmarks[i, 1] for i in palm_indices]))
                palm_cx_px = palm_x * w
                palm_cy_px = palm_y * h
                handedness = active_hand.handedness
                vx, vy = self.hand.palm_velocity.get(handedness, (0.0, 0.0))

                hand_global[0] = 1.0  # detected
                hand_global[1] = active_hand.openness
                hand_global[2] = palm_x
                hand_global[3] = palm_y
                hand_global[4] = vx / max(w, 1)
                hand_global[5] = vy / max(h, 1)
                if active_hand.bbox is not None:
                    hand_global[6] = (active_hand.bbox[2] - active_hand.bbox[0]) / w
                    hand_global[7] = (active_hand.bbox[3] - active_hand.bbox[1]) / h

        # ── 3. Box state (7 dims) ──
        box_feat = np.zeros(7, dtype=np.float32)
        box_open_det = det_dict.get("box_open")
        box_closed_det = det_dict.get("box_closed")
        box_feat[0] = box_open_det.confidence if box_open_det else 0.0
        box_feat[1] = box_closed_det.confidence if box_closed_det else 0.0
        active_box = box_open_det or box_closed_det
        if active_box and box_bbox:
            bx1, by1, bx2, by2 = box_bbox
            box_feat[2] = (bx1 + bx2) / (2 * w)
            box_feat[3] = (by1 + by2) / (2 * h)
            box_feat[4] = (bx2 - bx1) / w
            box_feat[5] = (by2 - by1) / h
            box_feat[6] = active_box.confidence

        # ── 4. Object interaction (27 dims) ──
        obj_interact = np.zeros(27, dtype=np.float32)
        for oi, obj_name in enumerate(INTERACT_OBJS):
            base = oi * 9
            d = det_dict.get(obj_name)
            if d:
                obj_interact[base + 0] = d.confidence
                obj_interact[base + 1] = d.center[0] / w
                obj_interact[base + 2] = d.center[1] / h
                obj_interact[base + 3] = (d.bbox[2] - d.bbox[0]) / w
                obj_interact[base + 4] = (d.bbox[3] - d.bbox[1]) / h
                if box_bbox:
                    obj_interact[base + 5] = self.yolo.compute_in_box_ratio(d.bbox, box_bbox)
                if active_hand is not None and palm_cx_px > 0:
                    obj_interact[base + 6] = 1.0 / (1.0 + np.sqrt(
                        (d.center[0] - palm_cx_px) ** 2 + (d.center[1] - palm_cy_px) ** 2
                    ) / w * 10)
                class_idx = OBJ_TO_IDX.get(obj_name, -1)
                if class_idx >= 0 and hand_obj_iou.shape[0] > 0:
                    hi = min(self.hand.active_hand_idx, hand_obj_iou.shape[0] - 1)
                    if hi >= 0:
                        obj_interact[base + 7] = hand_obj_iou[hi, class_idx]
                tracked_list = self.yolo.get_tracked_by_name(obj_name)
                if tracked_list:
                    obj_interact[base + 8] = 1.0 if tracked_list[0].in_init_region else 0.0

        # ── 5. Object velocity (6 dims) ──
        obj_velocity = np.zeros(6, dtype=np.float32)
        for oi, obj_name in enumerate(INTERACT_OBJS):
            tracked_list = self.yolo.get_tracked_by_name(obj_name)
            if tracked_list:
                tobj = tracked_list[0]
                obj_velocity[oi * 2] = tobj.velocity[0] / max(w, 1)
                obj_velocity[oi * 2 + 1] = tobj.velocity[1] / max(h, 1)

        feat = np.concatenate([
            hand_feat, hand_global, box_feat, obj_interact, obj_velocity
        ]).astype(np.float32)

        # EMA smoothing
        if self._ema_state is None:
            self._ema_state = feat.copy()
        else:
            self._ema_state = self.ema_alpha * feat + (1 - self.ema_alpha) * self._ema_state

        # Hold last valid when hand is lost
        if hand_global[0] > 0.01:
            self._last_valid = self._ema_state.copy()
            self._lost_count = 0
        else:
            self._lost_count += 1
            if self._lost_count <= self.hold_frames and self._last_valid is not None:
                return self._last_valid.copy()
            # Beyond hold, return zeros for hand features
            self._ema_state[0:50] = 0.0

        return self._ema_state.copy()

    def reset(self):
        self._ema_state = None
        self._last_valid = None
        self._lost_count = 0


def extract_video(video_path, extractor, yolo, hand_detector,
                  target_fps=15.0) -> dict:
    """Extract features from a single video"""
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Sampling: extract target_fps frames uniformly
    n_target = max(1, int(duration * target_fps))
    target_frame_indices = np.linspace(0, total_frames - 1, n_target, dtype=np.int32)
    target_frame_indices = sorted(set(target_frame_indices.tolist()))
    n_feat_frames = len(target_frame_indices)

    features = np.zeros((n_feat_frames, 90), dtype=np.float32)
    timestamps = np.zeros(n_feat_frames, dtype=np.float32)
    frame_indices = np.zeros(n_feat_frames, dtype=np.int32)
    feat_idx = 0

    yolo.reset_tracking()
    extractor.reset()
    hand_detector.palm_velocity.clear()

    fc = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fc += 1
        timestamp = (fc - 1) / video_fps

        # YOLO detection every frame (for tracking stability)
        detections = yolo.detect(frame)
        hands = hand_detector.detect(frame)

        # Box state
        box_bbox = yolo.get_box_bbox(detections)
        box_state, _ = yolo.get_box_state(detections)

        # Interaction for hand selection
        if box_bbox:
            interaction = extractor.compute_interaction(detections, hands, box_bbox, h, w)
        else:
            interaction = {"hand_obj_iou": np.zeros((2,5), dtype=np.float32),
                          "hand_obj_dist": np.zeros((2,5), dtype=np.float32)}

        # Select active hand
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

                    # Hand touch
                    oi = OBJ_TO_IDX.get(det.cls_name, -1)
                    touched = False
                    if oi >= 0:
                        hio = interaction["hand_obj_iou"]
                        for hi in range(min(2, hio.shape[0])):
                            if hio[hi, oi] > 0.04:
                                touched = True
                                break
                    yolo.update_hand_touch(det.track_id, touched)

        # Extract feature if this is a target frame
        if fc - 1 in target_frame_indices and feat_idx < n_feat_frames:
            feat = extractor.extract(frame, detections, hands, box_bbox, box_state)
            features[feat_idx] = feat
            timestamps[feat_idx] = timestamp
            frame_indices[feat_idx] = fc - 1
            feat_idx += 1

    cap.release()

    # Trim to actual extracted frames
    features = features[:feat_idx]
    timestamps = timestamps[:feat_idx]
    frame_indices = frame_indices[:feat_idx]

    return {
        "features": features,
        "timestamps": timestamps,
        "frame_indices": frame_indices,
        "video_fps": video_fps,
        "feature_fps": target_fps,
        "total_video_frames": total_frames,
        "n_feature_frames": feat_idx,
        "duration": duration,
        "width": w,
        "height": h,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-type", choices=["ok", "wr", "yuanzi", "all"],
                        default="all")
    parser.add_argument("--video-name", type=str, default=None,
                        help="Single video name (e.g. ok_1)")
    parser.add_argument("--yolo", default=None)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--ema-alpha", type=float, default=0.5)
    args = parser.parse_args()

    yolo_path = args.yolo or str(MODELS_DIR / "yolo_final_v1.pt")
    video_types = ["ok", "wr"] if args.video_type == "all" else [args.video_type]

    print(f"[Init] YOLO (imgsz={args.imgsz}) + MediaPipe...")
    yolo = YOLODetector(yolo_path, conf_thresh=args.conf, device=args.device,
                        imgsz=args.imgsz)
    hand_detector = HandDetector()
    extractor = FeatureExtractorV2Standalone(yolo, hand_detector,
                                             ema_alpha=args.ema_alpha)

    total_videos = 0
    for vtype in video_types:
        vdir = RAW_VIDEOS_DIR / vtype
        if not vdir.exists():
            print(f"  Skip {vtype}: dir not found")
            continue

        # Collect videos
        if args.video_name:
            vp = vdir / f"{args.video_name}.avi"
            videos = [(args.video_name, vp)] if vp.exists() else []
        else:
            import re
            videos = sorted(
                [(f.stem, f) for f in vdir.glob("*.avi")],
                key=lambda x: int(re.search(r'(\d+)', x[0]).group(1))
            )
        if not videos:
            continue

        out_dir = FEAT_OUT_DIR / vtype
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Processing] {vtype}: {len(videos)} videos")

        for vi, (vname, vpath) in enumerate(videos):
            t0 = time.time()
            try:
                result = extract_video(vpath, extractor, yolo, hand_detector,
                                       target_fps=args.target_fps)
            except Exception as e:
                print(f"  [{vi+1}/{len(videos)}] {vname}: ERROR {e}")
                continue

            elapsed = time.time() - t0

            # Save npz
            out_path = out_dir / f"{vname}.npz"
            np.savez_compressed(
                out_path,
                features=result["features"],
                timestamps=result["timestamps"],
                frame_indices=result["frame_indices"],
                video_name=vname,
                video_type=vtype,
                video_fps=result["video_fps"],
                feature_fps=result["feature_fps"],
                feature_version="v2_90",
                feature_dim=90,
            )

            print(f"  [{vi+1:3d}/{len(videos)}] {vname}: "
                  f"{result['n_feature_frames']:4d}f "
                  f"({result['duration']:.1f}s@{result['video_fps']:.0f}fps) "
                  f"{elapsed:.1f}s")

            total_videos += 1

    # Save feature meta
    meta = {
        "feature_dim": 90,
        "feature_order": [
            "hand_keypoints(0-41)",
            "hand_global(42-49)",
            "box_state(50-56)",
            "obj_interact_earphone(57-65)",
            "obj_interact_charger(66-74)",
            "obj_interact_green_bag(75-83)",
            "obj_velocity(84-89)",
        ],
        "feature_layout": {
            "hand_keypoints": [0, 42, "21 landmarks × (x,y) relative to palm center"],
            "hand_global": [42, 8, "detected,openness,palm_x,palm_y,palm_vx,palm_vy,bbox_w,bbox_h"],
            "box_state": [50, 7, "open_prob,closed_prob,inner_cx,inner_cy,inner_w,inner_h,inner_conf"],
            "obj_interact": [57, 27, "3 objects × 9: conf,cx,cy,w,h,in_box_ratio,dist_to_palm,touch_iou,in_init_region"],
            "obj_velocity": [84, 6, "3 objects × 2: vx,vy"],
        },
        "roi_config": "full frame normalized (w,h)",
        "normalization_method": "hand_keypoints: palm-relative + hand_scale; rest: /w or /h",
        "feature_fps": args.target_fps,
        "ema_alpha": args.ema_alpha,
        "object_order": ["earphone", "charger", "green_bag"],
        "label_mapping_version": "v2",
        "total_videos_processed": total_videos,
    }
    meta_path = FEAT_OUT_DIR / "feature_meta_v2_90.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nMeta saved to: {meta_path}")
    print(f"Total videos processed: {total_videos}")


if __name__ == "__main__":
    main()
