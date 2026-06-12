"""
用新YOLO重新提取所有视频的130维特征 → features_v5
使用features_v4的标签（手工标注的原子动作起止时间）

采样: stride=2 (每2帧取1帧)
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.yolo_detector import YOLODetector
from engine.hand_detector import HandDetector
from engine.temporal_lstm import FeatureExtractor

VIDEO_DIR = Path("/home/zhaowei/shabi/data")
FEAT_V4_DIR = Path("/home/zhaowei/shabi/data/features_v4")
FEAT_V5_DIR = Path("/home/zhaowei/shabi/data/features_v5")
MODEL_PATH = "/home/zhaowei/shabi/data/runs/detect/final_v1/weights/best.pt"
STRIDE = 2

OBJ_NAMES = ["box_closed", "box_open", "earphone", "charger", "green_bag"]


def compute_interaction(detections, hands, h, w, box_bbox=None):
    """和pipeline中_compute_interaction相同的逻辑"""
    det_dict = {d.cls_name: d for d in detections}
    hand_obj_dist = np.zeros((2, 5), dtype=np.float32)
    hand_obj_iou = np.zeros((2, 5), dtype=np.float32)
    hand_box_dist = np.ones(2, dtype=np.float32) * 999
    hands_active = False

    for hi, hand in enumerate(hands[:2]):
        if hand.bbox is None:
            continue
        hx1, hy1, hx2, hy2 = hand.bbox
        h_area = max(1, (hx2 - hx1) * (hy2 - hy1))

        for oi, oname in enumerate(OBJ_NAMES):
            d = det_dict.get(oname)
            if d is None:
                continue
            ox, oy = d.center
            dist = np.sqrt((hand.center[0] * w - ox) ** 2 +
                          (hand.center[1] * h - oy) ** 2) / w
            hand_obj_dist[hi, oi] = 1.0 / (1.0 + dist * 10)

            ox1, oy1, ox2, oy2 = d.bbox
            ix1, iy1 = max(hx1, ox1), max(hy1, oy1)
            ix2, iy2 = min(hx2, ox2), min(hy2, oy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                o_area = max(1, (ox2 - ox1) * (oy2 - oy1))
                union = h_area + o_area - inter
                hand_obj_iou[hi, oi] = inter / union if union > 0 else 0

            if dist < 0.15:
                hands_active = True

        if box_bbox is not None:
            bx_c = ((box_bbox[0] + box_bbox[2]) / 2, (box_bbox[1] + box_bbox[3]) / 2)
            hd = np.sqrt((hand.center[0] * w - bx_c[0]) ** 2 +
                        (hand.center[1] * h - bx_c[1]) ** 2) / w
            hand_box_dist[hi] = hd

    return {
        "hand_obj_dist": hand_obj_dist,
        "hand_obj_iou": hand_obj_iou,
        "hand_box_dist": hand_box_dist,
        "hands_active": hands_active,
    }


def get_box_state_and_bbox(detections):
    """从YOLO检测获取盒子状态和bbox"""
    det_dict = {d.cls_name: d for d in detections}
    box_open = det_dict.get("box_open")
    box_closed = det_dict.get("box_closed")

    box_state = "unknown"
    box_bbox = None

    if box_open and box_closed:
        if box_open.confidence > box_closed.confidence:
            box_state, box_bbox = "open", box_open.bbox
        else:
            box_state, box_bbox = "closed", box_closed.bbox
    elif box_open:
        box_state, box_bbox = "open", box_open.bbox
    elif box_closed:
        box_state, box_bbox = "closed", box_closed.bbox

    return box_state, box_bbox


def process_video(video_path, yolo_detector, hand_detector, feat_extractor, v4_labels):
    """处理单个视频，返回features和labels"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    features = []
    frame_indices = []
    fi = 0
    sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fi % STRIDE != 0:
            fi += 1
            continue

        frame_indices.append(fi)
        h, w = frame.shape[:2]

        # YOLO检测
        detections = yolo_detector.detect(frame)
        box_state, box_bbox = get_box_state_and_bbox(detections)

        # 手部检测
        hands = hand_detector.detect(frame)

        # 交互特征
        interaction = compute_interaction(detections, hands, h, w, box_bbox)

        # 130维特征
        feat = feat_extractor.extract(
            frame, detections, hands, interaction, box_bbox, box_state
        )
        features.append(feat)
        sampled += 1
        fi += 1

    cap.release()

    features = np.array(features, dtype=np.float32)

    # 对齐标签：v4_labels可能和采样帧数不同
    if v4_labels is not None:
        n_v4 = len(v4_labels)
        n_new = len(features)

        if n_v4 == n_new:
            labels = v4_labels
        elif n_v4 == total_frames:
            # v4标签是逐帧的，按stride采样
            labels = v4_labels[frame_indices]
        else:
            # 长度不同，按比例对齐
            indices = np.linspace(0, n_v4 - 1, n_new, dtype=int)
            labels = v4_labels[indices]
    else:
        labels = np.zeros(len(features), dtype=np.int64)

    return features, labels, sampled


def main():
    FEAT_V5_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    yolo = YOLODetector(MODEL_PATH, conf_thresh=0.25, device="cuda:0")
    hand_detector = HandDetector()
    feat_extractor = FeatureExtractor(yolo, hand_detector)

    # 收集所有视频
    all_videos = []
    for sub in ["yuanzi", "ok", "wr"]:
        vdir = VIDEO_DIR / sub
        if vdir.exists():
            for vp in sorted(vdir.glob("*.avi")):
                all_videos.append((sub, vp))

    print(f"Total videos: {len(all_videos)}")

    for sub, vp in all_videos:
        name = vp.stem
        out_path = FEAT_V5_DIR / f"{name}.npz"

        # 加载v4标签
        v4_labels = None
        if sub in ("ok", "wr"):
            y_file = FEAT_V4_DIR / f"{name}_y_seg.npy"
        else:
            # yuanzi: {n}_y.npy
            y_file = FEAT_V4_DIR / f"{name}_y.npy"

        if y_file.exists():
            v4_labels = np.load(y_file)
            print(f"  {name}: v4 labels shape={v4_labels.shape}", end="")
        else:
            print(f"  {name}: NO v4 labels!", end="")

        # 提取特征
        features, labels, n_frames = process_video(
            vp, yolo, hand_detector, feat_extractor, v4_labels
        )

        # 保存
        np.savez_compressed(out_path, X=features, y=labels)
        print(f" → features={features.shape}, labels={labels.shape}")

        # 打印标签分布
        for lbl in range(7):
            cnt = int(np.sum(labels == lbl))
            if cnt > 0:
                print(f"      label {lbl}: {cnt}")

    print(f"\nDone! Features saved to {FEAT_V5_DIR}")


if __name__ == "__main__":
    main()
