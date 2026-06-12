"""
SOP实时动作检测系统 — 统一入口
支持: 视频文件检测 / 实时相机检测 / 训练模式
"""
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from collections import deque
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from engine.yolo_detector import YOLODetector
from engine.hand_detector import HandDetector
from engine.physical_state import PhysicalStateEngine
from engine.sop_fsm import SOPStateMachine
from engine.temporal_lstm import MultiScalePredictor, FeatureExtractor, SOPActionLSTM, SOPActionGRU
from engine.fusion import FusionEngine
from engine.sop_fsm import SOPStateMachine
from utils.draw import (draw_detections, draw_hand, draw_step_bar, draw_info_panel)


class SOPRealtimeDetector:
    """SOP实时检测器 — 集成所有模块"""

    def __init__(self, yolo_model_path: str, lstm_model_path: str = None,
                 device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"设备: {self.device}")

        # 模块1: YOLO
        print(f"加载YOLO: {yolo_model_path}")
        self.yolo = YOLODetector(yolo_model_path, device=self.device)

        # 模块2: MediaPipe手部
        self.hand_det = HandDetector()

        # 模块3: 物理状态引擎
        self.physical = PhysicalStateEngine(
            confirm_frames=CONFIRM_FRAMES,
        )

        # 模块4: FSM
        self.fsm = SOPStateMachine(
            timeout=STATE_TIMEOUT,
            min_step_duration=MIN_STEP_DURATION,
        )

        # 模块5: LSTM (可选)
        self.lstm = None
        self.feature_extractor = None
        self.feature_buffer = deque(maxlen=LSTM_SEQ_LEN * 3)
        if lstm_model_path and Path(lstm_model_path).exists():
            print(f"加载LSTM: {lstm_model_path}")
            self._load_lstm(lstm_model_path)
        else:
            print("LSTM未加载 (仅使用物理状态+FSM)")

        # 模块6: 融合决策
        self.fusion = FusionEngine(
            confirm_count=CONFIRM_FRAMES,
        )

        self.frame_count = 0
        self.start_time = time.time()

    def _load_lstm(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]

        # Auto-detect model type
        if "lstm.weight_ih_l0" in state_dict:
            model = SOPActionLSTM(
                input_size=checkpoint.get("input_size", LSTM_INPUT_SIZE),
                hidden_size=checkpoint.get("hidden_size", LSTM_HIDDEN_SIZE),
                num_layers=checkpoint.get("num_layers", LSTM_NUM_LAYERS),
                num_classes=checkpoint.get("num_classes", LSTM_NUM_CLASSES),
                dropout=checkpoint.get("dropout", LSTM_DROPOUT),
            ).to(self.device)
        elif "gru.weight_ih_l0" in state_dict:
            model = SOPActionGRU(
                input_size=checkpoint.get("input_size", 21),
                hidden_size=checkpoint.get("hidden_size", 48),
                num_layers=checkpoint.get("num_layers", 1),
                num_classes=checkpoint.get("num_classes", LSTM_NUM_CLASSES),
                dropout=checkpoint.get("dropout", 0.6),
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {list(state_dict.keys())[:5]}")
        model.load_state_dict(state_dict)
        model.eval()
        self.feature_mode = "fingertip" if checkpoint.get("input_size", 130) == 21 else "full"
        self.lstm = MultiScalePredictor(model, self.device, MULTI_SCALE_WINDOWS)
        self.feature_extractor = FeatureExtractor(self.yolo, self.hand_det)

    def process_frame(self, frame: np.ndarray) -> dict:
        """处理单帧, 返回检测结果"""
        timestamp = time.time()
        self.frame_count += 1

        # 1. YOLO检测
        detections = self.yolo.detect(frame)

        # 2. 手部检测(MediaPipe)
        hands = self.hand_det.detect(frame)

        # 3. 解析YOLO检测结果
        box_state, box_conf = self.yolo.get_box_state(detections)
        box_bbox = self.yolo.get_box_bbox(detections)
        objects_in_box = self.yolo.get_objects_in_box(detections, box_bbox)

        # 4. 手物交互
        interaction = self.hand_det.compute_interaction(hands, detections,
                                                        box_bbox, frame.shape[:2])

        # 5. 物理状态更新
        hand_near = any(interaction["hand_box_dist"] < 0.3)
        phys_result = self.physical.update(
            detections, box_state, box_bbox,
            hand_near, interaction["hands_active"],
            hand_obj_iou=interaction["hand_obj_iou"],
            hand_box_dist=interaction["hand_box_dist"],
        )

        # 6. LSTM预测 (如果加载了)
        lstm_step, lstm_conf, lstm_top3 = 0, 0.0, []
        if self.feature_extractor is not None:
            feature = self.feature_extractor.extract(
                frame, detections, hands, interaction, box_bbox, box_state,
                fingertip=(self.feature_mode == "fingertip"))
            self.feature_buffer.append(feature)

            if len(self.feature_buffer) >= min(MULTI_SCALE_WINDOWS):
                lstm_step, lstm_conf, lstm_top3 = self.lstm.predict(
                    self.feature_buffer)

        # 7. FSM验证 (使用物理步骤)
        fsm_result = self.fsm.validate(
            phys_result.current_phys_step,
            max(lstm_conf, 0.5),
            not phys_result.is_error,
            timestamp,
        )

        # 8. 融合决策 (模型+物理→步骤，不含FSM序列验证)
        fusion_result = self.fusion.update(
            phys_result, lstm_step, lstm_conf, lstm_top3, timestamp
        )

        # 9. 构建返回结果
        det_dict = {}
        for name in ["box_closed", "box_open", "earphone", "charger", "green_bag"]:
            det = self.yolo.get_best_detection(detections, name)
            det_dict[name] = {"bbox": det.bbox, "conf": det.confidence} if det else None

        fps = self.frame_count / max(1, timestamp - self.start_time)

        return {
            "frame_id": self.frame_count,
            "fps": fps,
            "detections": det_dict,
            "hands": hands,
            "box_state": box_state,
            "box_bbox": box_bbox,
            "objects_in_box": objects_in_box,
            "step": fsm_result.step_id,
            "step_name": fsm_result.step_name,
            "is_correct": fsm_result.is_correct,
            "message": fsm_result.message,
            "error_type": fsm_result.error_type,
            "progress": fsm_result.progress,
            "has_error": fsm_result.has_error,
            "fusion_step": fusion_result.step,
            "fusion_conf": fusion_result.confidence,
            "lstm_step": lstm_step,
            "lstm_conf": lstm_conf,
            "lstm_top3": lstm_top3,
        }

    def reset(self):
        self.physical.reset()
        self.fsm.reset()
        self.fusion.reset()
        self.feature_buffer.clear()
        self.frame_count = 0
        self.start_time = time.time()


class _DrawableDet:
    """Lightweight adapter so result dict entries work with draw_detections."""
    __slots__ = ('cls_name', 'confidence', 'bbox')
    def __init__(self, name, d):
        self.cls_name = name
        self.confidence = d["conf"]
        self.bbox = d["bbox"]


def _make_det_list(det_dict):
    return [_DrawableDet(name, d) for name, d in det_dict.items() if d]


def _draw_result(frame, result):
    vis = frame.copy()
    if result["detections"]:
        vis = draw_detections(vis, _make_det_list(result["detections"]))
    vis = draw_hand(vis, result["hands"])
    vis = draw_step_bar(vis, result["step"], result["step_name"],
                       result["progress"], not result["is_correct"])
    return vis


def run_on_video(detector: SOPRealtimeDetector, video_path: str,
                 output_path: str = None, show: bool = True):
    """在视频文件上运行检测"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    print(f"视频: {video_path} ({w}x{h}, {total}帧)")
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        vis = _draw_result(frame, result)
        vis = draw_info_panel(vis, result)

        if out:
            out.write(vis)
        if show:
            cv2.imshow("SOP Realtime Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                detector.reset()

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"处理完成: {elapsed:.1f}s, 平均 {detector.frame_count/elapsed:.1f} FPS")


def run_on_camera(detector: SOPRealtimeDetector, camera_id: int = 0):
    """在本地相机上实时检测"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开相机: {camera_id}")
        return

    print(f"实时检测中... 按 'q' 退出, 'r' 重置")
    print("="*60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        vis = _draw_result(frame, result)

        # 控制台输出
        if detector.frame_count % 15 == 0:
            print(f"  [{detector.frame_count:4d}] {result['step_name']:12s} | "
                  f"{result['message'][:50]}")

        cv2.imshow("SOP Realtime Detection", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            detector.reset()

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SOP实时动作检测系统")
    parser.add_argument("--mode", choices=["video", "camera", "train-yolo", "train-lstm"],
                       default="video", help="运行模式")
    parser.add_argument("--yolo-model", default="models/yolo_final_v1.pt",
                       help="YOLO模型路径")
    parser.add_argument("--lstm-model", help="LSTM模型路径")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--camera", type=int, default=0, help="相机ID")
    parser.add_argument("--output", help="输出视频路径")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--data", default="/home/zhaowei/shabi/data/yolo_dataset_v2",
                       help="训练数据目录")
    args = parser.parse_args()

    print("="*60)
    print("SOP实时动作检测系统")
    print("动作序列: 开盒 → 放耳机 → 放插头 → 放绿袋 → 关盒")
    print("="*60)

    if args.mode == "train-yolo":
        from training.train_yolo import train
        data_yaml = Path(args.data) / "dataset.yaml"
        train(str(data_yaml), device=args.device)

    elif args.mode == "train-lstm":
        print("LSTM训练需要在服务器上运行:")
        print(f"  python training/train_sequence_v5.py --features-dir ... --yolo-model {args.yolo_model}")

    elif args.mode == "video":
        if not args.video:
            parser.error("--video required for video mode")
        detector = SOPRealtimeDetector(args.yolo_model, args.lstm_model, args.device)
        run_on_video(detector, args.video, args.output)

    elif args.mode == "camera":
        detector = SOPRealtimeDetector(args.yolo_model, args.lstm_model, args.device)
        run_on_camera(detector, args.camera)


if __name__ == "__main__":
    main()
