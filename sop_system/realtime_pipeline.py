"""
SOP Real-Time Action Detection Pipeline
========================================
YOLO+ByteTrack → MediaPipe → 特征提取 → 时序模型 → 事件检测 → FSM验证

架构 (文档对齐):
  1. YOLO + ByteTrack + 物体轨迹追踪
  2. MediaPipe Hands + 主动手选择
  3. FeatureExtractor: 手-物-盒交互特征
  4. 时序模型(BiGRU/TCN): 逐帧动作概率
  5. 事件检测: 模型+物理+轨迹 多条件确认
  6. SOPStateMachine: 序列验证 + 错误诊断

Usage:
  python realtime_pipeline.py                          # webcam
  python realtime_pipeline.py --video ok_1.avi         # video file
"""
import cv2
import numpy as np
import torch
import time
import sys
import argparse
from pathlib import Path
from collections import deque, Counter
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))

from engine.yolo_detector import YOLODetector, Detection, TrackedObject
from engine.hand_detector import HandDetector, HandInfo
from engine.physical_state import PhysicalStateEngine, PhysicalStateResult
from engine.sop_fsm import SOPStateMachine, FSMResult
from engine.temporal_lstm import SOPActionGRU, SOPActionTCN, FeatureExtractor

# ── Colors (BGR) ──
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
GRAY = (128, 128, 128)

STEP_COLORS = {
    0: GRAY, 1: (60, 60, 255), 2: (60, 255, 60),
    3: (60, 180, 255), 4: (220, 80, 255), 5: (255, 180, 60), 6: GREEN,
}

CLASS_COLORS = {
    "box_closed": (100, 100, 100), "box_open": (0, 200, 200),
    "earphone": (255, 100, 100), "charger": (100, 255, 100),
    "green_bag": (100, 200, 100),
}

STEP_NAMES = {
    0: "等待", 1: "S1-开盒", 2: "S2-耳机",
    3: "S3-插头", 4: "S4-绿袋", 5: "S5-关盒", 6: "完成",
}

# 期望的物品顺序（S2→S3→S4 对应的目标物体）
STEP_TARGET_OBJECT = {2: "earphone", 3: "charger", 4: "green_bag"}


@dataclass
class SOPResult:
    """统一检测结果 — 供显示使用"""
    step: int
    step_name: str
    is_correct: bool
    message: str
    error_type: str
    progress: float
    confidence: float
    has_error: bool
    model_step: int = 0
    model_conf: float = 0.0
    model_top3: List = field(default_factory=list)
    phys_step: int = 0
    box_is_open: bool = False
    box_state: str = "unknown"
    visible_objects: List = field(default_factory=list)
    objects_placed: List = field(default_factory=list)


class SOPRealtimePipeline:
    """SOP实时检测流水线 — 事件检测 + FSM序列验证"""

    def __init__(self, yolo_path: str, model_path: str,
                 conf_thresh: float = 0.35, device: str = "cuda"):
        self.device = device

        # ── 1. YOLO + ByteTrack + 轨迹追踪 ──
        print("[1/5] YOLO + ByteTrack + Trajectory...")
        self.yolo = YOLODetector(yolo_path, conf_thresh=conf_thresh, device=device)

        # ── 2. MediaPipe Hands ──
        print("[2/5] MediaPipe Hands...")
        self.hand_detector = HandDetector()

        # ── 3. Temporal model ──
        print("[3/5] Temporal model...")
        self.temporal_model, self.input_size = self._load_temporal(model_path)
        self.feature_extractor = FeatureExtractor(self.yolo, self.hand_detector)
        self.feature_version = "v1_130d"
        print(f"   Feature version: {self.feature_version}")
        print(f"   Model input dim: {self.input_size}")
        print(f"   Model path: {model_path}")

        # ── 4. Physical state engine ──
        print("[4/5] Physical state engine...")
        self.physical = PhysicalStateEngine(confirm_frames=6)

        # ── 5. SOP State Machine ──
        self.fsm = SOPStateMachine(timeout=30.0, min_step_duration=0.4)
        print("[5/5] Ready!")

        # ── Pipeline state ──
        self.feature_buffer = deque(maxlen=160)
        self.model_interval = 4      # run model every N frames
        self.frame_idx = 0
        self.last_result: Optional[SOPResult] = None
        self.error_flash = 0
        self.good_flash = 0
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()

        # Event detection smoothing
        self._model_history: deque = deque(maxlen=12)
        self._running_max = 0
        self._ema_model_probs = np.zeros(7, dtype=np.float32)  # EMA smoothed per-class probs
        self._emitted_events: set = set()  # 事件 one-shot: 已触发的步骤
        self._debug_interval = 10  # 每N帧打印一次详细日志

    def _load_temporal(self, path: str) -> Tuple[torch.nn.Module, int]:
        """Load temporal model. Auto-detect architecture from state_dict."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state_dict"]
        input_size = ckpt.get("input_size", 130)
        num_classes = ckpt.get("num_classes", 7)
        num_layers = ckpt.get("num_layers", 3)
        hidden_size = ckpt.get("hidden_size", 256)

        if "classifier.0.weight" in state_dict:
            fc0_out, fc0_in = state_dict["classifier.0.weight"].shape
            hidden_size = fc0_in // 2

            if "gru.weight_ih_l0" in state_dict:
                model = SOPActionGRU(input_size=input_size, hidden_size=hidden_size,
                                     num_layers=num_layers, num_classes=num_classes,
                                     dropout=0.45).to(self.device)
                print(f"   BiGRU: input={input_size}, hidden={hidden_size}, layers={num_layers}")
            else:
                model = self._make_old_gru_compat(input_size, hidden_size, num_classes)
                print(f"   GRU (compat): input={input_size}, hidden={hidden_size}")
        elif "tcn.0.conv1.weight" in state_dict:
            model = SOPActionTCN(input_size=input_size, num_classes=num_classes).to(self.device)
            print(f"   TCN: input={input_size}")
        else:
            raise ValueError(f"Unknown model: {list(state_dict.keys())[:5]}")

        model.load_state_dict(state_dict)
        model.eval()
        return model, input_size

    def _make_old_gru_compat(self, input_size, hidden_size, num_classes):
        """Create old window-classification GRU for backward compat."""
        class OldGRUCompat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = torch.nn.GRU(input_size, hidden_size, num_layers=2,
                                        batch_first=True, bidirectional=True, dropout=0.4)
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size * 2, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(32, num_classes),
                )

            def forward(self, x):
                gru_out, _ = self.gru(x)
                pooled = gru_out.mean(dim=1)
                return self.classifier(pooled)

        return OldGRUCompat().to(self.device)

    def reset(self):
        self.yolo.reset_tracking()
        self.physical.reset()
        self.fsm.reset()
        self.feature_buffer.clear()
        self.frame_idx = 0
        self.last_result = None
        self.error_flash = 0
        self.good_flash = 0
        self._model_history.clear()
        self._running_max = 0
        self._ema_model_probs = np.zeros(7, dtype=np.float32)
        self._emitted_events.clear()

    def process_frame(self, frame: np.ndarray, timestamp: float = None
                      ) -> Tuple[np.ndarray, SOPResult]:
        if timestamp is None:
            timestamp = time.time()

        h, w = frame.shape[:2]
        self.frame_idx += 1

        # ── 1. YOLO + ByteTrack + 轨迹更新 ──
        detections = self.yolo.detect(frame)

        # ── 2. MediaPipe hands ──
        hands = self.hand_detector.detect(frame)

        # ── 3. Box state ──
        box_state, box_conf = self.yolo.get_box_state(detections)
        box_bbox = self.yolo.get_box_bbox(detections)

        # ── 4. Hand-object interaction ──
        interaction = self.hand_detector.compute_interaction(
            hands, detections, box_bbox, (h, w))
        hand_near_box = any(d < 0.15 for d in interaction.get("hand_box_dist", [999, 999]))

        # ── 5. 更新物体轨迹的入盒状态和手接触状态 ──
        self._update_trajectory_states(detections, box_bbox, interaction)

        # ── 6. Physical state engine (传入轨迹状态) ──
        phys_result = self.physical.update(
            detections, box_state, box_bbox, hand_near_box, len(hands) > 0,
            hand_obj_iou=interaction.get("hand_obj_iou"),
            holding_objects=[],
            hand_box_dist=interaction.get("hand_box_dist"),
            tracked_objects=self.yolo.tracked_objects)

        # ── 7. 主动手选择 + 特征提取 ──
        current_step = self.fsm.current_step.value if self.fsm.current_step else 0
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

        # ── 8. Temporal model inference ──
        model_step, model_conf, model_top3 = 0, 0.0, []
        if self.frame_idx % self.model_interval == 0 and len(self.feature_buffer) >= 16:
            model_step, model_conf, model_top3 = self._run_model()

        # ── 9. Event detection: 模型+物理+轨迹 → detected_step ──
        detected_step, detected_conf, phys_ok = self._detect_event(
            phys_result, model_step, model_conf, model_top3,
            box_bbox, timestamp)

        # ── 10. FSM sequence validation ──
        fsm_result = self.fsm.validate(
            detected_step, detected_conf, phys_ok, timestamp)

        # ── 11. Build unified result ──
        prev_step = self.last_result.step if self.last_result else -1
        result = SOPResult(
            step=fsm_result.step_id,
            step_name=fsm_result.step_name,
            is_correct=fsm_result.is_correct,
            message=fsm_result.message,
            error_type=fsm_result.error_type,
            progress=fsm_result.progress,
            confidence=detected_conf,
            has_error=fsm_result.has_error,
            model_step=model_step,
            model_conf=model_conf,
            model_top3=model_top3,
            phys_step=phys_result.current_phys_step,
            box_is_open=phys_result.box_is_open,
            box_state=box_state,
            visible_objects=phys_result.visible_objects,
            objects_placed=list(phys_result.objects_placed),
        )

        if result.step != prev_step and result.step < 6:
            print(f"  [f{self.frame_idx}] S{prev_step}→S{result.step} "
                  f"({result.step_name}) model={model_step}:{model_conf:.2f} "
                  f"box={box_state} placed={result.objects_placed} "
                  f"{'ERR:' + result.error_type if result.error_type else ''}")

        # 定期详细调试日志
        if self.frame_idx % self._debug_interval == 0:
            self._debug_log(phys_result, model_step, model_conf, model_top3, result)

        if result.has_error:
            self.error_flash = 25
        elif result.step != prev_step and result.is_correct:
            self.good_flash = 25

        self.last_result = result

        # ── 12. Draw ──
        display = self._draw_overlay(frame, detections, hands, result,
                                      box_state, box_bbox, interaction)

        now = time.time()
        self.fps_buffer.append(1.0 / max(now - self.last_time, 0.001))
        self.last_time = now

        return display, result

    def _update_trajectory_states(self, detections: List[Detection],
                                   box_bbox: Optional[Tuple], interaction: dict):
        """同步 YOLODetector 中 TrackedObject 的入盒/手接触状态"""
        if box_bbox is None:
            return

        hand_obj_iou = interaction.get("hand_obj_iou",
                                        np.zeros((2, 5), dtype=np.float32))
        obj_name_to_idx = {"earphone": 2, "charger": 3, "green_bag": 4}

        for det in detections:
            if det.track_id < 0 or det.cls_name in ("box_open", "box_closed"):
                continue

            # 入盒判定：物体中心点是否在盒子bbox内
            cx, cy = det.center
            bx1, by1, bx2, by2 = box_bbox
            in_box = (bx1 <= cx <= bx2 and by1 <= cy <= by2)

            # 同时计算bbox入盒率
            in_box_ratio = self.yolo.compute_in_box_ratio(det.bbox, box_bbox)

            self.yolo.update_in_box_status(det.track_id, in_box or in_box_ratio > 0.3)

            # 手接触判定：手与物体的IoU
            touched = False
            oi = obj_name_to_idx.get(det.cls_name, -1)
            if oi >= 0:
                for hi in range(min(2, hand_obj_iou.shape[0])):
                    if hand_obj_iou[hi, oi] > 0.04:
                        touched = True
                        break
            self.yolo.update_hand_touch(det.track_id, touched)

    def _run_model(self) -> Tuple[int, float, list]:
        """Run temporal model on feature buffer, average last 8 frames."""
        features = np.stack(list(self.feature_buffer), axis=0)

        # 维度硬校验: 特征维度必须与模型输入维度一致
        feat_dim = features.shape[-1]
        assert feat_dim == self.input_size, (
            f"FEATURE/MODEL DIM MISMATCH: feature={feat_dim} ({self.feature_version}) "
            f"!= model_input={self.input_size}. Use FeatureExtractorV2 only with 90d model."
        )

        seq_len = min(128, len(features))
        seq = features[-seq_len:]
        x = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.temporal_model(x)  # (1, T, 7)
            probs = torch.softmax(logits[0], dim=-1)  # (T, 7)
            last_probs = probs[-8:] if probs.shape[0] >= 8 else probs
            avg_probs = last_probs.mean(dim=0).cpu().numpy()

        # EMA smooth the probabilities
        alpha = 0.4
        self._ema_model_probs = alpha * avg_probs + (1 - alpha) * self._ema_model_probs

        pred = int(np.argmax(self._ema_model_probs))
        conf = float(self._ema_model_probs[pred])
        top3 = sorted([(i, float(self._ema_model_probs[i])) for i in range(7)
                      if self._ema_model_probs[i] > 0.02], key=lambda x: -x[1])[:3]
        return pred, conf, top3

    def _detect_event(self, phys_result: PhysicalStateResult,
                      model_step: int, model_conf: float, model_top3: List,
                      box_bbox: Optional[Tuple], timestamp: float
                      ) -> Tuple[int, float, bool]:
        """
        事件检测 — 融合模型预测 + 物理状态 + 轨迹状态，输出当前步骤。

        文档对齐的多条件确认逻辑:
          S1 (open_box):     盒子打开 + 稳定 ≥ 模型阈值
          S2-S4 (put_obj):   模型预测 + 物体轨迹确认(离开初始区 + 入盒)
          S5 (close_box):    盒子关闭 + 三物全部入盒
        """
        # 平滑模型预测
        self._model_history.append(model_step)
        smoothed_step = 0
        if len(self._model_history) >= 3:
            smoothed_step = Counter(self._model_history).most_common(1)[0][0]
        if smoothed_step > self._running_max:
            self._running_max = smoothed_step

        phys_step = phys_result.current_phys_step
        phys_ok = not phys_result.is_error

        # ── 规则0: 错误物品入盒检测 (每个事件都评估, 不只expected) ──
        if phys_result.wrong_placement and phys_result.wrong_placement_frames >= 3:
            wrong_obj = phys_result.wrong_placement
            # 映射错误物品到其正确的步骤号 → FSM会发现这是跳步/乱序
            obj_to_step = {"earphone": 2, "charger": 3, "green_bag": 4}
            wrong_step = obj_to_step.get(wrong_obj, 0)
            if wrong_step > 0:
                expected_obj = STEP_TARGET_OBJECT.get(phys_step + 1, "")
                print(f"  [Event] WRONG OBJECT: {wrong_obj} (S{wrong_step}) "
                      f"expected {expected_obj} (S{phys_step + 1})")
                return (wrong_step, 0.6, False)  # phys_ok=False → FSM detects error

        # ── 规则1: 盒子关闭 → S5 ──
        if phys_result.box_is_closed:
            # 检查三物是否全部入盒
            all_placed = len(phys_result.objects_placed) >= 3
            if all_placed or phys_step >= 5:
                return (5, max(model_conf if model_step == 5 else 0.7, phys_result.box_state_conf), True)
            else:
                # 提前关闭 — 通知 FSM 判定
                return (5, phys_result.box_state_conf, True)

        # ── 规则2: 盒子打开 + idle → S1 ──
        if phys_result.box_is_open and phys_step <= 1:
            if model_step == 1 and model_conf > 0.4:
                return (1, model_conf, True)
            if phys_step == 1:
                return (1, 0.7, True)

        # ── 规则3: 盒子打开期间 S2-S4 — 模型信号 + 物体轨迹确认 ──
        if phys_result.box_is_open and 1 <= phys_step <= 4:
            # 检查当前步骤对应的目标物体状态
            target_obj = STEP_TARGET_OBJECT.get(phys_step + 1, None)  # 期望的下一个物体
            trajectory_ok = True

            if target_obj:
                tracked_list = self.yolo.get_tracked_by_name(target_obj)
                if tracked_list:
                    tobj = tracked_list[0]
                    # 物体离开初始区域 + 入盒稳定
                    trajectory_ok = (not tobj.in_init_region) or (tobj.stable_in_box_frames >= 3)

            # 模型信号
            model_ready = (self._running_max >= phys_step + 1 and model_conf > 0.3)
            # 模型Top3里有目标步骤也算
            model_hints_target = any(s == phys_step + 1 and c > 0.15 for s, c in model_top3)

            if model_ready or model_hints_target:
                return (phys_step + 1, max(model_conf, 0.5), phys_ok and trajectory_ok)
            # 物理步骤本身就是强信号
            if phys_step > self._running_max:
                return (phys_step, 0.6, phys_ok)

        # ── 规则4: 默认 — 保持物理步骤 ──
        step = max(phys_step, self._running_max)
        step = max(0, min(5, step))
        return (step, model_conf if model_step == step else 0.4, phys_ok)

    def _debug_log(self, phys_result, model_step, model_conf, model_top3, result):
        """详细调试日志 — 每N帧输出一次"""
        stages = phys_result.placement_stages
        stable = phys_result.stable_frames
        stage_str = " | ".join(
            f"{obj}={stages.get(obj, '?').name}(s{stable.get(obj,0)})"
            for obj in ["earphone", "charger", "green_bag"]
        )
        print(f"  [DEBUG f{self.frame_idx}] "
              f"expected=S{self.fsm.current_step.value} "
              f"event=S{result.step} "
              f"model=S{model_step}:{model_conf:.2f} "
              f"top3={[(s,round(c,2)) for s,c in model_top3[:3]]} "
              f"box={'open' if phys_result.box_is_open else 'closed' if phys_result.box_is_closed else '?'} "
              f"phys_step={phys_result.current_phys_step} "
              f"placed={phys_result.objects_placed} "
              f"wrong={phys_result.wrong_placement or '-'} "
              f"stages=[{stage_str}] "
              f"fsm={'OK' if result.is_correct else 'ERR:'+result.error_type}")

    def _draw_overlay(self, frame, detections, hands, result: SOPResult,
                      box_state, box_bbox, interaction):
        h, w = frame.shape[:2]
        display = frame.copy()

        for det in detections:
            if det.tracked and det.confidence < 0.3:
                continue
            color = CLASS_COLORS.get(det.cls_name, GRAY)
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = "%s %.2f" % (det.cls_name, det.confidence)
            cv2.putText(display, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if box_bbox is not None:
            bx1, by1, bx2, by2 = map(int, box_bbox)
            cv2.rectangle(display, (bx1, by1), (bx2, by2), CYAN, 3)
            cv2.putText(display, "BOX:%s" % box_state, (bx1, by1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 2)

        for hand in hands:
            for lm in hand.landmarks:
                px, py = int(lm[0] * w), int(lm[1] * h)
                cv2.circle(display, (px, py), 3, GREEN, -1)
            hx1, hy1, hx2, hy2 = hand.bbox
            cv2.rectangle(display, (hx1, hy1), (hx2, hy2), GREEN, 1)

        # ── 顶部错误/正确闪烁条 ──
        if self.error_flash > 0:
            self.error_flash -= 1
            err_bar = np.zeros((60, w, 3), dtype=np.uint8)
            err_bar[:] = (0, 0, 200)
            cv2.putText(err_bar, "✗ ERROR: %s" % result.error_type, (20, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            display = np.vstack([err_bar, display])
        elif self.good_flash > 0:
            self.good_flash -= 1
            ok_bar = np.zeros((60, w, 3), dtype=np.uint8)
            ok_bar[:] = (0, 140, 0)
            cv2.putText(ok_bar, "✓ CORRECT: %s" % result.step_name, (20, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            display = np.vstack([ok_bar, display])

        # ── 右侧面板 ──
        panel_w = 300
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (25, 25, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX

        y = 10
        # 状态指示器
        is_correct = result.is_correct and not result.has_error
        status_text = "CORRECT" if is_correct else ("ERROR" if result.has_error else "OK")
        status_color = GREEN if is_correct else (RED if result.has_error else YELLOW)
        cv2.rectangle(panel, (10, y), (panel_w - 10, y + 40),
                     (0, 50, 0) if is_correct else (50, 0, 0), -1)
        cv2.putText(panel, status_text, (panel_w // 2 - 80, y + 30), font, 1.0, status_color, 3)
        y += 50

        fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        cv2.putText(panel, "FPS: %.1f" % fps, (15, y + 12), font, 0.35, GRAY, 1)
        y += 22

        step = result.step
        step_color = GREEN if result.is_correct else RED
        cv2.putText(panel, "Step %d: %s" % (step, STEP_NAMES.get(step, "?")),
                   (15, y + 22), font, 0.7, step_color, 2)
        cv2.putText(panel, "Conf: %.2f" % result.confidence, (15, y + 42), font, 0.4, WHITE, 1)
        y += 55

        # 进度条
        cv2.putText(panel, "Progress", (15, y + 15), font, 0.4, WHITE, 1)
        y += 20
        bar_w = panel_w - 30
        for i in range(1, 6):
            cy = y + (i - 1) * 22
            filled = i <= step
            bar_color = STEP_COLORS[i] if filled else (50, 50, 50)
            cv2.rectangle(panel, (15, cy), (15 + bar_w, cy + 17), bar_color, -1)
            status = "OK" if filled else ("..." if i == step + 1 else "")
            step_name_list = ["Open Box", "Earphone", "Charger", "Green Bag", "Close Box"]
            cv2.putText(panel, "S%d %s %s" % (i, step_name_list[i-1], status),
                       (20, cy + 13), font, 0.35, (0, 0, 0) if filled else GRAY, 1)
        y += 120

        # 模型预测
        if result.model_top3:
            cv2.putText(panel, "Model Top3:", (15, y + 15), font, 0.35, GRAY, 1)
            y += 18
            for s, c in result.model_top3[:3]:
                bar_w2 = int((panel_w - 60) * c)
                cv2.rectangle(panel, (15, y), (15 + bar_w2, y + 14), STEP_COLORS.get(s, GRAY), -1)
                cv2.putText(panel, "S%d %.2f" % (s, c), (18, y + 11), font, 0.3, (255, 255, 255), 1)
                y += 16
            y += 5

        cv2.putText(panel, "Box: %s" % box_state, (15, y + 15), font, 0.35, CYAN, 1)
        y += 20
        if result.visible_objects:
            cv2.putText(panel, "Visible:", (15, y + 15), font, 0.33, WHITE, 1)
            y += 16
            for obj in result.visible_objects:
                cv2.putText(panel, "  " + obj, (15, y + 13), font, 0.3, YELLOW, 1)
                y += 14
        if result.objects_placed:
            y += 4
            cv2.putText(panel, "Placed: %s" % ", ".join(result.objects_placed),
                       (15, y + 13), font, 0.3, GREEN, 1)
            y += 14

        y = h - 50
        cv2.putText(panel, "Space=Pause  R=Reset  Q=Quit", (10, y + 15), font, 0.35, GRAY, 1)

        display = np.hstack([display, panel])

        # 底部信息条
        bar = np.zeros((30, display.shape[1], 3), dtype=np.uint8)
        bar[:] = (20, 20, 20)
        # FSM message
        cv2.putText(bar, result.message[:100], (10, 20), font, 0.4, WHITE, 1)
        prog_w = w - 40
        cv2.rectangle(bar, (10, 25), (10 + prog_w, 28), (60, 60, 60), -1)
        progress = max(0.0, min(1.0, result.progress))
        filled = int(prog_w * progress)
        if filled > 0:
            cv2.rectangle(bar, (10, 25), (10 + filled, 28),
                         RED if result.has_error else GREEN, -1)
        display = np.vstack([display, bar])

        return display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", default="models/yolo_final_v1.pt")
    parser.add_argument("--model", default="models/best_sequence_v5.pt")
    parser.add_argument("--video", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hikvision", action="store_true",
                       help="Use Hikvision MVS SDK instead of OpenCV")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    yolo_path = Path(args.yolo) if Path(args.yolo).is_absolute() else base_dir / args.yolo
    model_path = Path(args.model) if Path(args.model).is_absolute() else base_dir / args.model

    print("=" * 60)
    print("SOP Real-Time Detection Pipeline v2")
    print("  YOLO+Track → MediaPipe → Feature → Model → Event → FSM")
    print("=" * 60)
    print("  YOLO: %s  %s" % (yolo_path, "OK" if yolo_path.exists() else "MISSING"))
    print("  Model: %s  %s" % (model_path, "OK" if model_path.exists() else "MISSING"))

    if not yolo_path.exists():
        print("ERROR: YOLO model not found!")
        sys.exit(1)

    pipeline = SOPRealtimePipeline(str(yolo_path), str(model_path),
                                    conf_thresh=args.conf, device=args.device)

    # ── 打开视频源 ──
    cam = None
    is_video = args.video is not None
    if is_video:
        cap = cv2.VideoCapture(args.video)
        name = Path(args.video).name
    elif args.hikvision:
        from camera.hikvision_camera import open_camera
        ok, cam = open_camera(args.camera, prefer_hikvision=True,
                             width=args.width, height=args.height)
        if not ok:
            print("ERROR: Cannot open camera! Is MVS client closed?")
            print("Try: 1) Close MVS client  2) Replug USB cable")
            sys.exit(1)
        name = "Hikvision-%d" % args.camera
    else:
        from camera.opencv_camera import OpenCVCamera
        cam = OpenCVCamera(args.camera, width=args.width, height=args.height)
        if not cam.open():
            print("ERROR: Cannot open camera %d!" % args.camera)
            sys.exit(1)
        name = "Camera-%d" % args.camera

    if cam is not None:
        fps = cam.get_fps()
    else:
        if not cap.isOpened():
            print("ERROR: Cannot open video source!")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output_video, fourcc, fps,
                                      (args.width + 320, args.height + 30))

    def read_frame():
        if cam is not None:
            return cam.read()
        else:
            return cap.read()

    if args.headless:
        max_frames = args.max_frames or (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not cam else 99999)
        last_step = -1
        fc = 0
        while fc < max_frames:
            ret, frame = read_frame()
            if not ret:
                break
            fc += 1
            frame = cv2.resize(frame, (args.width, args.height))
            display, result = pipeline.process_frame(frame)
            if result.step != last_step:
                last_step = result.step
                print("  f%5d | Step %d %s conf=%.2f err=%s" % (
                    fc, result.step, result.step_name,
                    result.confidence, result.error_type or "-"))
            if out_writer:
                out_writer.write(display)
        final = pipeline.last_result
        if final:
            print("\nDone! Final: step=%d (%s)" % (final.step, final.step_name))
    else:
        win = "SOP Detection v2 - " + name
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, args.width + 320, args.height + 30)
        running = False
        paused = False
        display = None
        print("\nPress 'S' to start SOP detection  Q=Quit\n")

        while True:
            ret, frame = read_frame()
            if not ret:
                break
            frame = cv2.resize(frame, (args.width, args.height))

            if not running:
                display = frame.copy()
                h_disp, w_disp = display.shape[:2]
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w_disp, h_disp), (20, 20, 20), -1)
                display = cv2.addWeighted(display, 0.5, overlay, 0.5, 0)
                cv2.putText(display, "Press 'S' to Start", (w_disp // 2 - 230, h_disp // 2 - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 3)
                cv2.putText(display, "SOP Detection System v2", (w_disp // 2 - 230, h_disp // 2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
                cv2.putText(display, "Q=Quit", (20, h_disp - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1)
                cv2.imshow(win, display)

            elif paused:
                if display is not None:
                    h_disp, w_disp = display.shape[:2]
                    cv2.putText(display, "PAUSED", (w_disp // 2 - 80, h_disp // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
                    cv2.imshow(win, display)

            else:
                display, result = pipeline.process_frame(frame)
                cv2.imshow(win, display)
                if out_writer:
                    out_writer.write(display)

            key = cv2.waitKey(1)
            if key in (ord('q'), 27):
                break
            elif key == ord('s') and not running:
                running = True
                paused = False
                pipeline.reset()
                print("\n=== SOP Detection Started ===\n")
            elif key == ord('r') and running:
                pipeline.reset()
                print("\n=== Reset ===\n")
            elif key == 32 and running:
                paused = not paused
                if paused:
                    print("--- Paused ---")

        cv2.destroyAllWindows()

    if out_writer:
        out_writer.release()
    if cam is not None:
        cam.release()
    else:
        cap.release()


if __name__ == "__main__":
    main()
