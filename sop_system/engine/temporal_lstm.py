"""
时序动作识别模型 — 从帧特征序列识别当前原子动作
- SOPActionLSTM: BiLSTM 逐帧序列标注
- SOPActionGRU: BiGRU 逐帧序列标注
- SOPActionTCN: Causal TCN 在线推理
- MultiScalePredictor: 多尺度窗口投票

输入: (B, T, D) 特征序列
输出: (B, T, 7) 逐帧步骤概率 (0=背景,1-5=步骤,6=完成)
"""
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import List, Tuple


class SOPActionLSTM(nn.Module):
    """BiLSTM 逐帧序列标注 — 每帧输出一个动作标签"""

    def __init__(self, input_size: int = 130, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)                     # (B, T, H*2)
        return self.classifier(lstm_out)                # (B, T, num_classes)


class SOPActionGRU(nn.Module):
    """BiGRU 逐帧序列标注 — 每帧输出一个动作标签"""

    def __init__(self, input_size: int = 130, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, seq_len, num_classes) — 逐帧预测
        """
        gru_out, _ = self.gru(x)                     # (B, T, H*2)
        return self.classifier(gru_out)               # (B, T, num_classes)


class CausalTCNBlock(nn.Module):
    """因果膨胀卷积块 — 只看过去帧"""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding: only left side
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                               padding=pad, padding_mode='zeros')
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation,
                               padding=pad, padding_mode='zeros')
        self.norm = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        # x: (B, C, T)
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        if out.shape[-1] > residual.shape[-1]:
            out = out[..., :residual.shape[-1]]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        if out.shape[-1] > residual.shape[-1]:
            out = out[..., :residual.shape[-1]]
        out = self.norm(out)
        out = self.relu(out + residual)
        out = self.dropout(out)
        return out


class SOPActionTCN(nn.Module):
    """因果TCN — 在线推理友好，多层膨胀卷积扩大感受野"""

    def __init__(self, input_size: int = 130, num_classes: int = 7,
                 hidden_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.input_proj = nn.Conv1d(input_size, hidden_dim, 1)

        dilations = [1, 2, 4, 8, 16, 32]
        layers = []
        for d in dilations:
            layers.append(CausalTCNBlock(hidden_dim, hidden_dim, dilation=d, dropout=dropout))
        self.tcn = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, seq_len, num_classes) — 逐帧预测
        """
        x_t = x.transpose(1, 2)                       # (B, D, T)
        out = self.input_proj(x_t)
        out = self.tcn(out)
        out = self.classifier(out)                     # (B, num_classes, T)
        return out.transpose(1, 2)                     # (B, T, num_classes)


class MultiScalePredictor:
    """多尺度窗口投票 — 兼容逐帧输出模型"""

    def __init__(self, model: nn.Module, device: str = "cuda",
                 windows: List[int] = None, vote_threshold: float = 0.5):
        self.model = model
        self.device = device
        self.windows = windows or [64, 96, 128]
        self.vote_threshold = vote_threshold
        self.model.eval()

    def predict(self, feature_buffer: List[np.ndarray]) -> Tuple[int, float, List[Tuple[int, float]]]:
        """多尺度投票预测当前步骤"""
        if len(feature_buffer) < min(self.windows):
            return (0, 0.0, [])

        features = np.stack(list(feature_buffer), axis=0)  # (T, D)
        all_preds = []
        all_confs = []

        for win_size in self.windows:
            if len(features) < win_size:
                continue

            seq = features[-win_size:]
            # 均匀重采样到win_size帧
            indices = np.linspace(0, len(seq) - 1, win_size, dtype=int)
            seq = seq[indices]

            x = torch.FloatTensor(seq).unsqueeze(0).to(self.device)  # (1, T, D)
            with torch.no_grad():
                logits = self.model(x)  # (1, T, num_classes)
                # 取最后一帧的预测
                probs = torch.softmax(logits[0, -1], dim=0).cpu().numpy()

            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            all_preds.append(pred)
            all_confs.append(conf)

        if not all_preds:
            return (0, 0.0, [])

        votes = Counter(all_preds)
        most_common = votes.most_common(1)[0]
        winner = most_common[0]
        winner_confs = [c for p, c in zip(all_preds, all_confs) if p == winner]
        avg_conf = np.mean(winner_confs) if winner_confs else 0.0

        combined_conf = np.zeros(7)
        counts = np.zeros(7)
        for p, c in zip(all_preds, all_confs):
            combined_conf[p] += c
            counts[p] += 1
        for i in range(7):
            if counts[i] > 0:
                combined_conf[i] /= counts[i]
        top3 = sorted([(i, float(combined_conf[i])) for i in range(7)
                       if combined_conf[i] > 0], key=lambda x: -x[1])[:3]

        return (winner, avg_conf, top3)


class FeatureExtractor:
    """帧级特征提取器: YOLO + MediaPipe → 130维特征向量

    [  0- 83] 手关键点 (2手×21点×2坐标)           84维
    [ 84- 88] 物体存在性 (5类)                      5维
    [ 89- 98] 手物距离 (2手×5类)                   10维
    [ 99-108] 手物IoU (2手×5类)                    10维
    [109-118] 物体bbox (5类×2: cx/w,cy/h)         10维
    [119-123] 物体置信度 (5类)                      5维
    [124-125] 手张开度 (2手)                        2维
    [126-127] 盒子状态 (one-hot open/closed)        2维
    [128-129] 手活跃+手近盒标志                      2维
    """

    FINGERTIPS = [4, 8, 12, 16, 20]

    def __init__(self, yolo_detector, hand_detector):
        self.yolo = yolo_detector
        self.hand_detector = hand_detector
        self.obj_names = ["box_closed", "box_open", "earphone", "charger", "green_bag"]

    def extract(self, frame: np.ndarray, detections: List,
                hands: List, interaction: dict, box_bbox, box_state: str,
                fingertip: bool = False) -> np.ndarray:
        h, w = frame.shape[:2]
        det_dict = {d.cls_name: d for d in detections}

        obj_presence = np.zeros(5, dtype=np.float32)
        for i, name in enumerate(self.obj_names):
            if name in det_dict:
                obj_presence[i] = 1.0

        box_feat = np.zeros(2, dtype=np.float32)
        if box_state == "open":
            box_feat[0] = 1.0
        elif box_state == "closed":
            box_feat[1] = 1.0

        hand_openness = np.array([h.openness for h in hands[:2]], dtype=np.float32)
        if len(hand_openness) < 2:
            hand_openness = np.pad(hand_openness, (0, 2 - len(hand_openness)))

        hand_active = np.array([
            1.0 if len(hands) > 0 else 0.0,
            1.0 if interaction.get("hands_active", False) else 0.0,
        ], dtype=np.float32)

        hand_obj_dist = interaction.get("hand_obj_dist",
                                        np.zeros((2, 5), dtype=np.float32)).flatten()
        hand_obj_iou = interaction.get("hand_obj_iou",
                                       np.zeros((2, 5), dtype=np.float32)).flatten()

        obj_bbox = np.zeros(10, dtype=np.float32)
        obj_conf = np.zeros(5, dtype=np.float32)
        for i, name in enumerate(self.obj_names):
            d = det_dict.get(name)
            if d:
                obj_bbox[i * 2] = d.center[0] / w
                obj_bbox[i * 2 + 1] = d.center[1] / h
                obj_conf[i] = d.confidence

        if fingertip:
            hand_feat = np.zeros(20, dtype=np.float32)
            for hi, hand in enumerate(hands[:2]):
                for fi, lm_idx in enumerate(self.FINGERTIPS):
                    if lm_idx < len(hand.landmarks):
                        hand_feat[hi * 10 + fi * 2] = hand.landmarks[lm_idx, 0]
                        hand_feat[hi * 10 + fi * 2 + 1] = hand.landmarks[lm_idx, 1]
            return np.concatenate([
                hand_feat,       # 20
                hand_active,     # 1 (just one flag for fingertip mode)
            ]).astype(np.float32)
        else:
            hand_feat = np.zeros(84, dtype=np.float32)
            for hi, hand in enumerate(hands[:2]):
                for li in range(min(21, len(hand.landmarks))):
                    hand_feat[hi * 42 + li * 2] = hand.landmarks[li, 0]
                    hand_feat[hi * 42 + li * 2 + 1] = hand.landmarks[li, 1]

            return np.concatenate([
                hand_feat,       # 84
                obj_presence,    # 5
                hand_obj_dist,   # 10
                hand_obj_iou,    # 10
                obj_bbox,        # 10
                obj_conf,        # 5
                hand_openness,   # 2
                box_feat,        # 2
                hand_active,     # 2
            ]).astype(np.float32)


class FeatureExtractorV2:
    """帧级特征提取器 V2 — 90维手-物-盒交互特征 (文档对齐)

    [  0- 41] 主动手关键点 (21点×2坐标)              42维
    [ 42- 49] 手部全局状态 (detected,conf,palm_xy,palm_vxy,bbox_wh) 8维
    [ 50- 56] 纸盒状态 (open_prob,closed_prob,inner_cx,inner_cy,inner_w,inner_h,inner_conf) 7维
    [ 57- 83] 三个物体交互特征 (3×9)                 27维
              每个物体: conf,cx,cy,w,h,in_box_ratio,dist_to_palm,touch_iou,in_init_region
    [ 84- 89] 三个物体速度 (3×2)                       6维
    -------------------------------------------------
    总计                                             90维
    """

    OBJ_NAMES = ["box_closed", "box_open", "earphone", "charger", "green_bag"]
    INTERACT_OBJS = ["earphone", "charger", "green_bag"]  # S2-S4的目标物体
    OBJ_TO_IDX = {"earphone": 2, "charger": 3, "green_bag": 4}

    def __init__(self, yolo_detector, hand_detector):
        self.yolo = yolo_detector
        self.hand = hand_detector

    def extract(self, frame: np.ndarray, detections: List,
                hands: List, interaction: dict, box_bbox, box_state: str,
                current_step: int = 0) -> np.ndarray:
        """
        Args:
            frame: BGR图像
            detections: YOLO检测结果
            hands: MediaPipe手部
            interaction: compute_interaction()返回
            box_bbox: 盒子bbox
            box_state: 'open'/'closed'/'unknown'
            current_step: 当前SOP步骤 (用于选择目标物体)

        Returns:
            np.ndarray (90,) float32
        """
        h, w = frame.shape[:2]
        det_dict = {d.cls_name: d for d in detections}

        # ── 1. 主动手选择与特征 (42维) ──
        active_hand = self.hand.get_active_hand(hands)
        hand_feat = np.zeros(42, dtype=np.float32)

        if active_hand is not None:
            # 手掌中心 (wrist + MCP均值)
            palm_indices = [0, 5, 9, 13, 17]
            palm_x = float(np.mean([active_hand.landmarks[i, 0] for i in palm_indices]))
            palm_y = float(np.mean([active_hand.landmarks[i, 1] for i in palm_indices]))
            hand_scale = max(0.01, active_hand.openness * 0.5 + 0.1)

            for li in range(min(21, len(active_hand.landmarks))):
                hand_feat[li * 2] = (active_hand.landmarks[li, 0] - palm_x) / hand_scale
                hand_feat[li * 2 + 1] = (active_hand.landmarks[li, 1] - palm_y) / hand_scale

        # ── 2. 手部全局状态 (8维) ──
        hand_global = np.zeros(8, dtype=np.float32)
        if active_hand is not None:
            palm_indices = [0, 5, 9, 13, 17]
            palm_x = float(np.mean([active_hand.landmarks[i, 0] for i in palm_indices]))
            palm_y = float(np.mean([active_hand.landmarks[i, 1] for i in palm_indices]))
            handedness = active_hand.handedness
            vx, vy = self.hand.palm_velocity.get(handedness, (0.0, 0.0))

            hand_global[0] = 1.0  # hand_detected
            hand_global[1] = active_hand.openness
            hand_global[2] = palm_x
            hand_global[3] = palm_y
            hand_global[4] = vx
            hand_global[5] = vy
            hand_global[6] = (active_hand.bbox[2] - active_hand.bbox[0]) / w
            hand_global[7] = (active_hand.bbox[3] - active_hand.bbox[1]) / h

        # ── 3. 纸盒状态 (7维) ──
        box_feat = np.zeros(7, dtype=np.float32)
        # open/closed probabilities
        box_open_det = det_dict.get("box_open")
        box_closed_det = det_dict.get("box_closed")
        box_feat[0] = box_open_det.confidence if box_open_det else 0.0
        box_feat[1] = box_closed_det.confidence if box_closed_det else 0.0
        # box_inner (use open box or closed box bbox)
        active_box = box_open_det or box_closed_det
        if active_box and box_bbox:
            box_feat[2] = (box_bbox[0] + box_bbox[2]) / (2 * w)  # cx
            box_feat[3] = (box_bbox[1] + box_bbox[3]) / (2 * h)  # cy
            box_feat[4] = (box_bbox[2] - box_bbox[0]) / w        # w
            box_feat[5] = (box_bbox[3] - box_bbox[1]) / h        # h
            box_feat[6] = active_box.confidence                    # conf

        # ── 4. 三个物体的交互特征 (27维) ──
        obj_interact = np.zeros(27, dtype=np.float32)
        hand_obj_iou = interaction.get("hand_obj_iou",
                                        np.zeros((2, 5), dtype=np.float32))
        hand_obj_dist = interaction.get("hand_obj_dist",
                                         np.zeros((2, 5), dtype=np.float32))

        # 手掌中心像素坐标
        palm_cx_px, palm_cy_px = 0.0, 0.0
        if active_hand is not None:
            palm_indices = [0, 5, 9, 13, 17]
            palm_cx_px = float(np.mean([active_hand.landmarks[i, 0] for i in palm_indices])) * w
            palm_cy_px = float(np.mean([active_hand.landmarks[i, 1] for i in palm_indices])) * h

        for oi, obj_name in enumerate(self.INTERACT_OBJS):
            base = oi * 9
            d = det_dict.get(obj_name)
            if d:
                obj_interact[base + 0] = d.confidence                                # conf
                obj_interact[base + 1] = d.center[0] / w                             # cx
                obj_interact[base + 2] = d.center[1] / h                             # cy
                obj_interact[base + 3] = (d.bbox[2] - d.bbox[0]) / w                 # w
                obj_interact[base + 4] = (d.bbox[3] - d.bbox[1]) / h                 # h
                # in_box_ratio
                if box_bbox:
                    obj_interact[base + 5] = self.yolo.compute_in_box_ratio(d.bbox, box_bbox)
                # dist_to_palm
                if active_hand is not None and palm_cx_px > 0:
                    obj_interact[base + 6] = 1.0 / (1.0 + np.sqrt(
                        (d.center[0] - palm_cx_px) ** 2 + (d.center[1] - palm_cy_px) ** 2
                    ) / w * 10)
                # touch_iou
                class_idx = self.OBJ_TO_IDX.get(obj_name, -1)
                if class_idx >= 0 and hand_obj_iou.shape[0] > 0:
                    hi = min(self.hand.active_hand_idx, hand_obj_iou.shape[0] - 1)
                    if hi >= 0:
                        obj_interact[base + 7] = hand_obj_iou[hi, class_idx]
                # in_init_region (from tracked object)
                tracked_list = self.yolo.get_tracked_by_name(obj_name)
                if tracked_list:
                    obj_interact[base + 8] = 1.0 if tracked_list[0].in_init_region else 0.0

        # ── 5. 三个物体的速度 (6维) ──
        obj_velocity = np.zeros(6, dtype=np.float32)
        for oi, obj_name in enumerate(self.INTERACT_OBJS):
            tracked_list = self.yolo.get_tracked_by_name(obj_name)
            if tracked_list:
                tobj = tracked_list[0]
                obj_velocity[oi * 2] = tobj.velocity[0] / max(w, 1)
                obj_velocity[oi * 2 + 1] = tobj.velocity[1] / max(h, 1)

        return np.concatenate([
            hand_feat,      # 42
            hand_global,    # 8
            box_feat,       # 7
            obj_interact,   # 27
            obj_velocity,   # 6
        ]).astype(np.float32)
