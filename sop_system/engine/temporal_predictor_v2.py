"""
TemporalPredictorV2 — v2_90 TCN+BiGRU 实时推理包装器 (方案B修复版)

关键修复 (vs 旧版):
  - get_frame_prediction(): 未覆盖帧返回 None，禁止 fallback 到未来帧
  - EMA 只在成熟帧(有窗口覆盖的帧)上按时间顺序更新
  - 新增 get_latest_mature_prediction(): 返回最新成熟帧的预测
  - 新增 is_frame_mature(): 检查帧是否已有窗口覆盖

模型约束:
  - 只加载 v2_90 模型 (input_dim=90, T=48)
  - center-frame 输出，首帧成熟延迟 ≈ half_w/15 ≈ 0.8s
  - T=48 窗口延迟 ≈ 24/15 ≈ 1.6s (非实时帧，是历史帧)

Usage:
  predictor = TemporalPredictorV2("best.pt", device="cuda")
  predictor.reset()
  for feat in feature_stream:
      result = predictor.predict(feat)  # returns dict or None
      pred = predictor.get_latest_mature_prediction()  # latest reliable prediction
"""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class TemporalPredictorV2:
    """v2_90 TCN+BiGRU 实时推理包装器 (方案B修复版)

    关键语义:
      - 窗口 [t-47, t] 的预测对应中心帧 t-24
      - 只有被至少一个窗口覆盖的帧才是"成熟帧"
      - 未覆盖帧返回 None，不制造假预测
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 T: int = 48, stride: int = 4, ema_alpha: float = 0.35):
        self.T = T
        self.stride = stride
        self.ema_alpha = ema_alpha
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half_w = T // 4  # 12 frames — center region of each window

        # ── Load model ──
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        input_dim = ckpt.get("input_dim", None)
        assert input_dim == 90, (
            f"WRONG MODEL: input_dim={input_dim}, expected 90. "
            f"Refusing to load old model."
        )

        self.input_dim = input_dim
        self.num_classes_step = ckpt["num_classes_step"]
        self.num_classes_verb = ckpt["num_classes_verb"]
        self.num_classes_target = ckpt["num_classes_target"]

        # Import model class
        from train_temporal_v2 import TCNBiGRU
        self.model = TCNBiGRU(
            input_dim=input_dim,
            hidden_dim=ckpt["hidden_dim"],
            num_classes_step=self.num_classes_step,
            num_classes_verb=self.num_classes_verb,
            num_classes_target=self.num_classes_target,
            dropout=ckpt.get("dropout", 0.3),
            num_gru_layers=ckpt.get("num_gru_layers", 2),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"[TemporalPredictorV2] Loaded: input_dim={input_dim}, "
              f"T={T}, stride={stride}, center-frame "
              f"(first mature ~{self.half_w / 15:.1f}s @ 15fps, "
              f"total latency ~{self.T // 2 / 15:.1f}s)")

        # ── State ──
        self._buffer = []           # accumulated feature frames
        self._last_infer_end = 0    # last window start index processed
        self._frame_probs = {}      # frame_idx -> sum of probs (np.array)
        self._frame_counts = {}     # frame_idx -> count of contributing windows
        self._ema_state = None      # per-class EMA state (np.array, shape [num_classes])
        self._last_ema_frame = -1   # last frame index that updated EMA
        self._total_frames = 0
        self._last_pred = None      # latest window prediction dict

    def reset(self):
        self._buffer.clear()
        self._last_infer_end = 0
        self._frame_probs.clear()
        self._frame_counts.clear()
        self._ema_state = None
        self._last_ema_frame = -1
        self._total_frames = 0
        self._last_pred = None

    def predict(self, feature: np.ndarray) -> Optional[Dict]:
        """Add one feature frame, run new windows. Returns latest window result.

        Returns None while buffering (first T-1 frames).
        """
        self._buffer.append(feature.astype(np.float32))
        self._total_frames += 1

        n = len(self._buffer)
        latest_result = None

        for start in range(self._last_infer_end, n - self.T + 1, self.stride):
            end = start + self.T
            if end > n:
                break
            probs_step, probs_verb, probs_target = self._infer_window(start, end)

            # Assign to center region: [start + half_w, end - half_w)
            assign_start = start + self.half_w
            assign_end = end - self.half_w

            for fi in range(assign_start, assign_end):
                if 0 <= fi < self._total_frames:
                    self._frame_probs[fi] = (
                        self._frame_probs.get(fi, np.zeros(self.num_classes_step, dtype=np.float64))
                        + probs_step
                    )
                    self._frame_counts[fi] = self._frame_counts.get(fi, 0) + 1

            self._last_infer_end = start + self.stride

            latest_result = {
                "step_probs": probs_step,
                "verb_probs": probs_verb,
                "target_probs": probs_target,
                "center_frame": start + self.T // 2,
            }

        if latest_result is not None:
            self._last_pred = latest_result

        return latest_result

    def is_frame_mature(self, frame_idx: int) -> bool:
        """Check if frame has been covered by at least one window."""
        return frame_idx in self._frame_probs and self._frame_counts.get(frame_idx, 0) > 0

    def get_frame_prediction(self, frame_idx: int) -> Optional[Dict]:
        """Get EMA-smoothed prediction for a specific frame.

        方案B修复: 未覆盖帧返回 None，不 fallback。
        EMA 只对已覆盖帧按时间顺序更新。
        """
        # ── 修复点1: 未覆盖帧直接返回 None ──
        if not self.is_frame_mature(frame_idx):
            return None

        # Average overlapping window predictions
        probs = (
            self._frame_probs[frame_idx]
            / max(self._frame_counts[frame_idx], 1)
        )

        # ── 修复点2: 因果EMA，只对已到达的帧更新 ──
        # Update EMA for any skipped frames between last update and current
        if self._ema_state is None:
            # First mature frame: initialize EMA with its probs
            self._ema_state = probs.copy()
            self._last_ema_frame = frame_idx
        elif frame_idx > self._last_ema_frame:
            # Update EMA for gaps (frames that were never updated)
            for gap_fi in range(self._last_ema_frame + 1, frame_idx + 1):
                if self.is_frame_mature(gap_fi):
                    gap_probs = (
                        self._frame_probs[gap_fi]
                        / max(self._frame_counts[gap_fi], 1)
                    )
                else:
                    # Uncovered frame in EMA chain: skip, keep previous state
                    # (this is correct — we don't want fake data in EMA)
                    gap_probs = None

                if gap_probs is not None:
                    self._ema_state = (
                        self.ema_alpha * gap_probs
                        + (1.0 - self.ema_alpha) * self._ema_state
                    )

            self._last_ema_frame = frame_idx

        pred = int(np.argmax(self._ema_state))
        conf = float(np.max(self._ema_state))
        top3_idx = np.argsort(self._ema_state)[::-1][:3]

        return {
            "step": pred,
            "confidence": conf,
            "step_probs": self._ema_state.copy(),
            "top3": [(int(i), float(self._ema_state[i])) for i in top3_idx],
        }

    def get_latest_mature_prediction(self) -> Optional[Dict]:
        """Get prediction for the most recent mature (covered) frame.

        This is the primary method for real-time FSM consumption.
        """
        # Find the latest frame that has window coverage
        for fi in range(self._total_frames - 1, -1, -1):
            if self.is_frame_mature(fi):
                return self.get_frame_prediction(fi)
        return None

    def get_mature_prediction_for_frame(self, frame_idx: int) -> Optional[Dict]:
        """Get prediction for a specific frame, waiting until it's mature.

        Returns None if frame is not yet mature (no window has covered it).
        """
        return self.get_frame_prediction(frame_idx)

    def _infer_window(self, start: int, end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run model on buffer[start:end] window."""
        window = np.stack(self._buffer[start:end], axis=0)  # [T, D]
        x = torch.FloatTensor(window).unsqueeze(0).to(self.device)  # [1, T, D]

        with torch.inference_mode():
            step_logits, verb_logits, target_logits = self.model(x)
            step_probs = torch.softmax(step_logits, dim=1).cpu().numpy()[0]
            verb_probs = torch.softmax(verb_logits, dim=1).cpu().numpy()[0]
            target_probs = torch.softmax(target_logits, dim=1).cpu().numpy()[0]

        return step_probs, verb_probs, target_probs

    # ── Properties ──

    @property
    def is_ready(self) -> bool:
        """True when at least one frame has mature prediction."""
        return len(self._frame_probs) > 0

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def latest_prediction(self) -> Optional[Dict]:
        return self._last_pred

    @property
    def first_mature_frame(self) -> int:
        """Index of the first frame that has window coverage."""
        if not self._frame_probs:
            return -1
        return min(self._frame_probs.keys())

    @property
    def latest_mature_frame(self) -> int:
        """Index of the most recent frame that has window coverage."""
        if not self._frame_probs:
            return -1
        return max(self._frame_probs.keys())

    @property
    def model_info(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "T": self.T,
            "stride": self.stride,
            "num_classes_step": self.num_classes_step,
            "num_classes_verb": self.num_classes_verb,
            "num_classes_target": self.num_classes_target,
            "device": str(self.device),
            "fix_version": "方案B — no fallback, causal EMA",
        }
