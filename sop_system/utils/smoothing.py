"""
帧间平滑工具 — 指数滑动平均 (EMA) 和缓冲区确认

用于平滑 YOLO 检测、手部关键点、入盒比例等抖动信号。
"""
import numpy as np
from collections import deque
from typing import Any, List


class EMASmoother:
    """指数滑动平均 (Exponential Moving Average) 平滑器"""

    def __init__(self, alpha: float = 0.3, init_value: float = 0.0):
        self.alpha = alpha
        self.value = init_value
        self.initialized = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self, value: float = 0.0):
        self.value = value
        self.initialized = False


class EMASmoother1D:
    """向量版EMA平滑器"""

    def __init__(self, size: int, alpha: float = 0.3):
        self.alpha = alpha
        self.value = np.zeros(size, dtype=np.float32)
        self.initialized = False

    def update(self, x: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.value = x.astype(np.float32)
            self.initialized = True
        else:
            self.value = self.alpha * x.astype(np.float32) + (1 - self.alpha) * self.value
        return self.value.copy()

    def reset(self):
        self.value = np.zeros_like(self.value)
        self.initialized = False


class ConfirmationBuffer:
    """确认缓冲区 — 连续N帧中目标状态占比超过阈值才确认"""

    def __init__(self, size: int = 10, threshold: float = 0.7):
        self.buffer: deque = deque(maxlen=size)
        self.size = size
        self.threshold = threshold

    def update(self, state: bool) -> bool:
        """返回是否确认"""
        self.buffer.append(state)
        if len(self.buffer) < self.size:
            return False
        return sum(self.buffer) / self.size >= self.threshold

    def reset(self):
        self.buffer.clear()


class StateChangeDetector:
    """状态变化检测器 — 带迟滞和最小持续时间"""

    def __init__(self, confirm_frames: int = 5, cooldown_frames: int = 10):
        self.confirm_frames = confirm_frames
        self.cooldown_frames = cooldown_frames
        self._current_state: Any = None
        self._candidate_state: Any = None
        self._candidate_count = 0
        self._cooldown = 0

    def update(self, new_state: Any) -> tuple:
        """返回 (当前状态, 是否刚变化)"""
        if self._cooldown > 0:
            self._cooldown -= 1

        if new_state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = new_state
            self._candidate_count = 1

        changed = False
        if (self._candidate_count >= self.confirm_frames
                and new_state != self._current_state
                and self._cooldown == 0):
            self._current_state = new_state
            self._cooldown = self.cooldown_frames
            changed = True

        return self._current_state, changed

    def reset(self):
        self._current_state = None
        self._candidate_state = None
        self._candidate_count = 0
        self._cooldown = 0
