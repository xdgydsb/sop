"""
融合决策 — 时序模型(主分类) + YOLO盒子状态(硬约束) + FSM(序列约束)

核心策略（简洁）:
1. 盒子关闭 → S5 (硬约束，不需要确认放置)
2. 盒子打开 + Idle → S1
3. 盒子打开期间: 时序模型running_max驱动 S2→S3→S4
4. FSM确保单调推进，每步最多前进1
"""
import time
from collections import deque, Counter
from typing import List
from dataclasses import dataclass, field


STEP_NAMES = {
    0: "Idle", 1: "S1-Open Box", 2: "S2-Earphone",
    3: "S3-Charger", 4: "S4-Green Bag", 5: "S5-Close Box", 6: "Done",
}


@dataclass
class FusionResult:
    step: int
    step_name: str
    is_correct: bool
    message: str
    error_type: str
    progress: float
    confidence: float
    model_step: int = 0
    model_conf: float = 0.0
    model_top3: List = field(default_factory=list)
    phys_step: int = 0
    box_is_open: bool = False
    visible_objects: List = field(default_factory=list)
    objects_placed: List = field(default_factory=list)


class FusionEngine:
    """时序模型 + YOLO + FSM"""

    def __init__(self, confirm_count: int = 2, model_window: int = 16,
                 min_step_sec: float = 0.4):
        self.confirm_count = confirm_count
        self.model_window = model_window
        self.min_step_sec = min_step_sec
        self.current_step = 0
        self._step_start_time: float = None
        self._candidate_step = 0
        self._candidate_count = 0
        self._model_history: deque = deque(maxlen=model_window)
        self._running_max = 0  # 模型预测的历史最大值

    def reset(self):
        self.current_step = 0
        self._step_start_time = None
        self._candidate_step = 0
        self._candidate_count = 0
        self._model_history.clear()
        self._running_max = 0

    def update(self, phys_result, model_step: int, model_conf: float,
               model_top3: List, timestamp: float = None) -> FusionResult:
        if timestamp is None:
            timestamp = time.time()
        if self._step_start_time is None:
            self._step_start_time = timestamp

        step_elapsed = timestamp - self._step_start_time
        error_type = ""

        # ── 模型预测平滑 ──
        self._model_history.append(model_step)
        smoothed_step = 0
        if len(self._model_history) >= 3:
            smoothed_step = Counter(self._model_history).most_common(1)[0][0]

        # 更新running_max
        if smoothed_step > self._running_max:
            self._running_max = smoothed_step

        # ── 核心决策 ──
        target = self.current_step

        # 规则1: 盒子关闭 + 已开始SOP → S5（先检查模型是否看到S4）
        if phys_result.box_is_closed and self.current_step >= 2:
            # 检查原始或平滑模型输出是否有S4信号
            model_sees_s4 = (model_step == 4 or
                             (len(model_top3) > 1 and model_top3[1][0] == 4 and model_top3[1][1] > 0.1))
            if (model_sees_s4 or self._running_max >= 4) and self.current_step < 4:
                target = 4  # 模型之前预测过S4，先补上
            else:
                target = 5

        # 规则2: 盒子打开 + Idle → S1
        elif phys_result.box_is_open and self.current_step == 0 and step_elapsed > 0.3:
            target = 1

        # 规则3: 盒子打开 + S1~S4 → 模型running_max + YOLO辅助驱动
        elif phys_result.box_is_open and 1 <= self.current_step <= 4:
            next_step = self.current_step + 1

            # YOLO辅助：检测到目标物体存在
            vis = set(phys_result.visible_objects)
            yolo_hint = False
            if self.current_step == 1 and "earphone" in vis:
                yolo_hint = True  # S2目标物体可见
            elif self.current_step == 2 and "charger" in vis:
                yolo_hint = True  # S3目标物体可见
            elif self.current_step == 3 and "green_bag" in vis:
                yolo_hint = True  # S4目标物体可见

            # 主信号：模型预测 + YOLO辅助
            model_ready = self._running_max >= next_step and step_elapsed > self.min_step_sec
            if model_ready or (yolo_hint and step_elapsed > 1.0):
                target = next_step
            # 兜底超时
            elif step_elapsed > 3.0 and next_step <= 4:
                target = next_step

        # 规则4: 已完成S5后盒子又开了 → 新循环开始
        elif self.current_step == 5 and phys_result.box_is_open:
            # 重置为S1（新盒子）
            target = 5  # 保持S5，等待reset

        # ── 确认计数器 ──
        if target != self.current_step:
            if target == self._candidate_step:
                self._candidate_count += 1
            else:
                self._candidate_step = target
                self._candidate_count = 1

            if self._candidate_count >= self.confirm_count:
                old = self.current_step
                self.current_step = target
                self._step_start_time = timestamp
                self._candidate_step = 0
                self._candidate_count = 0
                self._model_history.clear()
                self._running_max = max(self._running_max, target)
                print("  [Fusion] S%d->S%d (%s) model_s=%d run_max=%d conf=%.2f" % (
                    old, target, STEP_NAMES.get(target, "?"),
                    smoothed_step, self._running_max, model_conf))
        else:
            self._candidate_step = 0
            self._candidate_count = 0

        # ── 错误检测 ──
        if self.current_step in (2, 3, 4) and phys_result.box_is_closed:
            error_type = "BOX_NOT_OPEN"
        if self.current_step == 5 and phys_result.box_is_open:
            error_type = "BOX_NOT_CLOSED"

        return FusionResult(
            step=self.current_step,
            step_name=STEP_NAMES.get(self.current_step, "?"),
            is_correct=error_type == "",
            message=STEP_NAMES.get(self.current_step, "?"),
            error_type=error_type,
            progress=self.current_step / 5.0,
            confidence=model_conf,
            model_step=smoothed_step,
            model_conf=model_conf,
            model_top3=model_top3,
            phys_step=phys_result.current_phys_step,
            box_is_open=phys_result.box_is_open,
            visible_objects=phys_result.visible_objects,
            objects_placed=list(phys_result.objects_placed),
        )
