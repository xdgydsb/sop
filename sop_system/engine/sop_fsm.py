"""
SOP有限状态机 — 序列约束与错误检测
合法转移: IDLE(0) → S1(1) → S2(2) → S3(3) → S4(4) → S5(5) → DONE(6)
检测: 乱序、漏步、超时、错误物品
"""
import time
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


class SOPStep(Enum):
    IDLE = 0
    S1_OPEN = 1
    S2_EARPHONE = 2
    S3_CHARGER = 3
    S4_BAG = 4
    S5_CLOSE = 5
    COMPLETE = 6
    ERROR = 7


STEP_NAMES = {
    SOPStep.IDLE: "等待开始",
    SOPStep.S1_OPEN: "1.打开纸盒",
    SOPStep.S2_EARPHONE: "2.放入耳机盒",
    SOPStep.S3_CHARGER: "3.放入插头",
    SOPStep.S4_BAG: "4.放入绿袋",
    SOPStep.S5_CLOSE: "5.关闭纸盒",
    SOPStep.COMPLETE: "完成",
    SOPStep.ERROR: "错误",
}

# Event name → SOPStep mapping (for event-driven FSM)
# box_closed → COMPLETE: closing the box IS the final action.
# No intermediate S5_CLOSE state that waits for another event.
EVENT_TO_STEP = {
    "box_opened": SOPStep.S1_OPEN,
    "earphone_in_box": SOPStep.S2_EARPHONE,
    "charger_in_box": SOPStep.S3_CHARGER,
    "green_bag_in_box": SOPStep.S4_BAG,
    "box_closed": SOPStep.S5_CLOSE,
    "early_close_alarm": SOPStep.ERROR,
}


@dataclass
class FSMResult:
    """FSM每次update的输出"""
    current_step: SOPStep
    step_id: int
    step_name: str
    is_correct: bool
    message: str
    error_type: str           # WRONG_ORDER / MISSING_STEP / TIMEOUT / ""
    progress: float           # 0.0-1.0
    is_complete: bool
    has_error: bool
    timestamp: float
    duration_current: float   # 当前步骤已持续时间
    step_history: List[Dict] = field(default_factory=list)


class SOPStateMachine:
    """
    SOP有限状态机
    定义合法的步骤转移, 检测非法转移并诊断为错误
    """

    # 合法转移表: 当前步骤 → [允许的下一步骤]
    VALID_TRANSITIONS = {
        SOPStep.IDLE: [SOPStep.S1_OPEN],
        SOPStep.S1_OPEN: [SOPStep.S2_EARPHONE, SOPStep.ERROR],
        SOPStep.S2_EARPHONE: [SOPStep.S3_CHARGER, SOPStep.ERROR],
        SOPStep.S3_CHARGER: [SOPStep.S4_BAG, SOPStep.ERROR],
        SOPStep.S4_BAG: [SOPStep.S5_CLOSE, SOPStep.COMPLETE, SOPStep.ERROR],
        SOPStep.S5_CLOSE: [SOPStep.COMPLETE, SOPStep.ERROR],
        SOPStep.COMPLETE: [],
        SOPStep.ERROR: [SOPStep.IDLE],  # 重置
    }

    def __init__(self, timeout: float = 30.0, min_step_duration: float = 0.5):
        self.current_step = SOPStep.IDLE
        self.timeout = timeout
        self.min_step_duration = min_step_duration
        self.step_start_time: Optional[float] = None
        self.step_history: List[Dict] = []
        self.error_occurred = False
        self.last_error_msg = ""

    def _step_id_to_enum(self, step_id: int) -> SOPStep:
        mapping = {0: SOPStep.IDLE, 1: SOPStep.S1_OPEN, 2: SOPStep.S2_EARPHONE,
                    3: SOPStep.S3_CHARGER, 4: SOPStep.S4_BAG, 5: SOPStep.S5_CLOSE,
                    6: SOPStep.COMPLETE}
        return mapping.get(step_id, SOPStep.ERROR)

    def validate(self, detected_step: int, confidence: float,
                 physical_state_ok: bool = True,
                 timestamp: float = None) -> FSMResult:
        """
        验证检测到的步骤是否合法。

        Args:
            detected_step: 检测到的步骤ID (0-6)
            confidence: 置信度 0-1
            physical_state_ok: 物理状态是否支持此步骤
            timestamp: 当前时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        if self.step_start_time is None:
            self.step_start_time = timestamp

        det_step = self._step_id_to_enum(detected_step)
        duration = timestamp - self.step_start_time

        # 低置信度 → 保持当前状态
        if confidence < 0.4:
            return FSMResult(
                current_step=self.current_step,
                step_id=self.current_step.value,
                step_name=STEP_NAMES[self.current_step],
                is_correct=True,
                message=f"保持当前: {STEP_NAMES[self.current_step]}",
                error_type="",
                progress=min(self.current_step.value / 5.0, 1.0),
                is_complete=(self.current_step == SOPStep.COMPLETE),
                has_error=False,
                timestamp=timestamp,
                duration_current=duration,
                step_history=self.step_history.copy(),
            )

        # 已完成 → 忽略新检测
        if self.current_step == SOPStep.COMPLETE:
            return FSMResult(
                current_step=SOPStep.COMPLETE, step_id=6,
                step_name=STEP_NAMES[SOPStep.COMPLETE],
                is_correct=True, message="流程已完成",
                error_type="", progress=1.0, is_complete=True, has_error=False,
                timestamp=timestamp, duration_current=duration,
                step_history=self.step_history.copy(),
            )

        # Already in ERROR state — auto-recover when physical evidence is strong
        if self.current_step == SOPStep.ERROR:
            if physical_state_ok and confidence > 0.8 and det_step.value > 0:
                # Auto-recover: reset to the step before the detected one
                prev_step = self._step_id_to_enum(max(0, det_step.value - 1))
                if prev_step != SOPStep.ERROR:
                    self.current_step = prev_step
                    self.step_start_time = timestamp
                    self.error_occurred = False
                    # Re-validate with recovered state
                    return self.validate(detected_step, confidence, physical_state_ok, timestamp)
            return FSMResult(
                current_step=SOPStep.ERROR, step_id=7,
                step_name=STEP_NAMES[SOPStep.ERROR],
                is_correct=False, message=self.last_error_msg,
                error_type="", progress=-1, is_complete=False, has_error=True,
                timestamp=timestamp, duration_current=duration,
                step_history=self.step_history.copy(),
            )

        # 物理状态不匹配 → 拒绝
        if not physical_state_ok and det_step.value > self.current_step.value:
            return FSMResult(
                current_step=self.current_step,
                step_id=self.current_step.value,
                step_name=STEP_NAMES[self.current_step],
                is_correct=True,
                message=f"物理状态不匹配, 保持: {STEP_NAMES[self.current_step]}",
                error_type="", progress=self.current_step.value / 5.0,
                is_complete=False, has_error=False,
                timestamp=timestamp, duration_current=duration,
                step_history=self.step_history.copy(),
            )

        # 相同步骤 → 保持
        if det_step == self.current_step:
            # 超时检查
            if self.current_step.value < 6 and duration > self.timeout:
                self.current_step = SOPStep.ERROR
                self.error_occurred = True
                self.last_error_msg = f"步骤'{STEP_NAMES[det_step]}'超时 ({duration:.1f}s)"
                return FSMResult(
                    current_step=SOPStep.ERROR, step_id=7,
                    step_name=STEP_NAMES[SOPStep.ERROR],
                    is_correct=False, message=self.last_error_msg, error_type="TIMEOUT",
                    progress=-1, is_complete=False, has_error=True,
                    timestamp=timestamp, duration_current=duration,
                    step_history=self.step_history.copy(),
                )
            return FSMResult(
                current_step=self.current_step, step_id=self.current_step.value,
                step_name=STEP_NAMES[self.current_step],
                is_correct=True, message=STEP_NAMES[self.current_step],
                error_type="", progress=self.current_step.value / 5.0,
                is_complete=False, has_error=False,
                timestamp=timestamp, duration_current=duration,
                step_history=self.step_history.copy(),
            )

        # 合法转移
        if det_step in self.VALID_TRANSITIONS.get(self.current_step, []):
            if det_step == SOPStep.ERROR:
                return self._handle_error(SOPStep.ERROR, timestamp, duration)

            # Minimum step duration check (prevent flicker)
            # Bypass for: IDLE start, high-confidence physical evidence, or fast recovery
            bypass_duration = (
                self.current_step == SOPStep.IDLE
                or confidence > 0.85
                or (det_step.value == self.current_step.value + 1 and physical_state_ok)
            )
            if duration >= self.min_step_duration or bypass_duration:
                old_step = self.current_step
                self.step_history.append({
                    "step": old_step.value, "step_name": STEP_NAMES[old_step],
                    "duration": duration, "end_time": timestamp,
                })
                self.current_step = det_step
                self.step_start_time = timestamp
                return FSMResult(
                    current_step=self.current_step, step_id=det_step.value,
                    step_name=STEP_NAMES[det_step],
                    is_correct=True,
                    message=f"步骤推进: {STEP_NAMES[old_step]} → {STEP_NAMES[det_step]}",
                    error_type="",
                    progress=det_step.value / 5.0,
                    is_complete=(det_step == SOPStep.COMPLETE),
                    has_error=False,
                    timestamp=timestamp, duration_current=0,
                    step_history=self.step_history.copy(),
                )

        # 非法转移 → 错误诊断
        return self._handle_error(det_step, timestamp, duration)

    def _handle_error(self, det_step: SOPStep, timestamp: float,
                      duration: float) -> FSMResult:
        error_type, error_msg = self._diagnose_error(det_step)
        self.last_error_msg = error_msg
        self.current_step = SOPStep.ERROR
        self.error_occurred = True
        return FSMResult(
            current_step=SOPStep.ERROR, step_id=7,
            step_name=STEP_NAMES[SOPStep.ERROR],
            is_correct=False, message=error_msg, error_type=error_type,
            progress=-1, is_complete=False, has_error=True,
            timestamp=timestamp, duration_current=duration,
            step_history=self.step_history.copy(),
        )

    def _diagnose_error(self, detected_step: SOPStep) -> Tuple[str, str]:
        """诊断具体错误类型"""
        # 乱序(回退)
        if detected_step.value < self.current_step.value:
            return ("WRONG_ORDER",
                    f"乱序! 检测到{STEP_NAMES[detected_step]}, "
                    f"但当前已在{STEP_NAMES[self.current_step]}")
        # 漏步(跳过多个步骤)
        if detected_step.value > self.current_step.value + 1:
            skipped = list(range(self.current_step.value + 1, detected_step.value))
            skipped_names = [STEP_NAMES[self._step_id_to_enum(s)] for s in skipped]
            return ("MISSING_STEP",
                    f"漏步! 从{STEP_NAMES[self.current_step]}跳到"
                    f"{STEP_NAMES[detected_step]}, 遗漏: {skipped_names}")
        # 其他非法转移
        return ("WRONG_ORDER",
                f"非法转移: {STEP_NAMES[self.current_step]} → {STEP_NAMES[detected_step]}")

    def validate_event(self, event_name: str, confidence: float,
                       timestamp: float = None) -> FSMResult:
        """Validate an event (e.g. 'earphone_in_box') against the FSM.

        Maps the event name to a target SOPStep, then delegates to the
        existing validate() method with physical_state_ok=True since the
        EventDetector has already verified physical conditions.

        Args:
            event_name: Standard event name (box_opened, earphone_in_box, etc.)
            confidence: Event confidence 0-1 from EventDetector
            timestamp: Current time

        Returns:
            FSMResult with transition validation
        """
        target_step = EVENT_TO_STEP.get(event_name)
        if target_step is None:
            return FSMResult(
                current_step=self.current_step,
                step_id=self.current_step.value,
                step_name=STEP_NAMES[self.current_step],
                is_correct=False,
                message=f"Unknown event: {event_name}",
                error_type="",
                progress=self.get_progress(),
                is_complete=(self.current_step == SOPStep.COMPLETE),
                has_error=False,
                timestamp=timestamp or time.time(),
                duration_current=(timestamp or time.time()) - (self.step_start_time or timestamp or time.time()),
                step_history=self.step_history.copy(),
            )

        # early_close_alarm → ERROR
        if event_name == "early_close_alarm":
            return self._handle_error(SOPStep.ERROR, timestamp or time.time(),
                                     (timestamp or time.time()) - (self.step_start_time or timestamp or time.time()))

        # Delegate to existing validate with physical_state_ok=True
        return self.validate(
            detected_step=target_step.value,
            confidence=confidence,
            physical_state_ok=True,  # EventDetector already verified physics
            timestamp=timestamp,
        )

    def reset(self):
        self.current_step = SOPStep.IDLE
        self.step_start_time = None
        self.step_history.clear()
        self.error_occurred = False
        self.last_error_msg = ""

    def get_progress(self) -> float:
        if self.current_step == SOPStep.COMPLETE:
            return 1.0
        if self.current_step == SOPStep.ERROR:
            return -1.0
        return self.current_step.value / 5.0
