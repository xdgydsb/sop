"""
物理状态引擎 — 盒子状态持久化 + 物品放置事件检测

事件检测（文档对齐的多条件确认）:
  S1 (box_opened):     盒子打开 稳定10-15帧 + 模型确认
  S2-S4 (object_in_box): 物体轨迹：离开初始区→手接近→接触→入盒→稳定→手离开
  S5 (box_closed):     盒子关闭 + 三物全部入盒确认

每物体维护放置阶段:
  IDLE → APPROACHING → TOUCHING → ENTERING → STABLE_IN_BOX → HAND_LEFT → CONFIRMED
"""
import time
import numpy as np
from typing import List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

OBJECT_ORDER = ["earphone", "charger", "green_bag"]


class PlacementStage(Enum):
    """物体放置阶段"""
    IDLE = auto()            # 等待开始
    APPROACHING = auto()     # 手正在接近物体
    TOUCHING = auto()        # 手接触物体
    ENTERING = auto()        # 物体正在进入盒子
    STABLE_IN_BOX = auto()   # 物体稳定在盒内
    HAND_LEFT = auto()       # 手已离开，物体仍在盒内
    CONFIRMED = auto()       # 放置确认完成
    OCCLUDED = auto()        # 短暂丢失 (1-5帧遮挡)
    LOST = auto()            # 长时间丢失 (>5帧)，可能重置


@dataclass
class PhysicalStateResult:
    box_is_open: bool
    box_is_closed: bool
    box_state_conf: float
    hand_near_box: bool
    hands_detected: bool
    visible_objects: List[str]
    objects_placed: List[str]
    objects_in_box: List[str]
    placed_this_frame: Optional[str]
    current_phys_step: int
    holding: List[str] = field(default_factory=list)
    # Per-object placement stage info
    placement_stages: dict = field(default_factory=dict)  # obj_name → PlacementStage
    stable_frames: dict = field(default_factory=dict)     # obj_name → int
    # Per-object condition details (for EventDetector)
    per_object_conditions: dict = field(default_factory=dict)  # obj_name → {"touched": bool, "left_init": bool, "entered_box": bool, "stable_frames": int, "hand_left_frames": int, "occlusion_frames": int}
    # Wrong-object detection
    wrong_placement: Optional[str] = None  # 错误放入的物品名 (如 charger 在 earphone 之前放入)
    wrong_placement_frames: int = 0        # 错误放置持续的帧数

    @property
    def is_error(self) -> bool:
        return self.box_is_open and self.box_is_closed


@dataclass
class _ObjectPlacementState:
    """单物体的放置过程追踪"""
    stage: PlacementStage = PlacementStage.IDLE
    stable_in_box_frames: int = 0
    hand_touch_frames: int = 0
    hand_approach_frames: int = 0   # 手距离减小的连续帧数
    frames_since_touch: int = 0
    prev_hand_dist: float = 999.0
    was_visible: bool = False
    left_init_region: bool = False
    lost_frames: int = 0            # 连续丢失帧数 (容忍短暂遮挡)
    occlusion_frames: int = 0       # OCCLUDED状态持续帧数
    hand_left_frames: int = 0       # 手离开后的连续帧数
    was_touched: bool = False       # 是否被手接触过
    did_leave_init: bool = False    # 是否离开初始区域
    did_enter_box: bool = False     # 是否进入盒子


class PhysicalStateEngine:
    """盒子状态 + 物品放置事件检测 (多条件确认)"""

    def __init__(self, confirm_frames: int = 8, max_lost: int = 300,
                 stable_threshold: int = 10,
                 occlusion_threshold: int = 5,
                 lost_reset_threshold: int = 15,
                 confirmed_stable_min: int = 5,
                 hand_left_min: int = 3):
        self.confirm_frames = confirm_frames
        self.max_lost = max_lost
        self.stable_threshold = stable_threshold  # 稳定在盒内的帧数阈值
        self.occlusion_threshold = occlusion_threshold  # OCCLUDED→LOST 帧数
        self.lost_reset_threshold = lost_reset_threshold  # LOST→IDLE 帧数
        self.confirmed_stable_min = confirmed_stable_min  # CONFIRMED所需最小稳定帧数
        self.hand_left_min = hand_left_min  # CONFIRMED所需手离开帧数

        # 盒子状态
        self._box_buffer: deque = deque(maxlen=confirm_frames)
        self._last_box_state: Optional[str] = None
        self._box_lost_frames = 0

        # EMA平滑的盒子概率
        self._ema_box_open = 0.0
        self._ema_box_closed = 0.0

        # 物品放置状态
        self._objects_placed: List[str] = []
        self._obj_states = {obj: _ObjectPlacementState() for obj in OBJECT_ORDER}
        self._place_cooldown = 0
        self._wrong_obj: Optional[str] = None
        self._wrong_obj_frames = 0

    def reset(self):
        self._box_buffer.clear()
        self._last_box_state = None
        self._box_lost_frames = 0
        self._ema_box_open = 0.0
        self._ema_box_closed = 0.0
        for obj in OBJECT_ORDER:
            self._obj_states[obj] = _ObjectPlacementState()
        self._objects_placed.clear()
        self._place_cooldown = 0
        self._wrong_obj = None
        self._wrong_obj_frames = 0

    def update(self, detections: List, box_state: str, box_bbox: Optional[Tuple],
               hand_near_box: bool, hands_detected: bool,
               hand_obj_iou: Optional[np.ndarray] = None,
               holding_objects: Optional[List[str]] = None,
               hand_box_dist: Optional[np.ndarray] = None,
               tracked_objects: Optional[dict] = None) -> PhysicalStateResult:
        """
        主更新函数。

        Args:
            detections: YOLO检测列表
            box_state: 'open' / 'closed' / 'unknown'
            box_bbox: 盒子bbox
            hand_near_box: 手是否靠近盒子
            hands_detected: 是否检测到手
            hand_obj_iou: (2, 5) 手物IoU矩阵
            holding_objects: 手持握的物品列表
            hand_box_dist: (2,) 手到盒子距离
            tracked_objects: {track_id: TrackedObject} 来自 YOLODetector
        """
        # ── 1. 盒子状态 EMA 平滑 ──
        alpha = 0.3
        is_open_raw = 1.0 if box_state == "open" else 0.0
        is_closed_raw = 1.0 if box_state == "closed" else 0.0
        self._ema_box_open = alpha * is_open_raw + (1 - alpha) * self._ema_box_open
        self._ema_box_closed = alpha * is_closed_raw + (1 - alpha) * self._ema_box_closed

        # 缓冲区判定
        self._box_buffer.append(box_state)
        box_is_open_raw = self._confirmed_state("open")
        box_is_closed_raw = self._confirmed_state("closed")

        if box_is_open_raw:
            self._last_box_state = "open"
            self._box_lost_frames = 0
        elif box_is_closed_raw:
            self._last_box_state = "closed"
            self._box_lost_frames = 0
        elif self._last_box_state is not None:
            self._box_lost_frames += 1

        # 冲突解决: 如果 open 和 closed 同时确认，取 EMA 值更高者
        if box_is_open_raw and box_is_closed_raw:
            if self._ema_box_open >= self._ema_box_closed:
                box_is_closed_raw = False
            else:
                box_is_open_raw = False

        persist = self._box_lost_frames <= self.max_lost
        box_is_open = (
            box_is_open_raw
            or (persist and self._last_box_state == "open" and self._ema_box_open > 0.3)
        )
        box_is_closed = (
            box_is_closed_raw
            or (persist and self._last_box_state == "closed" and self._ema_box_closed > 0.2)
        )

        box_conf = max(self._ema_box_open, self._ema_box_closed)

        # ── 2. 可见物体 ──
        det_dict = {d.cls_name: d for d in detections}
        visible_objects = [name for name in OBJECT_ORDER
                          if name in det_dict and det_dict[name].confidence > 0.2]

        # ── 3. 手物交互矩阵提取 ──
        hand_obj_iou = (hand_obj_iou if hand_obj_iou is not None
                       else np.zeros((2, 5), dtype=np.float32))
        hand_box_dist = (hand_box_dist if hand_box_dist is not None
                        else np.ones(2, dtype=np.float32) * 999)

        # ── 4. 物体放置事件检测 (多条件确认) ──
        placed_this_frame = None

        if self._place_cooldown > 0:
            self._place_cooldown -= 1

        if box_is_open and self._place_cooldown == 0:
            placed_this_frame = self._detect_placement(
                det_dict, visible_objects, hand_obj_iou,
                hand_box_dist, box_bbox, tracked_objects)

        # ── 5. 物理步骤计算 ──
        n_placed = len(self._objects_placed)
        if box_is_closed and n_placed >= 2:
            phys_step = 5
        elif box_is_open:
            phys_step = 1 + n_placed
        elif box_is_closed:
            phys_step = 0
        else:
            phys_step = 0

        # ── 6. 构建放置阶段信息 ──
        stages = {obj: self._obj_states[obj].stage for obj in OBJECT_ORDER}
        stable_frames = {obj: self._obj_states[obj].stable_in_box_frames for obj in OBJECT_ORDER}
        per_obj_cond = {}
        for obj in OBJECT_ORDER:
            s = self._obj_states[obj]
            per_obj_cond[obj] = {
                "touched": s.was_touched,
                "left_init": s.did_leave_init,
                "entered_box": s.did_enter_box,
                "stable_frames": s.stable_in_box_frames,
                "hand_left_frames": s.hand_left_frames,
                "occlusion_frames": s.occlusion_frames,
            }

        return PhysicalStateResult(
            box_is_open=box_is_open,
            box_is_closed=box_is_closed,
            box_state_conf=box_conf,
            hand_near_box=hand_near_box,
            hands_detected=hands_detected,
            visible_objects=visible_objects,
            objects_placed=list(self._objects_placed),
            objects_in_box=[],
            placed_this_frame=placed_this_frame,
            current_phys_step=phys_step,
            holding=[],
            placement_stages=stages,
            stable_frames=stable_frames,
            per_object_conditions=per_obj_cond,
            wrong_placement=self._wrong_obj,
            wrong_placement_frames=self._wrong_obj_frames,
        )

    def _detect_placement(self, det_dict: dict, visible_objects: List[str],
                          hand_obj_iou: np.ndarray, hand_box_dist: np.ndarray,
                          box_bbox: Optional[Tuple],
                          tracked_objects: Optional[dict]) -> Optional[str]:
        """
        多条件物体放置检测。

        对每个未放置的物体，按顺序检查:
          1. 物体被手接近 (距离连续减小)
          2. 手接触物体 (IoU > 阈值)
          3. 物体进入盒子 (中心在盒内 / in_box_ratio > 0.3)
          4. 物体稳定在盒内 (连续 N 帧)
          5. 手离开物体 (IoU ≈ 0)，物体仍在盒内
          → 放置确认
        """
        obj_name_to_idx = {"earphone": 2, "charger": 3, "green_bag": 4}

        for obj_name in OBJECT_ORDER:
            if obj_name in self._objects_placed:
                continue

            state = self._obj_states[obj_name]
            oi = obj_name_to_idx.get(obj_name, -1)
            is_visible = obj_name in visible_objects

            # 获取该物体的手物IoU (取双手最大)
            touch_iou = 0.0
            if oi >= 0 and hand_obj_iou.shape[0] > 0:
                touch_iou = float(max(hand_obj_iou[:, oi]))

            # 获取追踪状态
            tobj = None
            if tracked_objects:
                for t in tracked_objects.values():
                    if t.cls_name == obj_name and t.frames_lost < 10:
                        tobj = t
                        break

            in_box = False
            in_box_ratio = 0.0
            if tobj is not None and box_bbox is not None:
                in_box = tobj.in_box
                stable = tobj.stable_in_box_frames
            else:
                stable = 0

            # ── 阶段转换逻辑 ──
            if state.stage == PlacementStage.IDLE:
                if not is_visible:
                    continue
                state.was_visible = True

                # 检测手接近
                hand_dist = min(hand_box_dist[hand_box_dist < 999]) if len(hand_box_dist) > 0 else 999
                if hand_dist < state.prev_hand_dist and hand_dist < 0.4:
                    state.hand_approach_frames += 1
                else:
                    state.hand_approach_frames = 0
                state.prev_hand_dist = hand_dist

                if state.hand_approach_frames >= 3 or touch_iou > 0.03:
                    state.stage = PlacementStage.APPROACHING

            elif state.stage == PlacementStage.APPROACHING:
                if touch_iou > 0.04:
                    state.hand_touch_frames += 1
                    if state.hand_touch_frames >= 2:
                        state.was_touched = True
                        state.stage = PlacementStage.TOUCHING
                elif not is_visible:
                    state.stage = PlacementStage.IDLE  # 物体消失，回退

            elif state.stage == PlacementStage.TOUCHING:
                if touch_iou < 0.02:
                    state.frames_since_touch += 1
                else:
                    state.frames_since_touch = 0

                if in_box or (tobj and tobj.stable_in_box_frames >= 2):
                    state.did_enter_box = True
                    state.stage = PlacementStage.ENTERING
                    state.stable_in_box_frames = tobj.stable_in_box_frames if tobj else 0

                if state.frames_since_touch > 30:
                    state.stage = PlacementStage.IDLE  # 超时回退

            elif state.stage == PlacementStage.ENTERING:
                if tobj:
                    state.stable_in_box_frames = tobj.stable_in_box_frames
                    state.occlusion_frames = 0
                    state.lost_frames = 0
                    if in_box:
                        state.did_enter_box = True
                elif not is_visible:
                    state.occlusion_frames += 1
                    state.lost_frames += 1
                    # CRITICAL: disappearance does NOT mean in_box!
                    # Brief occlusion is tolerated but does NOT count toward stable_in_box
                    if state.occlusion_frames <= 3:
                        pass  # tolerate brief loss
                    elif state.occlusion_frames <= self.occlusion_threshold:
                        state.stage = PlacementStage.OCCLUDED
                    else:
                        state.stage = PlacementStage.LOST
                        state.stable_in_box_frames = max(0, state.stable_in_box_frames - 1)
                else:
                    state.stable_in_box_frames += 1 if in_box else 0
                    state.occlusion_frames = 0
                    state.lost_frames = 0

                if state.stable_in_box_frames >= self.stable_threshold:
                    state.did_enter_box = True
                    state.stage = PlacementStage.STABLE_IN_BOX

            elif state.stage == PlacementStage.OCCLUDED:
                if is_visible:
                    state.occlusion_frames = 0
                    state.stage = PlacementStage.ENTERING  # resume
                else:
                    state.occlusion_frames += 1
                    if state.occlusion_frames > self.occlusion_threshold:
                        state.stage = PlacementStage.LOST
                        state.stable_in_box_frames = max(0, state.stable_in_box_frames - 1)

            elif state.stage == PlacementStage.LOST:
                if is_visible:
                    state.occlusion_frames = 0
                    state.stage = PlacementStage.ENTERING  # resume tracking
                else:
                    state.occlusion_frames += 1
                    if state.occlusion_frames > self.lost_reset_threshold:
                        # Fully reset — object truly gone
                        state.stage = PlacementStage.IDLE
                        state.was_touched = False
                        state.did_leave_init = False
                        state.did_enter_box = False
                        state.stable_in_box_frames = 0
                        state.occlusion_frames = 0
                        state.lost_frames = 0

            elif state.stage == PlacementStage.STABLE_IN_BOX:
                if tobj is None and not is_visible:
                    state.occlusion_frames += 1
                    if state.occlusion_frames > 5:
                        state.stage = PlacementStage.LOST
                else:
                    state.occlusion_frames = 0
                if touch_iou < 0.02:
                    state.frames_since_touch += 1
                    state.hand_left_frames += 1
                else:
                    state.hand_left_frames = 0
                if state.frames_since_touch >= 5:
                    state.stage = PlacementStage.HAND_LEFT

            elif state.stage == PlacementStage.HAND_LEFT:
                if touch_iou < 0.02:
                    state.hand_left_frames += 1
                else:
                    state.hand_left_frames = 0
                # Stricter CONFIRMED: must satisfy ALL conditions
                # "not is_visible" alone does NOT mean confirmed!
                if state.was_touched and state.did_enter_box and \
                   state.stable_in_box_frames >= self.confirmed_stable_min and \
                   state.hand_left_frames >= self.hand_left_min:
                    state.stage = PlacementStage.CONFIRMED
                elif not is_visible and state.occlusion_frames < 5:
                    state.occlusion_frames += 1
                elif state.occlusion_frames > 10:
                    state.stage = PlacementStage.LOST

            elif state.stage == PlacementStage.CONFIRMED:
                expected = OBJECT_ORDER[len(self._objects_placed)]
                if obj_name == expected:
                    self._objects_placed.append(obj_name)
                    self._place_cooldown = 20
                    print(f"  [Phys] PLACED (multi-condition): {obj_name} "
                          f"touched={state.was_touched} entered={state.did_enter_box} "
                          f"stable={state.stable_in_box_frames} hand_left={state.hand_left_frames}")
                    return obj_name
                else:
                    # 错误物品先入盒 — 记录但不推进
                    self._wrong_obj = obj_name
                    self._wrong_obj_frames += 1
                    print(f"  [Phys] WRONG OBJECT: {obj_name} placed (expected {expected})")

        return None

    def _confirmed_state(self, state_name: str) -> bool:
        """检查缓冲区中某状态是否确认"""
        if len(self._box_buffer) < self.confirm_frames:
            return False
        return list(self._box_buffer).count(state_name) >= self.confirm_frames * 0.55
