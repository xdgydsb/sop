"""MediaPipe手部检测 — 21点关键点 + 手物交互分析

Supports both MediaPipe APIs:
  - Legacy: mp.solutions.hands (mediapipe < 0.10.14)
  - Modern: mp.tasks.vision.HandLandmarker (mediapipe >= 0.10.14)
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

# ── API compatibility detection ──
_HAS_SOLUTIONS = hasattr(mp, 'solutions')
_HAS_TASKS = hasattr(mp, 'tasks')

if _HAS_SOLUTIONS:
    _API = 'solutions'
elif _HAS_TASKS:
    _API = 'tasks'
else:
    raise ImportError("mediapipe has neither 'solutions' nor 'tasks' — broken install?")


@dataclass
class HandInfo:
    hand_index: int                 # 0 or 1
    landmarks: np.ndarray           # (21, 3) — x,y,z 归一化坐标
    handedness: str                 # "Left" / "Right"
    bbox: Tuple[int,int,int,int]    # (x1,y1,x2,y2) 像素坐标
    center: Tuple[float,float]      # (cx,cy) 归一化
    openness: float                 # 手张开度 0-1
    is_holding: bool = False        # 是否在持握物品
    holding_object: Optional[str] = None  # 持握的物品名称


class HandDetector:
    def __init__(self, min_detection_conf: float = 0.5,
                 min_tracking_conf: float = 0.5):
        self._api = _API
        self.active_hand_idx: int = -1
        self._prev_palm: Dict[str, Tuple[float, float]] = {}
        self.palm_velocity: Dict[str, Tuple[float, float]] = {}

        if self._api == 'solutions':
            self._init_solutions(min_detection_conf, min_tracking_conf)
        else:
            self._init_tasks(min_detection_conf, min_tracking_conf)

    # ── Solutions API (mediapipe < 0.10.14) ──
    def _init_solutions(self, min_detection_conf, min_tracking_conf):
        self._solutions_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

    def _detect_solutions(self, frame: np.ndarray, h: int, w: int) -> List[HandInfo]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._solutions_hands.process(rgb)
        hands = []
        if not results.multi_hand_landmarks:
            return hands

        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                                dtype=np.float32)
            bbox, center, openness = self._compute_hand_geom(lm_array, w, h)
            handedness = "Unknown"
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label
            self._update_palm_velocity(lm_array, handedness)
            hands.append(HandInfo(hand_index=idx, landmarks=lm_array,
                                  handedness=handedness, bbox=bbox,
                                  center=center, openness=openness))
        return hands

    def _close_solutions(self):
        self._solutions_hands.close()

    # ── Tasks API (mediapipe >= 0.10.14) ──
    def _init_tasks(self, min_detection_conf, min_tracking_conf):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        from pathlib import Path

        # Find model file: check project models/ first, then package path
        model_path = Path(__file__).parent.parent / 'models' / 'hand_landmarker.task'
        if not model_path.is_file():
            # Fallback: try mediapipe's own bundled model location
            import mediapipe as _mp
            model_path = Path(_mp.__path__[0]) / 'modules' / 'hand_landmark' / 'hand_landmarker.task'
        if not model_path.is_file():
            raise FileNotFoundError(
                f"HandLandmarker model not found. Download from:\n"
                f"  https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/latest/"
                f"hand_landmarker.task\n"
                f"and save to: {Path(__file__).parent.parent / 'models' / 'hand_landmarker.task'}"
            )
        base_options = mp_tasks.BaseOptions(
            model_asset_path=str(model_path)
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_tracking_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._tasks_landmarker = vision.HandLandmarker.create_from_options(options)

    def _detect_tasks(self, frame: np.ndarray, h: int, w: int) -> List[HandInfo]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._tasks_landmarker.detect(mp_image)
        hands = []
        if not result.hand_landmarks:
            return hands

        for idx, landmarks in enumerate(result.hand_landmarks):
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks],
                                dtype=np.float32)
            bbox, center, openness = self._compute_hand_geom(lm_array, w, h)
            handedness = "Unknown"
            if result.handedness and idx < len(result.handedness):
                handedness = result.handedness[idx][0].category_name
            self._update_palm_velocity(lm_array, handedness)
            hands.append(HandInfo(hand_index=idx, landmarks=lm_array,
                                  handedness=handedness, bbox=bbox,
                                  center=center, openness=openness))
        return hands

    def _close_tasks(self):
        pass  # tasks API doesn't require explicit close

    # ── Shared geometry ──
    @staticmethod
    def _compute_hand_geom(lm_array, w, h):
        xs = lm_array[:, 0]; ys = lm_array[:, 1]
        bbox = (int(float(min(xs))*w), int(float(min(ys))*h),
                int(float(max(xs))*w), int(float(max(ys))*h))
        center = (float(np.mean(xs)), float(np.mean(ys)))
        thumb_tip = lm_array[4, :2]
        pinky_tip = lm_array[20, :2]
        openness = min(1.0, float(np.linalg.norm(thumb_tip - pinky_tip)) / 0.28)
        return bbox, center, openness

    def _update_palm_velocity(self, lm_array, handedness):
        palm_x = float(np.mean([lm_array[i, 0] for i in [0, 5, 9, 13, 17]]))
        palm_y = float(np.mean([lm_array[i, 1] for i in [0, 5, 9, 13, 17]]))
        prev = self._prev_palm.get(handedness)
        vx = palm_x - prev[0] if prev else 0.0
        vy = palm_y - prev[1] if prev else 0.0
        self._prev_palm[handedness] = (palm_x, palm_y)
        self.palm_velocity[handedness] = (vx, vy)

    # ── Public API ──

    def detect(self, frame: np.ndarray) -> List[HandInfo]:
        """检测帧中所有手"""
        h, w = frame.shape[:2]
        if self._api == 'solutions':
            return self._detect_solutions(frame, h, w)
        else:
            return self._detect_tasks(frame, h, w)

    def select_active_hand(self, hands: List[HandInfo],
                           target_object_bbox: Optional[Tuple] = None,
                           box_bbox: Optional[Tuple] = None) -> int:
        """
        选择主动手 — 距离目标物体或盒子最近的手。

        Args:
            hands: 检测到的手列表
            target_object_bbox: 当前步骤目标物体的bbox (如 earphone bbox)
            box_bbox: 盒子bbox (fallback)

        Returns:
            active_hand_idx: 0 或 1, 如果没有手则 -1
        """
        if not hands:
            self.active_hand_idx = -1
            return -1

        if len(hands) == 1:
            self.active_hand_idx = 0
            return 0

        # 确定参考点：优先目标物体，其次盒子
        ref_bbox = target_object_bbox or box_bbox
        if ref_bbox is None:
            self.active_hand_idx = 0
            return 0

        ref_cx = (ref_bbox[0] + ref_bbox[2]) / 2
        ref_cy = (ref_bbox[1] + ref_bbox[3]) / 2

        # 选距离参考点最近的手
        best_idx = 0
        best_dist = float('inf')
        for i, hand in enumerate(hands[:2]):
            hcx = hand.center[0]  # 归一化坐标
            hcy = hand.center[1]
            # 假设 ref 是像素坐标，需要归一化比较
            # hand.center 是 0-1 归一化的，这里用归一化距离比较
            # 实际上我们只需要比较相对距离
            dist = (hcx - ref_cx / 640) ** 2 + (hcy - ref_cy / 480) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        self.active_hand_idx = best_idx
        return best_idx

    def get_active_hand(self, hands: List[HandInfo]) -> Optional[HandInfo]:
        """获取当前主动手"""
        if not hands or self.active_hand_idx < 0:
            return hands[0] if hands else None
        idx = min(self.active_hand_idx, len(hands) - 1)
        return hands[idx]

    def compute_interaction(self, hands: List[HandInfo],
                            detections: List,  # List[Detection]
                            box_bbox: Optional[Tuple],
                            frame_shape: Tuple[int,int]) -> Dict:
        """
        计算手物交互特征。
        Returns dict with:
          - hand_obj_dist: (2,5) 每只手到每个物体的归一化距离
          - hand_obj_iou: (2,5) 手bbox与物品bbox的IoU
          - hand_box_dist: (2,) 手到盒子中心的距离
          - holding: bool每只手是否在持握物品
        """
        h, w = frame_shape
        n_hands = min(len(hands), 2)
        n_objs = 5  # box_closed, box_open, earphone, charger, green_bag

        hand_obj_dist = np.zeros((2, n_objs), dtype=np.float32)
        hand_obj_iou = np.zeros((2, n_objs), dtype=np.float32)
        hand_box_dist = np.ones(2, dtype=np.float32) * 999

        obj_map = {"box_closed": 0, "box_open": 1, "earphone": 2,
                    "charger": 3, "green_bag": 4}

        for hi, hand in enumerate(hands[:2]):
            hx1, hy1, hx2, hy2 = hand.bbox
            h_area = max(1, (hx2-hx1) * (hy2-hy1))

            for det in detections:
                oi = obj_map.get(det.cls_name, -1)
                if oi < 0:
                    continue
                ox1, oy1, ox2, oy2 = det.bbox
                ocx, ocy = (ox1+ox2)/2, (oy1+oy2)/2

                # 归一化距离
                dist = np.sqrt((hand.center[0]*w - ocx)**2 + (hand.center[1]*h - ocy)**2) / w
                hand_obj_dist[hi, oi] = 1.0 / (1.0 + dist * 10)

                # IoU
                ix1, iy1 = max(hx1, ox1), max(hy1, oy1)
                ix2, iy2 = min(hx2, ox2), min(hy2, oy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1) * (iy2-iy1)
                    o_area = max(1, (ox2-ox1) * (oy2-oy1))
                    union = h_area + o_area - inter
                    hand_obj_iou[hi, oi] = inter / union if union > 0 else 0

            # 手到盒子距离
            if box_bbox is not None:
                bx_c = ((box_bbox[0]+box_bbox[2])/2, (box_bbox[1]+box_bbox[3])/2)
                hd = np.sqrt((hand.center[0]*w - bx_c[0])**2 +
                             (hand.center[1]*h - bx_c[1])**2) / w
                hand_box_dist[hi] = hd

        # 判断持握
        holding = [False, False]
        holding_obj = [None, None]
        for hi in range(n_hands):
            for oi, cls_name in enumerate(["box_closed", "box_open", "earphone",
                                            "charger", "green_bag"]):
                if hand_obj_iou[hi, oi] > 0.15:
                    holding[hi] = True
                    holding_obj[hi] = cls_name
                    break

        return {
            "hand_obj_dist": hand_obj_dist,
            "hand_obj_iou": hand_obj_iou,
            "hand_box_dist": hand_box_dist,
            "holding": holding,
            "holding_obj": holding_obj,
            "hands_active": len(hands) > 0,
        }

    def draw_landmarks(self, frame: np.ndarray, hands: List[HandInfo]) -> np.ndarray:
        """在帧上绘制手部关键点"""
        for hand in hands:
            # Use mediapipe drawing if available, else simple dots
            hx1, hy1, hx2, hy2 = hand.bbox
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 100), 2)
            for lm in hand.landmarks:
                px, py = int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0])
                cv2.circle(frame, (px, py), 3, (0, 200, 100), -1)
        return frame

    def close(self):
        if self._api == 'solutions':
            self._close_solutions()
        else:
            self._close_tasks()
