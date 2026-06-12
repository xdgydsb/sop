"""
DetectionStabilizer — per-class EMA-based detection confirmation.

Key principles:
  - Display bbox ONLY from current frame's detection (no persistence)
  - Confirmed state uses EMA-smoothed confidence (not binary hit/miss)
  - Spatial consistency: new bbox must have IoU with previous, or re-stabilize
  - max_lost controls FSM confirmation persistence, NOT bbox display

Rule: "看不到物体就不能显示框" — bbox = None when object not detected this frame.
      No tracker-predicted boxes, no last_bbox, no fallback.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


def _iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1.0)


@dataclass
class ClassStabilizerConfig:
    high_thr: float = 0.50
    low_thr: float = 0.30
    min_hits: int = 3
    max_lost: int = 8
    ema_alpha: float = 0.35


DEFAULT_CONFIGS = {
    "box_closed": ClassStabilizerConfig(high_thr=0.55, low_thr=0.35, min_hits=3, max_lost=10, ema_alpha=0.45),
    "box_open":   ClassStabilizerConfig(high_thr=0.55, low_thr=0.35, min_hits=3, max_lost=10, ema_alpha=0.45),
    "earphone":   ClassStabilizerConfig(high_thr=0.22, low_thr=0.15, min_hits=4, max_lost=40, ema_alpha=0.40),
    "charger":    ClassStabilizerConfig(high_thr=0.22, low_thr=0.15, min_hits=4, max_lost=40, ema_alpha=0.40),
    "green_bag":  ClassStabilizerConfig(high_thr=0.22, low_thr=0.15, min_hits=4, max_lost=40, ema_alpha=0.40),
}


@dataclass
class _ClassState:
    hits: int = 0
    lost: int = 0
    confirmed: bool = False
    ema_conf: float = 0.0
    current_bbox: Optional[Tuple] = None
    current_conf: float = 0.0
    current_center: Optional[Tuple] = None
    source: str = "yolo"
    candidate_bbox: Optional[Tuple] = None
    candidate_hits: int = 0


class DetectionStabilizer:
    """Per-class detection confirmation with strict display rules.

    DISPLAY: bbox is None when object is NOT detected this frame.
    CONFIRMATION: EMA confidence >= high_thr AND hits >= min_hits.
    SPATIAL: large jumps (IoU < 0.15) must re-stabilize over 3 frames.
    """

    def __init__(self, configs: Dict[str, ClassStabilizerConfig] = None):
        self.configs = configs or DEFAULT_CONFIGS
        self._state: Dict[str, _ClassState] = {}

    def reset(self):
        self._state.clear()

    def update(self, detections: List) -> Dict[str, Dict]:
        best_per_class: Dict[str, Tuple] = {}
        for d in detections:
            cls = d.cls_name
            if cls not in self.configs:
                continue
            conf = d.confidence
            if cls not in best_per_class or conf > best_per_class[cls][0]:
                src = getattr(d, 'source', 'yolo')
                best_per_class[cls] = (conf, d.bbox, d.center, src)

        result = {}
        for cls_name, cfg in self.configs.items():
            if cls_name not in self._state:
                self._state[cls_name] = _ClassState()
            st = self._state[cls_name]
            current = best_per_class.get(cls_name)

            if current is not None:
                conf, bbox, center, source = current

                # Spatial consistency
                if st.current_bbox is not None:
                    iou = _iou(bbox, st.current_bbox)
                    if iou < 0.08:
                        # Large position jump — verify via candidate mechanism
                        # High confidence → accept immediately (real movement)
                        if conf >= cfg.high_thr:
                            st.current_bbox = bbox
                            st.current_conf = conf
                            st.current_center = center
                            st.source = source
                            st.ema_conf = (1 - cfg.ema_alpha) * st.ema_conf + cfg.ema_alpha * conf
                            st.candidate_bbox = None
                            st.candidate_hits = 0
                            if conf >= cfg.low_thr:
                                st.hits += 1
                                st.lost = 0
                        elif st.candidate_bbox is not None and _iou(bbox, st.candidate_bbox) >= 0.4:
                            st.candidate_hits += 1
                            st.candidate_bbox = bbox
                            st.lost = max(0, st.lost - 1)
                            if st.candidate_hits >= 3:
                                # Verified — accept new position
                                st.hits = max(st.hits, 1)
                                st.lost = 0
                                st.ema_conf = 0.5 * st.ema_conf + 0.5 * conf
                                st.current_bbox = bbox
                                st.current_conf = conf
                                st.current_center = center
                                st.source = source
                                st.candidate_bbox = None
                                st.candidate_hits = 0
                        else:
                            # New candidate — don't display yet
                            st.candidate_bbox = bbox
                            st.candidate_hits = 1
                            st.lost = max(0, st.lost - 1)
                    else:
                        # Normal frame-to-frame jitter — accept
                        st.candidate_bbox = None
                        st.candidate_hits = 0
                        st.current_bbox = bbox
                        st.current_conf = conf
                        st.current_center = center
                        st.source = source
                        st.ema_conf = (1 - cfg.ema_alpha) * st.ema_conf + cfg.ema_alpha * conf
                        if conf >= cfg.low_thr:
                            st.hits += 1
                            st.lost = 0
                        else:
                            st.lost += 1
                        if st.hits >= cfg.min_hits and st.ema_conf >= cfg.high_thr:
                            st.confirmed = True
                else:
                    # First detection or reappearing after brief loss
                    st.current_bbox = bbox
                    st.current_conf = conf
                    st.current_center = center
                    st.source = source
                    # Blend with previous EMA if it exists (don't reset to raw)
                    if st.ema_conf > 0:
                        st.ema_conf = (1 - cfg.ema_alpha) * st.ema_conf + cfg.ema_alpha * conf
                    else:
                        st.ema_conf = conf
                    if conf >= cfg.low_thr:
                        st.hits += 1
                        st.lost = 0
                    else:
                        st.lost += 1
                    if st.hits >= cfg.min_hits and st.ema_conf >= cfg.high_thr:
                        st.confirmed = True
            else:
                st.lost += 1
                st.hits = max(0, st.hits - 1)
                st.ema_conf *= 0.85
                st.candidate_bbox = None
                st.candidate_hits = 0
                # 看不到物体就不能显示框 — clear immediately
                st.current_bbox = None
                st.current_conf = 0.0
                st.current_center = None
                if st.lost > cfg.max_lost:
                    st.confirmed = False
                    st.ema_conf = 0.0
                    st.hits = 0

            result[cls_name] = {
                "confirmed": st.confirmed,
                "bbox": list(st.current_bbox) if st.current_bbox else None,
                "conf": round(st.current_conf, 3),
                "ema_conf": round(st.ema_conf, 3),
                "source": st.source,
                "hits": st.hits,
                "lost": st.lost,
            }

        return result

    def get_display_detections(self) -> List[Dict]:
        """Confirmed AND currently-detected objects only."""
        result = []
        for cls_name, st in self._state.items():
            if st.confirmed and st.current_bbox is not None:
                entry = {
                    "class": cls_name,
                    "conf": round(st.current_conf, 3),
                    "bbox": [int(v) for v in st.current_bbox],
                }
                if cls_name == "charger":
                    entry["method"] = st.source
                result.append(entry)
        return result

    def is_class_confirmed(self, cls_name: str) -> bool:
        st = self._state.get(cls_name)
        return st is not None and st.confirmed

    def get_class_bbox(self, cls_name: str) -> Optional[Tuple]:
        st = self._state.get(cls_name)
        if st and st.confirmed and st.current_bbox is not None:
            return st.current_bbox
        return None

    def get_class_ema_conf(self, cls_name: str) -> float:
        st = self._state.get(cls_name)
        return st.ema_conf if st else 0.0

    @property
    def confirmed_classes(self) -> List[str]:
        return [name for name, st in self._state.items()
                if st.confirmed and st.current_bbox is not None]
