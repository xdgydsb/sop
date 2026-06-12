"""
SOP 实时动作检测 — 统一本地运行器
====================================
单进程、无 WebSocket、一条命令跑起来。

Usage:
  python local_run.py --camera hik       # 海康工业相机
  python local_run.py --camera webcam    # USB 摄像头
  python local_run.py --video demo.mp4   # 视频文件

依赖: pip install -r requirements.txt
"""
import sys
import os
import time
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
from engine.yolo_detector import YOLODetector
from engine.hand_detector import HandDetector
from engine.detection_stabilizer import DetectionStabilizer
from engine.box_state_stabilizer import BoxStateStabilizer
from engine.object_state_tracker import ObjectStateTracker, ObjectState
from engine.event_detector import EventDetector, EVENT_SEQUENCE, EVENT_TO_STEP_ID
from engine.action_segmenter import ActionSegmenter
from engine.sop_fsm import SOPStateMachine, SOPStep, STEP_NAMES

# ── Optional: TemporalPredictorV2 ──
_TEMPORAL_AVAILABLE = True
try:
    from engine.temporal_predictor_v2 import TemporalPredictorV2
    from tools.extract_features_v2_90 import FeatureExtractorV2Standalone
except ImportError:
    _TEMPORAL_AVAILABLE = False

# ── Optional: Hikvision MVS SDK ──
_HIK_AVAILABLE = True
try:
    # Auto-detect MVS SDK path: env var → common install locations
    _MVS_ROOT = os.environ.get("MVS_ROOT", "")
    if not _MVS_ROOT or not os.path.isdir(_MVS_ROOT):
        _CANDIDATES = [
            r"D:\develop\MVS",
            r"C:\Program Files (x86)\MVS",
            r"C:\Program Files\MVS",
            r"D:\MVS",
        ]
        for _c in _CANDIDATES:
            if os.path.isdir(_c):
                _MVS_ROOT = _c
                break
        else:
            raise ImportError("MVS SDK not found — set MVS_ROOT env var or install MVS")
    _MVS_SDK_PYTHON = os.path.join(_MVS_ROOT, "Development", "Samples", "Python")
    _MVS_MV_IMPORT = os.path.join(_MVS_SDK_PYTHON, "MvImport")
    _MVS_LIB_PATHS = [
        os.path.join(_MVS_ROOT, "Development", "Libraries", "win64"),
        os.path.join(_MVS_ROOT, "Runtime", "win64"),
    ]
    for _p in _MVS_LIB_PATHS:
        if os.path.isdir(_p):
            try:
                os.add_dll_directory(_p)
            except Exception:
                pass
    sys.path.insert(0, _MVS_SDK_PYTHON)
    sys.path.insert(0, _MVS_MV_IMPORT)
    import ctypes
    from ctypes import c_ubyte, POINTER, byref, cast, memset, sizeof
    from MvCameraControl_class import (MvCamera, MV_CC_DEVICE_INFO_LIST,
                                        MVCC_INTVALUE, MVCC_FLOATVALUE, MV_FRAME_OUT)
    from CameraParams_header import MV_CC_DEVICE_INFO, MV_USB_DEVICE, MV_GIGE_DEVICE
    from MvErrorDefine_const import MV_OK
    import PixelType_header as px
except (ImportError, OSError):
    _HIK_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# ROI / Validation helpers (mirror inference_server.py)
# ═══════════════════════════════════════════════════════════════

def load_roi_config(config_path: str = None) -> Dict:
    """Load ROI configuration from YAML file (same as inference_server.py)."""
    import yaml
    if config_path is None:
        config_path = str(ROOT / "configs" / "realcam_sop.yaml")
    path = Path(config_path)
    if not path.exists():
        print(f"[Config] WARNING: {path} not found, using defaults")
        return {
            "box_roi": [300, 150, 950, 650],
            "box_inner_roi": [380, 200, 870, 600],
            "earphone_init_roi": [100, 200, 350, 450],
            "charger_init_roi": [600, 100, 900, 350],
            "green_bag_init_roi": [50, 400, 300, 650],
        }
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[Config] Loaded ROI config from {path}")
    return cfg

# Default ROIs (overridden by YAML config if available)
DEFAULT_BOX_ROI = (300, 150, 950, 650)       # matched to realcam_sop.yaml
DEFAULT_BOX_INNER_ROI = (380, 200, 870, 600)  # matched to realcam_sop.yaml
DEFAULT_INIT_ROI_MAP = {
    "earphone": (100, 200, 350, 450),
    "charger": (600, 100, 900, 350),
    "green_bag": (50, 400, 300, 650),
}

FPS_TARGET = 20.0
T = 48
STRIDE = 4

STEP_NAMES_CN = {
    0: "等待开始", 1: "S1 打开纸盒", 2: "S2 放入耳机盒",
    3: "S3 放入充电插头", 4: "S4 放入绿色袋子", 5: "S5 关闭纸盒", 6: "完成", 7: "错误",
}


def _bbox_area(bbox):
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    return w * h


def _iou_with_roi(bbox, roi):
    x1 = max(bbox[0], roi[0]); y1 = max(bbox[1], roi[1])
    x2 = min(bbox[2], roi[2]); y2 = min(bbox[3], roi[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = _bbox_area(bbox)
    a2 = (roi[2] - roi[0]) * (roi[3] - roi[1])
    return inter / max(a1 + a2 - inter, 1.0)


def _validate_box_detection(bbox, roi, min_area=800, max_area=300000,
                             min_overlap=0.10):
    """Validate box detection. Lower min_overlap (0.10) for portability —
    different camera setups will have different box positions.
    Run tools/calibrate_rois.py for best accuracy."""
    area = _bbox_area(bbox)
    if area < min_area:
        return "small_box_area"
    if area > max_area:
        return "large_box_area"
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return "bad_box_aspect"
    ar = w / max(h, 1)
    if not (0.3 < ar < 3.0):
        return "bad_box_aspect"
    x1 = max(bbox[0], roi[0]); y1 = max(bbox[1], roi[1])
    x2 = min(bbox[2], roi[2]); y2 = min(bbox[3], roi[3])
    if x2 <= x1 or y2 <= y1:
        return "outside_box_roi"
    inter = (x2 - x1) * (y2 - y1)
    if inter / max(area, 1.0) < min_overlap:
        return "outside_box_roi"
    return ""


def _suppress_objects_in_closed_box(det_summary, box_state, box_bbox):
    if box_state != "closed" or not box_bbox:
        return det_summary
    bx1, by1, bx2, by2 = box_bbox
    margin = 15
    result = []
    for d in det_summary:
        cls_name = d.get("class", "")
        if cls_name == "box_closed":
            result.append(d)
            continue
        if cls_name == "box_open":
            # box_open is impossible when box state is CLOSED — suppress
            continue
        b = d.get("bbox")
        if b is None:
            result.append(d)
            continue
        cx = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2
        if bx1 - margin <= cx <= bx2 + margin and by1 - margin <= cy <= by2 + margin:
            continue
        result.append(d)
    return result


def _filter_charger_box_overlap(detections, box_bbox, containment_threshold=0.65):
    """Remove charger detections that are mostly INSIDE the box bbox.

    Charger sitting ON the box will have edge overlap only (containment ~0.1-0.3).
    A false positive on the box itself has high containment (>0.65).
    """
    filtered = []
    for d in detections:
        if d.cls_name == "charger" and box_bbox:
            dx1 = max(d.bbox[0], box_bbox[0])
            dy1 = max(d.bbox[1], box_bbox[1])
            dx2 = min(d.bbox[2], box_bbox[2])
            dy2 = min(d.bbox[3], box_bbox[3])
            if dx2 > dx1 and dy2 > dy1:
                inter_area = (dx2 - dx1) * (dy2 - dy1)
                charger_area = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
                if charger_area > 0:
                    containment = inter_area / charger_area
                    if containment > containment_threshold:
                        continue
        filtered.append(d)
    return filtered


# ═══════════════════════════════════════════════════════════════
# Camera sources
# ═══════════════════════════════════════════════════════════════

class WebcamSource:
    def __init__(self, camera_id=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, _ = self.cap.read()
        if not ok:
            raise RuntimeError(f"无法打开摄像头 ID={camera_id}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Webcam] {self.width}x{self.height}")

    def read(self):
        ok, frame = self.cap.read()
        return (frame, time.time()) if ok else (None, None)

    def close(self):
        self.cap.release()


class VideoSource:
    def __init__(self, video_path):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Video] {video_path} ({self.width}x{self.height}, "
              f"{self.total} frames, {self.fps:.1f} fps)")

    def read(self):
        ok, frame = self.cap.read()
        return (frame, time.time()) if ok else (None, None)

    def close(self):
        self.cap.release()


class HikvisionSource:
    """Hikvision MVS SDK camera wrapper (MV-CS050-10UC)."""

    def __init__(self):
        if not _HIK_AVAILABLE:
            raise RuntimeError(
                "海康 MVS SDK 未安装或路径不正确。\n"
                "  设置环境变量 MVS_ROOT 指向 MVS 安装目录，\n"
                "  或使用: --camera webcam"
            )
        self.cam = None
        self._open()

    def _open(self):
        MvCamera.MV_CC_Initialize()
        dl = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, dl)
        if dl.nDeviceNum == 0:
            raise RuntimeError("未找到海康相机 (MVS 软件关闭了吗?)")

        selected_idx = -1
        target_serial = "DA9562204"
        for i in range(dl.nDeviceNum):
            dev = cast(dl.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if dev.nTLayerType == MV_USB_DEVICE:
                info = dev.SpecialInfo.stUsb3VInfo
                serial = ""
                for ch in info.chSerialNumber:
                    if ch == 0: break
                    serial += chr(ch)
                model = self._dec(info.chModelName)
                print(f"  [{i}] USB: {model} SN:{serial}")
                if serial == target_serial:
                    selected_idx = i
                if selected_idx < 0:
                    selected_idx = i

        if selected_idx < 0:
            raise RuntimeError("未找到 USB 相机")

        st_dev = cast(dl.pDeviceInfo[selected_idx],
                      POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam = MvCamera()
        self.cam.MV_CC_CreateHandle(st_dev)
        self.cam.MV_CC_OpenDevice()
        self.cam.MV_CC_SetEnumValue("TriggerMode", 0)

        st_w = MVCC_INTVALUE(); st_h = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("Width", st_w)
        self.cam.MV_CC_GetIntValue("Height", st_h)
        self.width, self.height = st_w.nCurValue, st_h.nCurValue
        print(f"[Hikvision] {self.width}x{self.height}")

        st_exp = MVCC_FLOATVALUE()
        self.cam.MV_CC_GetFloatValue("ExposureTime", st_exp)
        if st_exp.fCurValue < 15000:
            self.cam.MV_CC_SetFloatValue("ExposureTime", 35000.0)
            print(f"  曝光: {st_exp.fCurValue:.0f} -> 35000 us")

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            raise RuntimeError(f"StartGrabbing FAILED: {ret}")

        self._buf_size = st_w.nCurValue * st_h.nCurValue * 3
        print(f"  ✓ 取流已开始")

    @staticmethod
    def _dec(c_ubyte_val):
        c_char_p = ctypes.cast(c_ubyte_val, ctypes.c_char_p)
        try:
            return c_char_p.value.decode('gbk')
        except Exception:
            return str(c_char_p.value) if c_char_p.value else ""

    def read(self):
        st_frame = MV_FRAME_OUT()
        memset(byref(st_frame), 0, sizeof(MV_FRAME_OUT))
        ret = self.cam.MV_CC_GetImageBuffer(st_frame, 1000)
        if ret != MV_OK or st_frame.pBufAddr is None:
            return None, None

        try:
            frame = self._convert(st_frame)
            return frame, time.time()
        finally:
            self.cam.MV_CC_FreeImageBuffer(st_frame)

    def _convert(self, st_frame):
        fi = st_frame.stFrameInfo
        w, h, fmt = fi.nWidth, fi.nHeight, fi.enPixelType
        buf = st_frame.pBufAddr
        if fmt == px.PixelType_Gvsp_Mono8:
            raw = np.ctypeslib.as_array(
                ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
            return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_GRAY2BGR)
        if fmt == px.PixelType_Gvsp_RGB8_Packed:
            raw = np.ctypeslib.as_array(
                ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h * 3))).contents)
            return cv2.cvtColor(raw.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
        if fmt == px.PixelType_Gvsp_BGR8_Packed:
            raw = np.ctypeslib.as_array(
                ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h * 3))).contents)
            return raw.reshape((h, w, 3))
        if fmt == px.PixelType_Gvsp_BayerRG8:
            raw = np.ctypeslib.as_array(
                ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
            return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerRG2BGR)
        if fmt == px.PixelType_Gvsp_BayerGB8:
            raw = np.ctypeslib.as_array(
                ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
            return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerGB2BGR)
        if fmt == px.PixelType_Gvsp_BayerGR8:
            raw = np.ctypeslib.as_array(
                ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
            return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerGR2BGR)
        if fmt == px.PixelType_Gvsp_BayerBG8:
            raw = np.ctypeslib.as_array(
                ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
            return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerBG2BGR)
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * fi.nFrameLen)).contents)
        return raw.reshape((h, w, 3))

    def close(self):
        if self.cam is None:
            return
        try:
            self.cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            self.cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            self.cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass


def create_source(args):
    """Create camera/video source from args."""
    if args.video:
        return VideoSource(args.video)
    if args.camera == "hik":
        return HikvisionSource()
    return WebcamSource(camera_id=args.webcam_id)


# ═══════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════

class SOPLocalPipeline:
    """Full SOP pipeline — mirrors inference_server.py Stage 3."""

    def __init__(self, args):
        self.device = args.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Pipeline] device={self.device}")

        # YOLO
        yolo_path = args.yolo or str(ROOT / "models" / "yolo_final_v1.pt")
        if not os.path.isfile(yolo_path):
            raise FileNotFoundError(f"YOLO 模型不存在: {yolo_path}")
        self.yolo = YOLODetector(yolo_path, conf_thresh=args.conf,
                                 device=self.device, imgsz=args.imgsz,
                                 use_tracker=True, bbox_ema_alpha=0.60,
                                 use_clahe=False)
        # MediaPipe
        self.hand = HandDetector()
        self.hand.palm_velocity.clear()

        # ── Stabilizers ──
        self.det_stabilizer = DetectionStabilizer()
        self.box_stabilizer = BoxStateStabilizer(
            open_thr=0.45, closed_thr=0.35, vote_window=5, vote_need=3)

        # ── ObjectStateTracker ──
        self.obj_tracker = ObjectStateTracker(
            stable_in_box_min=5, outside_box_min=5, hand_absent_min=5,
            lost_max=90, occluded_max=20)

        # ── EventDetector ──
        # Use updated defaults from engine fixes (was closed_stable_frames=30):
        # closed_stable_frames=15, hand_absent_needed=8,
        # fast_closed_needed=10, fast_hand_absent_needed=5
        self.event_detector = EventDetector(
            closed_stable_frames=15, early_close_confirm_frames=25,
            event_cooldown=10)

        # ── FSM ──
        self.fsm = SOPStateMachine(timeout=90.0, min_step_duration=0.2)
        self.STEPS = SOPStep

        # ── ActionSegmenter ──
        self.action_seg = ActionSegmenter()

        # ── Temporal (optional) ──
        self.predictor = None
        self.extractor = None
        if not args.no_temporal and _TEMPORAL_AVAILABLE:
            model_path = args.model or str(
                ROOT / "models" / "temporal" / "v2_90_tcn_bigru" / "checkpoints" / "best.pt")
            if os.path.isfile(model_path):
                print(f"[Pipeline] 加载时序模型: {model_path}")
                self.extractor = FeatureExtractorV2Standalone(
                    self.yolo, self.hand, ema_alpha=0.5, hold_frames=5)
                self.predictor = TemporalPredictorV2(
                    model_path, device=self.device, T=T, stride=STRIDE)
            else:
                print(f"[Pipeline] 时序模型不存在，跳过")
        else:
            print(f"[Pipeline] 时序模型: 跳过 (--no-temporal 或 模块不可用)")

        # ── ROI ──
        # Load from YAML config if available (matches inference_server.py behavior)
        cfg = load_roi_config()
        self.BOX_ROI = tuple(cfg["box_roi"])
        self.BOX_INNER_ROI = tuple(cfg["box_inner_roi"])
        self.INIT_ROI_MAP = {
            "earphone": tuple(cfg.get("earphone_init_roi", DEFAULT_INIT_ROI_MAP["earphone"])),
            "charger": tuple(cfg.get("charger_init_roi", DEFAULT_INIT_ROI_MAP["charger"])),
            "green_bag": tuple(cfg.get("green_bag_init_roi", DEFAULT_INIT_ROI_MAP["green_bag"])),
        }

        # ── State ──
        self.mode = "PREVIEW"  # PREVIEW → ARMED → RUNNING → COMPLETE
        self._expected_event = EVENT_SEQUENCE[0]  # "box_opened"
        self.EVENT_SEQUENCE = EVENT_SEQUENCE
        self._accepted_events: List[str] = []
        self._max_fsm_path: List[int] = []
        self._fsm_events: List[Dict] = []
        self._latest_top3 = []
        self._latest_step_probs = []

        self._count = 0
        self._feat_count = 0
        self._last_feat_time = -1.0
        self._feat_interval = 1.0 / FPS_TARGET
        self._s5_complete_frames = 0
        self._wrong_active_consec = 0
        self._box_was_open = False
        self._action_result = {}
        self._start_time = time.time()

        # FPS tracking
        self._fps_times = []
        self.disp_fps = 0.0

    def process_frame(self, frame, timestamp):
        """Process one frame. Returns (annotated_frame, det_summary, info_dict)."""
        self._count += 1
        h, w = frame.shape[:2]

        # FPS
        self._fps_times.append(time.time())
        if len(self._fps_times) > 30:
            self._fps_times = self._fps_times[-30:]
        if len(self._fps_times) >= 2:
            dt = self._fps_times[-1] - self._fps_times[0]
            self.disp_fps = (len(self._fps_times) - 1) / dt if dt > 0 else 0

        # ── YOLO + MediaPipe ──
        detections = self.yolo.detect(frame)
        hands = self.hand.detect(frame)

        # ── Box ROI validation ──
        valid_detections = []
        for d in detections:
            if d.cls_name in ("box_open", "box_closed"):
                reason = _validate_box_detection(d.bbox, self.BOX_ROI)
                if not reason:
                    valid_detections.append(d)
            else:
                valid_detections.append(d)
        detections = valid_detections

        # ── Box state: best per class ──
        best_open = max((d for d in detections if d.cls_name == "box_open"),
                        key=lambda d: d.confidence, default=None)
        best_closed = max((d for d in detections if d.cls_name == "box_closed"),
                          key=lambda d: d.confidence, default=None)
        open_conf = best_open.confidence if best_open else 0.0
        closed_conf = best_closed.confidence if best_closed else 0.0
        open_bbox = best_open.bbox if best_open else None
        closed_bbox = best_closed.bbox if best_closed else None

        # ── BoxStateStabilizer ──
        self.box_stabilizer.update(open_conf, closed_conf, open_bbox, closed_bbox)
        box_state_str = self.box_stabilizer.state_str
        box_bbox = self.box_stabilizer.bbox

        # Track box_was_open
        if box_state_str == "open":
            self._box_was_open = True

        # ── DetectionStabilizer (objects only) ──
        object_dets = [d for d in detections
                       if d.cls_name not in ("box_open", "box_closed")]

        # ── Focused box-interior crop detection ──
        # When an object is being placed and full-frame YOLO didn't detect it,
        # run a focused higher-resolution pass on just the box area at lower
        # confidence (0.06). Objects inside the box are harder to detect due to
        # different lighting/background and partial occlusion by box walls.
        _OBJ_FOR_EVENT = {
            "earphone_in_box": "earphone",
            "charger_in_box": "charger",
            "green_bag_in_box": "green_bag",
        }
        if (self.mode == "RUNNING" and box_state_str == "open"
                and self._expected_event in _OBJ_FOR_EVENT):
            expected_obj = _OBJ_FOR_EVENT[self._expected_event]
            obj_state = self.obj_tracker.get_state(expected_obj)
            if obj_state in (ObjectState.LEFT_INIT, ObjectState.VISIBLE_IN_BOX,
                             ObjectState.OCCLUDED):
                yolo_has_obj = any(
                    d.cls_name == expected_obj and d.confidence >= 0.10
                    for d in object_dets
                )
                if not yolo_has_obj:
                    crop_src = box_bbox if box_bbox else self.BOX_ROI
                    if crop_src:
                        cx1, cy1, cx2, cy2 = crop_src
                        margin = int((cx2 - cx1) * 0.08)
                        crop_roi = (cx1 - margin, cy1 - margin,
                                    cx2 + margin, cy2 + margin)
                        crop_dets = self.yolo.detect_crop(
                            frame, crop_roi, conf_thresh=0.06)
                        crop_matches = [d for d in crop_dets
                                       if d.cls_name == expected_obj]
                        if crop_matches:
                            if self._count % 20 == 0:
                                best = max(crop_matches, key=lambda d: d.confidence)
                                print(f"[FocusedCrop] {expected_obj} found "
                                      f"conf={best.confidence:.3f}", flush=True)
                            object_dets.extend(crop_matches)

        stable = self.det_stabilizer.update(object_dets)

        # ── Filter charger false positives (ONLY when box is CLOSED) ──
        if box_bbox and box_state_str == "closed":
            detections = _filter_charger_box_overlap(detections, box_bbox)

        # ── Display detections ──
        det_summary = self.det_stabilizer.get_display_detections()

        # Remove box entries from DetectionStabilizer — box is not drawn as a bbox.
        # Box state is shown as a small text indicator only (see draw_status).
        det_summary = [d for d in det_summary
                       if d.get("class") not in ("box_open", "box_closed")]

        det_summary = _suppress_objects_in_closed_box(
            det_summary, box_state_str, box_bbox)

        # ── Mode-gated execution ──
        now = time.time()
        do_feat = (now - self._last_feat_time >= self._feat_interval
                   or self._last_feat_time < 0)

        step_name = "IDLE"
        alarm = None
        is_complete = False

        # Box inner ROI
        box_inner_roi = self.BOX_INNER_ROI
        if box_bbox is not None:
            bx1, by1, bx2, by2 = box_bbox
            bw, bh = bx2 - bx1, by2 - by1
            inset_x = bw * 0.15
            inset_y = bh * 0.15
            box_inner_roi = (bx1 + inset_x, by1 + inset_y,
                             bx2 - inset_x, by2 - inset_y)

        # Update event evidence
        box_state_info = self.box_stabilizer.get_state_info()
        self.event_detector.update_evidence(box_state_info)

        if self.mode in ("ARMED", "RUNNING"):
            # hand_in_box check
            hand_in_box = False
            box_ref = box_bbox if box_bbox is not None else self.BOX_ROI
            if hands and box_ref:
                bx1, by1, bx2, by2 = box_ref
                bw, bh = bx2 - bx1, by2 - by1
                mx, my = bw * 0.3, bh * 0.3
                ebx1, eby1 = bx1 - mx, by1 - my
                ebx2, eby2 = bx2 + mx, by2 + my
                for hand_obj in hands:
                    hx1, hy1, hx2, hy2 = hand_obj.bbox
                    ox1 = max(ebx1, hx1); oy1 = max(eby1, hy1)
                    ox2 = min(ebx2, hx2); oy2 = min(eby2, hy2)
                    if ox2 > ox1 and oy2 > oy1:
                        hand_in_box = True
                        break

            # ObjectStateTracker
            for obj_name in ["earphone", "charger", "green_bag"]:
                obj_info = stable.get(obj_name, {})
                detected = obj_info.get("bbox") is not None
                bbox = obj_info.get("bbox")
                conf = obj_info.get("conf", 0.0)
                self.obj_tracker.update(
                    obj_name, detected, bbox, conf,
                    box_inner_roi=box_inner_roi,
                    init_roi=self.INIT_ROI_MAP.get(obj_name),
                    hand_in_box=hand_in_box,
                )

            # TemporalPredictor (auxiliary)
            temporal_aux = None
            if do_feat and self.predictor is not None:
                self._last_feat_time = now
                self._feat_count += 1
                feat_dets = [d for d in detections
                            if d.cls_name not in ("box_open", "box_closed")]
                if feat_dets and box_bbox:
                    self.extractor.compute_interaction(feat_dets, hands, box_bbox, h, w)
                feat = self.extractor.extract(frame, feat_dets, hands,
                                              box_bbox, box_state_str)
                self.predictor.predict(feat)
                mature = self.predictor.get_latest_mature_prediction()
                if mature is not None:
                    step_probs_arr = [round(float(p), 4) for p in mature["step_probs"]]
                    self._latest_top3 = [(int(i), round(float(p), 4))
                                         for i, p in mature["top3"]]
                    self._latest_step_probs = step_probs_arr
                    temporal_aux = {
                        "step_probs": mature["step_probs"],
                        "confidence": mature["confidence"],
                        "top3": mature["top3"],
                    }

            # EventDetector
            event = self.event_detector.detect(
                box_state=self.box_stabilizer.state,
                box_state_info=box_state_info,
                obj_tracker=self.obj_tracker,
                expected_event=self._expected_event,
                accepted_events=self._accepted_events,
                temporal_aux=temporal_aux,
                hand_in_box=hand_in_box,
            )

            if event is not None:
                if self.mode == "ARMED":
                    if event.event_name == "box_opened":
                        self.mode = "RUNNING"
                        self._accept_event(event, now)
                        print(f"[SOP] MODE → RUNNING (box_opened)")
                elif self.mode == "RUNNING":
                    if event.event_name == "early_close_alarm":
                        self.event_detector.mark_event_accepted(event.event_name)
                        self._fsm_events.append({
                            "step": 7, "step_name": "EARLY_CLOSE_ALARM",
                            "event_name": event.event_name,
                            "confidence": round(event.confidence, 4),
                            "is_correct": False, "has_error": True,
                            "error_type": "EARLY_CLOSE",
                            "message": "Box closed but S2/S3/S4 not done!",
                        })
                        self.mode = "ERROR"
                    elif event.event_name == self._expected_event:
                        self._accept_event(event, now)
                        if self.fsm.current_step == self.STEPS.COMPLETE:
                            self.mode = "COMPLETE"
                            print(f"[SOP] ★ COMPLETE!")

            # ActionSegmenter
            obj_summary = self.obj_tracker.get_summary()
            self._action_result = self.action_seg.update(
                expected_event=self._expected_event,
                box_state_str=box_state_str,
                box_previous_state_str=box_state_info["previous_state"],
                obj_tracker_summary=obj_summary,
                accepted_events=self._accepted_events,
                now=now,
                open_evidence=self.event_detector.open_evidence_frames,
                closed_evidence=self.event_detector.closed_evidence_frames,
            )

            # Wrong-order alarm
            if self._action_result.get("action_phase") == "WRONG_ACTIVE":
                self._wrong_active_consec += 1
                if self._wrong_active_consec >= 15:
                    wrong_obj = self._action_result.get("wrong_object", "?")
                    self._fsm_events.append({
                        "step": 7, "step_name": "WRONG_ORDER",
                        "event_name": "wrong_active",
                        "confidence": 0.85,
                        "is_correct": False, "has_error": True,
                        "error_type": "WRONG_ORDER",
                        "message": f"Wrong object in box: {wrong_obj}",
                    })
                    self.mode = "ERROR"
            else:
                self._wrong_active_consec = max(0, self._wrong_active_consec - 1)

            # S5 → COMPLETE auto-advance
            if self.fsm.current_step == self.STEPS.S5_CLOSE:
                if box_state_str == "closed" and not hand_in_box:
                    self._s5_complete_frames += 1
                elif hand_in_box:
                    self._s5_complete_frames = 0
                if self._s5_complete_frames >= 10:
                    self.fsm.current_step = self.STEPS.COMPLETE
                    self.mode = "COMPLETE"
                    self._fsm_events.append({
                        "step": 6, "step_name": "COMPLETE",
                        "event_name": "box_closed_confirmed",
                        "confidence": 0.95,
                        "is_correct": True, "has_error": False,
                        "error_type": "",
                        "message": "S5→COMPLETE: box stably closed, hand absent",
                    })
                    print("[SOP] ★ S5→COMPLETE auto-advance")

            step_name = STEP_NAMES_CN[self.fsm.current_step.value]
            is_complete = (self.fsm.current_step == self.STEPS.COMPLETE)

            errors = [e for e in self._fsm_events if e.get("has_error")]
            if errors:
                alarm = {"type": errors[-1]["error_type"],
                         "message": errors[-1]["message"]}

        elif do_feat and self.predictor is not None:
            # PREVIEW: temporal warmup only
            self._last_feat_time = now
            self._feat_count += 1
            feat_dets = [d for d in detections
                        if d.cls_name not in ("box_open", "box_closed")]
            if box_bbox:
                self.extractor.compute_interaction(feat_dets, hands, box_bbox, h, w)
            feat = self.extractor.extract(frame, feat_dets, hands,
                                          box_bbox, box_state_str)
            self.predictor.predict(feat)
            mature = self.predictor.get_latest_mature_prediction()
            if mature is not None:
                self._latest_top3 = [(int(i), round(float(p), 4))
                                     for i, p in mature["top3"]]
                self._latest_step_probs = [round(float(p), 4)
                                           for p in mature["step_probs"]]

        # ── Build info dict ──
        info = {
            "fps": self.disp_fps,
            "mode": self.mode,
            "step_name": step_name,
            "step_id": self.fsm.current_step.value,
            "box_state": box_state_str,
            "box_conf": self.box_stabilizer.open_conf if box_state_str == "open"
                        else self.box_stabilizer.closed_conf,
            "expected_event": self._expected_event,
            "accepted": list(self._accepted_events),
            "fsm_path": list(self._max_fsm_path),
            "is_complete": is_complete,
            "alarm": alarm,
            "count": self._count,
            "feat_count": self._feat_count,
            "temporal_available": self.predictor is not None,
            "latest_top3": self._latest_top3,
            "latest_step_probs": self._latest_step_probs,
            "action_phase": self._action_result.get("action_phase", "WAITING"),
            "current_action": self._action_result.get("current_action", "none"),
        }

        return det_summary, info, box_bbox, box_state_str, hands

    def start(self):
        """Begin SOP detection (ARMED mode)."""
        self.mode = "ARMED"
        self._accepted_events.clear()
        self._max_fsm_path.clear()
        self._fsm_events.clear()
        self._expected_event = EVENT_SEQUENCE[0]
        self._s5_complete_frames = 0
        self._wrong_active_consec = 0
        self._box_was_open = False
        self.obj_tracker.reset()
        self.action_seg.reset()
        self.event_detector.reset()
        self.fsm = SOPStateMachine(timeout=90.0, min_step_duration=0.2)
        if self.predictor:
            self.predictor.reset()
        print("[SOP] ARMED — waiting for box_opened...")

    def reset(self):
        """Reset pipeline to PREVIEW."""
        self.mode = "PREVIEW"
        self._accepted_events.clear()
        self._max_fsm_path.clear()
        self._fsm_events.clear()
        self._expected_event = EVENT_SEQUENCE[0]
        self._s5_complete_frames = 0
        self._wrong_active_consec = 0
        self._box_was_open = False
        self.det_stabilizer.reset()
        self.box_stabilizer.reset()
        self.obj_tracker.reset()
        self.action_seg.reset()
        self.event_detector.reset()
        self.yolo.reset_tracking()
        self.fsm = SOPStateMachine(timeout=90.0, min_step_duration=0.2)
        if self.predictor:
            self.predictor.reset()
        print("[SOP] RESET → PREVIEW")

    def _accept_event(self, event, now):
        event_name = event.event_name
        if event_name in self._accepted_events:
            return
        fs_result = self.fsm.validate_event(event_name, event.confidence, timestamp=now)
        self._fsm_events.append({
            "step": fs_result.step_id,
            "step_name": STEP_NAMES_CN.get(fs_result.step_id, f"Step{fs_result.step_id}"),
            "event_name": event_name,
            "confidence": round(event.confidence, 4),
            "is_correct": fs_result.is_correct,
            "has_error": fs_result.has_error,
            "error_type": fs_result.error_type,
            "message": fs_result.message,
        })
        if not fs_result.is_correct:
            print(f"[SOP] Event REJECTED by FSM: {event_name} → {fs_result.message}")
            return
        self.event_detector.mark_event_accepted(event_name, obj_tracker=self.obj_tracker)
        self._accepted_events.append(event_name)
        current_path = [EVENT_TO_STEP_ID.get(e, 0) for e in self._accepted_events
                        if EVENT_TO_STEP_ID.get(e, 0) > 0]
        if len(current_path) > len(self._max_fsm_path):
            self._max_fsm_path = current_path
        self._advance_expected_event()
        from engine.action_segmenter import OBJECT_FOR_EVENT
        next_obj = OBJECT_FOR_EVENT.get(self._expected_event)
        if next_obj:
            self.obj_tracker.reset_object(next_obj)
        print(f"[SOP] Event ACCEPTED: {event_name} path={current_path} "
              f"next={self._expected_event}")

    def _advance_expected_event(self):
        try:
            idx = self.EVENT_SEQUENCE.index(self._expected_event)
            if idx + 1 < len(self.EVENT_SEQUENCE):
                self._expected_event = self.EVENT_SEQUENCE[idx + 1]
        except ValueError:
            pass


# ═══════════════════════════════════════════════════════════════
# Display — matches hik_stream_client.py overlay style
# ═══════════════════════════════════════════════════════════════

COLORS = {
    "box_open": (0, 255, 0),
    "box_closed": (0, 165, 255),
    "earphone": (255, 255, 0),
    "charger": (255, 200, 0),
    "green_bag": (0, 200, 100),
}


def _draw_sop_progress(frame, current_step_id, fsm_path, is_complete, alarm,
                       mode="PREVIEW", model_pred=-1, step_probs=None,
                       action_phase="WAITING", current_action="none"):
    """Draw S1-S5 progress bar at top-center of frame.

    Rule: PREVIEW/ARMED → all neutral gray (no predictions shown — they are noise)
          RUNNING → green for completed, cyan for active, dark for future
          COMPLETE → all green
          ERROR → red on last active step
    """
    _action_to_step = {
        "S1_open_box": 1, "S2_put_earphone": 2, "S3_put_charger": 3,
        "S4_put_green_bag": 4, "S5_close_box": 5,
    }
    h, w = frame.shape[:2]
    step_names = ["S1:Open", "S2:Earphn", "S3:Charger", "S4:Bag", "S5:Close"]
    bar_y = 8
    bar_h = 22
    total_steps = 5
    margin_left = w // 2 - (total_steps * 100 + (total_steps - 1) * 8) // 2
    step_w = 100
    gap = 8

    is_running = mode in ("RUNNING", "COMPLETE", "ERROR")
    is_preview = mode in ("PREVIEW", "ARMED")

    if is_preview:
        # ── PREVIEW / ARMED: all neutral, NO predictions on bar ──
        colors = [(45, 45, 45)] * total_steps

    elif is_running:
        display_step = current_step_id
        display_path = fsm_path
        colors = []
        for i in range(total_steps):
            sid = i + 1
            if alarm:
                c = (0, 0, 255) if sid == display_step else (
                    (0, 180, 0) if sid in display_path else (80, 80, 80))
            elif is_complete:
                c = (0, 220, 0)
            else:
                next_step = display_step + 1
                next_active = (
                    action_phase in ("ACTIVE", "COMPLETING", "WRONG_ACTIVE")
                    and _action_to_step.get(current_action, -1) == next_step
                )
                if sid <= display_step:
                    c = (0, 180, 0)          # completed — green
                elif sid == next_step and next_active:
                    c = (0, 240, 255)         # actively in progress — cyan
                elif sid == next_step:
                    c = (60, 60, 90)          # waiting — dark blue-gray
                else:
                    c = (80, 80, 80)          # future — gray
            colors.append(c)

    for i in range(total_steps):
        sid = i + 1
        x1 = margin_left + i * (step_w + gap)
        x2 = x1 + step_w
        y1 = bar_y
        y2 = bar_y + bar_h

        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        text = step_names[i]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        tx = x1 + (step_w - tw) // 2
        ty = y1 + (bar_h + th) // 2
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_status(frame, det_summary, hands, box_bbox, box_state, info,
                local_fps=0.0):
    """Overlay all info directly on frame — matches original client style."""
    h, w = frame.shape[:2]
    px, py = 10, 30
    lh = 24
    mode = info.get("mode", "PREVIEW")
    is_running = mode in ("RUNNING", "COMPLETE", "ERROR")
    is_armed = mode == "ARMED"

    # ── Detection boxes (objects only — box is drawn separately below) ──
    for det in det_summary:
        cls_name = det.get("class", "")
        conf = det.get("conf", 0)
        bbox = det.get("bbox")
        if bbox is None:
            continue
        # Box is drawn by the highlight section below — skip here to avoid double-draw
        if cls_name in ("box_open", "box_closed"):
            continue
        if cls_name in ("earphone", "charger", "green_bag") and conf < 0.25:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        color = COLORS.get(cls_name, (0, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}",
                    (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ── Box state indicator (text only, no bbox — box is stationary and known) ──
    if box_bbox and box_state:
        bx1, by1, bx2, by2 = map(int, box_bbox)
        state_colors = {
            "open": (0, 255, 0), "closed": (0, 165, 255),
            "transition": (0, 200, 255), "unknown": (100, 100, 100),
        }
        box_color = state_colors.get(box_state, (200, 200, 200))
        # Small text indicator near top-left, no rectangle
        cv2.putText(frame, f"BOX:{box_state}", (frame.shape[1] - 180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    # ── CLOSED box safety net: hide interior objects ──
    if box_state == "closed" and box_bbox:
        interior_objects = {"earphone", "charger", "green_bag"}
        bx1, by1, bx2, by2 = box_bbox
        margin = 20
        for det in det_summary:
            cls = det.get("class", "")
            if cls not in interior_objects:
                continue
            dbbox = det.get("bbox", [])
            if len(dbbox) != 4:
                continue
            cx = (dbbox[0] + dbbox[2]) / 2
            cy = (dbbox[1] + dbbox[3]) / 2
            if bx1 - margin <= cx <= bx2 + margin and by1 - margin <= cy <= by2 + margin:
                ex1, ey1, ex2, ey2 = map(int, dbbox)
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 0), -1)

    # ── Hand keypoints ──
    for hand in hands:
        if hand.bbox is not None:
            hx1, hy1, hx2, hy2 = hand.bbox
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 150, 0), 1)
        for lm in hand.landmarks:
            lx, ly = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(frame, (lx, ly), 2, (0, 200, 100), -1)

    # ── FPS ──
    cv2.putText(frame, f"FPS:{local_fps:.1f}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ── MODE line ──
    mode_colors = {
        "PREVIEW": (100, 200, 100), "ARMED": (255, 200, 0),
        "RUNNING": (0, 255, 255), "COMPLETE": (0, 255, 0),
        "ERROR": (0, 0, 255),
    }
    mode_color = mode_colors.get(mode, (200, 200, 200))
    cv2.putText(frame, f"MODE: {mode}",
                (px, py + lh), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)

    # ── Info line ──
    top3 = info.get("latest_top3", [])
    top3_str = " ".join(f"S{s}={c:.2f}" for s, c in top3[:3]) if top3 else "-"
    cv2.putText(frame, f"Dets:{len(det_summary)} Hands:{len(hands)} Top3:{top3_str}",
                (px, py + lh * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # ── RUNNING / ARMED: FSM-driven display ──
    if is_running or is_armed:
        y_off = py + lh * 4
        step_name = info.get("step_name", "-")
        step_id = info.get("step_id", 0)
        expected = info.get("expected_event", "")
        accepted = info.get("accepted", [])
        alarm = info.get("alarm")
        fsm_path = info.get("fsm_path", [])
        is_complete = info.get("is_complete", False)
        step_probs = info.get("latest_step_probs", [])

        if expected:
            cv2.putText(frame, f"Expect: {expected}",
                        (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 200), 2)
            y_off += lh

        if is_running:
            cv2.putText(frame, f"Step: {step_name}",
                        (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)
            y_off += lh

        if accepted:
            ae_text = "OK: " + " -> ".join(accepted)
            cv2.putText(frame, ae_text[:80],
                        (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 200, 100), 1)
            y_off += lh

        if alarm:
            cv2.putText(frame, f"ALARM: {alarm['type']} — {alarm['message']}",
                        (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            y_off += lh

        # S1-S5 progress bar
        model_pred = top3[0][0] if top3 else -1
        action_phase = info.get("action_phase", "WAITING")
        current_action = info.get("current_action", "none")
        _draw_sop_progress(frame, step_id, fsm_path, is_complete, alarm,
                          mode, model_pred, step_probs,
                          action_phase=action_phase,
                          current_action=current_action)

    elif mode == "PREVIEW":
        model_pred = top3[0][0] if top3 else -1
        if model_pred > 0:
            cv2.putText(frame, f"Warmup - guess: S{model_pred}  (SPACE to start)",
                        (px, py + lh * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (100, 100, 80), 1)
        _draw_sop_progress(frame, 0, [], False, None, mode, model_pred,
                          info.get("latest_step_probs", []))

    # ── Mode-specific center message ──
    if mode == "PREVIEW":
        cv2.putText(frame, "MODE: PREVIEW", (w // 2 - 100, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)
        cv2.putText(frame, "SPACE=Start  R=Reset",
                    (w // 2 - 130, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    elif mode == "ARMED":
        cv2.putText(frame, "READY — Open the box to start",
                    (w // 2 - 180, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(frame, "Waiting for box_opened...",
                    (w // 2 - 140, h // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    elif mode == "RUNNING":
        cv2.putText(frame, "SOP RUNNING...", (w // 2 - 100, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif mode == "COMPLETE":
        cv2.putText(frame, "SOP COMPLETE!", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(frame, "SPACE=Restart  R=Reset",
                    (w // 2 - 130, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif mode == "ERROR":
        cv2.putText(frame, "ERROR DETECTED", (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    return frame


# ═══════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════

def run_loop(source, pipeline: SOPLocalPipeline, no_display=False):
    """Main detection loop: camera → pipeline → display (client-style)."""
    print("\n" + "=" * 60)
    print("控制: [SPACE] 开始/停止  [R] 重置  [Q] 退出")
    print("动作序列: 开盒 → 放耳机 → 放插头 → 放绿袋 → 关盒")
    print("=" * 60 + "\n")

    if not no_display:
        cv2.namedWindow("SOP Local Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SOP Local Detection", 1280, 720)

    frame_count = 0

    try:
        while True:
            frame, ts = source.read()
            if frame is None:
                if isinstance(source, VideoSource):
                    print("\n[Video] 播放完毕，按任何键退出...")
                time.sleep(0.01)
                if not no_display:
                    key = cv2.waitKey(5) & 0xFF
                    if key in (ord('q'), 27):
                        break
                continue

            frame_count += 1

            # Process
            det_summary, info, box_bbox, box_state, hands = pipeline.process_frame(frame, ts)

            # Draw
            if not no_display:
                display = draw_status(frame.copy(), det_summary, hands,
                                      box_bbox, box_state, info,
                                      local_fps=info.get('fps', 0.0))
                cv2.imshow("SOP Local Detection", display)

                key = cv2.waitKey(5) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord(' '):  # SPACE = start / stop
                    if pipeline.mode == "RUNNING":
                        pipeline.reset()
                        print("[Key] SPACE → RESET (from RUNNING)", flush=True)
                    elif pipeline.mode in ("PREVIEW", "ERROR", "COMPLETE"):
                        pipeline.start()
                        print("[Key] SPACE → START (ARMED)", flush=True)
                elif key in (ord('r'), ord('R')):  # reset
                    pipeline.reset()
                    print("[Key] R → RESET", flush=True)
            else:
                if frame_count % 30 == 0:
                    print(f"  [{frame_count:5d}] {info['step_name']:15s} "
                          f"box={info['box_state']:6s} fps={info['fps']:.1f}  "
                          f"expected={info['expected_event']}  "
                          f"accepted={info['accepted']}",
                          flush=True)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        source.close()
        if not no_display:
            cv2.destroyAllWindows()

    print(f"\n总帧数: {frame_count}")
    return pipeline


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SOP 实时动作检测 — 本地运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python local_run.py --camera hik         # 海康工业相机
  python local_run.py --camera webcam      # USB 摄像头
  python local_run.py --video demo.mp4     # 视频文件 (循环)
  python local_run.py --video demo.mp4 --no-display --no-temporal   # 无界面批量跑
        """)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", choices=["hik", "webcam"], default="webcam",
                     help="相机源 (默认: webcam)")
    src.add_argument("--video", help="视频文件路径")
    parser.add_argument("--webcam-id", type=int, default=0, help="USB 摄像头 ID")
    parser.add_argument("--device", default="auto",
                        help="推理设备 (auto/cuda/cpu, 默认: auto)")
    parser.add_argument("--yolo", help="YOLO 模型路径")
    parser.add_argument("--model", help="时序模型路径")
    parser.add_argument("--conf", type=float, default=0.12, help="YOLO 置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument("--no-temporal", action="store_true", help="禁用时序模型")
    parser.add_argument("--no-display", action="store_true", help="不显示 GUI 窗口")

    args = parser.parse_args()

    print("=" * 60)
    print("SOP 实时动作检测 — 本地运行器")
    print(f"  设备: {args.device} (auto → {'cuda' if torch.cuda.is_available() else 'cpu'})")
    print(f"  相机: {args.camera if not args.video else 'video'}")
    if args.video:
        print(f"  视频: {args.video}")
    print("=" * 60)

    # Create source
    try:
        source = create_source(args)
    except Exception as e:
        print(f"✗ 无法打开输入源: {e}")
        return 1

    # Create pipeline
    try:
        pipeline = SOPLocalPipeline(args)
    except Exception as e:
        print(f"✗ 管线初始化失败: {e}")
        source.close()
        return 1

    # Run
    run_loop(source, pipeline, no_display=args.no_display)
    return 0


if __name__ == "__main__":
    sys.exit(main())
