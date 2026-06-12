"""
相机实时 SOP 检测烟雾测试 — 全链路集成

真实单遍式流式推理:
  海康相机 → YOLO → MediaPipe → FeatureExtractorV2Standalone
  → TemporalPredictorV2.predict() → get_latest_mature_prediction()
  → SOPStateMachine → 实时显示 + 保存 trace

Usage:
  C:/Users/HP/AppData/Local/Programs/Python/Python310/python.exe tools/camera_smoke_test_v2.py

前提:
  1. MVS 软件已关闭
  2. 相机 USB3 线已连接
  3. 模型文件在 models/ 目录下

Interactive:
  s — 开始新测试
  r — 重置/取消当前测试
  q — 退出
"""
import sys
import os
import json
import time
import ctypes
import argparse
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from ctypes import c_uint, c_void_p, c_bool, POINTER, byref, cast, memset, sizeof
from typing import Optional, Dict, List
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── MVS SDK setup ──
_MVS_ROOT = r"D:\develop\MVS"
_MVS_SDK_PYTHON = os.path.join(_MVS_ROOT, "Development", "Samples", "Python")
_MVS_MV_IMPORT = os.path.join(_MVS_SDK_PYTHON, "MvImport")

_MVS_LIB_PATHS = [
    os.path.join(_MVS_ROOT, "Development", "Libraries", "win64"),
    os.path.join(_MVS_ROOT, "Runtime", "win64"),
]
for p in _MVS_LIB_PATHS:
    if os.path.isdir(p):
        try:
            os.add_dll_directory(p)
        except Exception:
            pass

sys.path.insert(0, _MVS_SDK_PYTHON)
sys.path.insert(0, _MVS_MV_IMPORT)

from MvCameraControl_class import (MvCamera, MV_CC_DEVICE_INFO_LIST,
                                    MVCC_INTVALUE, MVCC_FLOATVALUE,
                                    MVCC_ENUMVALUE, MV_FRAME_OUT)
from CameraParams_header import (MV_CC_DEVICE_INFO, MV_USB_DEVICE,
                                  MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE)
from MvErrorDefine_const import MV_OK
import PixelType_header as px

from engine.yolo_detector import YOLODetector
from engine.hand_detector import HandDetector
from engine.sop_fsm import SOPStateMachine, SOPStep, STEP_NAMES
from engine.temporal_predictor_v2 import TemporalPredictorV2
from tools.extract_features_v2_90 import FeatureExtractorV2Standalone, INTERACT_OBJS

# ── Config ──
REPORTS_DIR = ROOT / "reports" / "camera_smoke_v2"
FPS_TARGET = 15.0
T = 48
STRIDE = 4
N_CLASSES = 6
POST_S5_DELAY = 1.5   # seconds to wait after S5 before auto-COMPLETE

# ── Display colors ──
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)

_PRINTED_FORMAT = None


def decoding_char(c_ubyte_value) -> str:
    c_char_p_value = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
    try:
        return c_char_p_value.value.decode('gbk')
    except (UnicodeDecodeError, AttributeError):
        try:
            return c_char_p_value.value.decode('utf-8')
        except Exception:
            return str(c_char_p_value.value) if c_char_p_value.value else ""


def to_hex_str(num: int) -> str:
    cha_dic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    if num < 0:
        num = num + 2 ** 32
    hex_str = ""
    while num >= 16:
        digit = num % 16
        hex_str = cha_dic.get(digit, str(digit)) + hex_str
        num //= 16
    hex_str = cha_dic.get(num, str(num)) + hex_str
    return "0x" + hex_str


def convert_frame_to_bgr(st_frame) -> np.ndarray:
    global _PRINTED_FORMAT
    fi = st_frame.stFrameInfo
    w, h = fi.nWidth, fi.nHeight
    fmt = fi.enPixelType
    buf_addr = st_frame.pBufAddr
    payload = fi.nFrameLen

    if _PRINTED_FORMAT != fmt:
        pf_name = {px.PixelType_Gvsp_BayerRG8: "BayerRG8",
                   px.PixelType_Gvsp_BayerGB8: "BayerGB8",
                   px.PixelType_Gvsp_BayerGR8: "BayerGR8",
                   px.PixelType_Gvsp_BayerBG8: "BayerBG8",
                   px.PixelType_Gvsp_Mono8: "Mono8",
                   px.PixelType_Gvsp_RGB8_Packed: "RGB8",
                   px.PixelType_Gvsp_BGR8_Packed: "BGR8"}.get(fmt, "?")
        print(f"[Camera] {w}x{h} {pf_name} (fmt={to_hex_str(fmt)})")
        _PRINTED_FORMAT = fmt

    if fmt == px.PixelType_Gvsp_Mono8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_GRAY2BGR)

    if fmt == px.PixelType_Gvsp_RGB8_Packed:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h * 3))).contents)
        return cv2.cvtColor(raw.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)

    if fmt == px.PixelType_Gvsp_BGR8_Packed:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h * 3))).contents)
        return raw.reshape((h, w, 3))

    if fmt == px.PixelType_Gvsp_BayerRG8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerRG2BGR)

    if fmt == px.PixelType_Gvsp_BayerGB8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerGB2BGR)

    if fmt == px.PixelType_Gvsp_BayerGR8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerGR2BGR)

    if fmt == px.PixelType_Gvsp_BayerBG8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerBG2BGR)

    # Fallback
    try:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * payload)).contents)
        return raw.reshape((h, w, 3))
    except Exception:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf_addr, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_GRAY2BGR)


def open_hik_camera():
    """Open MV-CS050-10UC, return (cam, w, h) or (None, 0, 0) on failure."""
    print("[Camera] 初始化 SDK...")
    ret = MvCamera.MV_CC_Initialize()
    if ret != MV_OK:
        print(f"  MV_CC_Initialize FAILED: {to_hex_str(ret)}")
        return None, 0, 0

    print("[Camera] 枚举设备...")
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if ret != MV_OK or device_list.nDeviceNum == 0:
        print(f"  无设备: ret={to_hex_str(ret)} n={device_list.nDeviceNum}")
        return None, 0, 0

    print(f"  找到 {device_list.nDeviceNum} 个设备")
    target_serial = "DA9562204"
    selected_idx = -1

    for i in range(device_list.nDeviceNum):
        mvcc_dev_info = cast(device_list.pDeviceInfo[i],
                             POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            info = mvcc_dev_info.SpecialInfo.stUsb3VInfo
            model = decoding_char(info.chModelName)
            serial = ""
            for ch in info.chSerialNumber:
                if ch == 0:
                    break
                serial += chr(ch)
            print(f"  [{i}] USB: {model} SN:{serial}")
            if serial == target_serial:
                selected_idx = i
            if selected_idx < 0:
                selected_idx = i

    if selected_idx < 0:
        print("  未找到 USB 相机")
        return None, 0, 0

    print(f"  选择 [{selected_idx}]")
    st_device = cast(device_list.pDeviceInfo[selected_idx],
                     POINTER(MV_CC_DEVICE_INFO)).contents

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(st_device)
    if ret != MV_OK:
        print(f"  CreateHandle FAILED: {to_hex_str(ret)}")
        return None, 0, 0

    ret = cam.MV_CC_OpenDevice()
    if ret != MV_OK:
        print(f"  OpenDevice FAILED: {to_hex_str(ret)} (MVS 软件关闭了吗?)")
        cam.MV_CC_DestroyHandle()
        return None, 0, 0

    # TriggerMode = Off
    cam.MV_CC_SetEnumValue("TriggerMode", 0)

    # Read current exposure, keep it if reasonable
    st_exp = MVCC_FLOATVALUE()
    cam.MV_CC_GetFloatValue("ExposureTime", st_exp)
    if st_exp.fCurValue < 15000:
        cam.MV_CC_SetFloatValue("ExposureTime", 35000.0)
        print(f"  曝光: {st_exp.fCurValue:.0f} → 35000 us")
    else:
        print(f"  曝光: {st_exp.fCurValue:.0f} us (保持)")

    # Get resolution
    st_w = MVCC_INTVALUE()
    st_h = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("Width", st_w)
    cam.MV_CC_GetIntValue("Height", st_h)
    w, h = st_w.nCurValue, st_h.nCurValue
    print(f"  分辨率: {w}x{h}")

    # Start grabbing
    ret = cam.MV_CC_StartGrabbing()
    if ret != MV_OK:
        print(f"  StartGrabbing FAILED: {to_hex_str(ret)}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        return None, 0, 0

    print("  ✓ 取流已开始")
    return cam, w, h


def close_hik_camera(cam, grabbing: bool):
    """Clean up camera resources."""
    if cam is None:
        return
    if grabbing:
        try:
            cam.MV_CC_StopGrabbing()
        except Exception:
            pass
    try:
        cam.MV_CC_CloseDevice()
    except Exception:
        pass
    try:
        cam.MV_CC_DestroyHandle()
    except Exception:
        pass
    try:
        MvCamera.MV_CC_Finalize()
    except Exception:
        pass


class SmokeTestSession:
    """Manages one smoke test session: camera frames → pipeline → FSM → traces."""

    def __init__(self, yolo: YOLODetector, hand_detector: HandDetector,
                 extractor: FeatureExtractorV2Standalone,
                 predictor: TemporalPredictorV2):
        self.yolo = yolo
        self.hand_detector = hand_detector
        self.extractor = extractor
        self.predictor = predictor
        self.fsm = SOPStateMachine(timeout=90.0, min_step_duration=0.3)

        self.active = False
        self.test_name = ""
        self.start_time = 0.0
        self.frame_count = 0
        self.feat_count = 0
        self._last_feat_time = -1.0
        self._feat_interval = 1.0 / FPS_TARGET

        # Traces
        self.event_trace = []
        self.fsm_events = []
        self.temporal_trace = []
        self._prev_fsm_step = -1
        self._none_count = 0

        # Post-S5 completion tracking
        self._s5_reached_time = None   # timestamp when FSM first reached S5_CLOSE
        self._post_s5_delay = POST_S5_DELAY  # seconds to wait after S5 before auto-COMPLETE

        # Display
        self.disp_step = "IDLE"
        self.disp_pred = "-"
        self.disp_conf = 0.0
        self.disp_fps = 0.0
        self.disp_mature = 0
        self.disp_none = 0
        self._fps_times = []

    def reset(self, test_name: str):
        self.predictor.reset()
        self.yolo.reset_tracking()
        self.extractor.reset()
        self.hand_detector.palm_velocity.clear()
        self.fsm = SOPStateMachine(timeout=90.0, min_step_duration=0.3)

        self.active = True
        self.test_name = test_name
        self.start_time = time.time()
        self.frame_count = 0
        self.feat_count = 0
        self._last_feat_time = -1.0
        self.event_trace = []
        self.fsm_events = []
        self.temporal_trace = []
        self._prev_fsm_step = -1
        self._none_count = 0
        self._s5_reached_time = None
        self._fps_times = []
        self.disp_step = "IDLE"
        self.disp_pred = "-"
        self.disp_conf = 0.0
        self.disp_mature = 0
        self.disp_none = 0

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """Process one frame. Returns annotated frame for display."""
        if not self.active:
            return frame

        self.frame_count += 1
        h, w = frame.shape[:2]

        # FPS
        self._fps_times.append(time.time())
        if len(self._fps_times) > 30:
            self._fps_times = self._fps_times[-30:]
        if len(self._fps_times) >= 2:
            dt = self._fps_times[-1] - self._fps_times[0]
            self.disp_fps = (len(self._fps_times) - 1) / dt if dt > 0 else 0

        # ── YOLO + MediaPipe ──
        try:
            detections = self.yolo.detect(frame)
            hands = self.hand_detector.detect(frame)
        except Exception:
            return frame

        box_bbox = self.yolo.get_box_bbox(detections)
        box_state_str, _ = self.yolo.get_box_state(detections)

        # Hand selection
        if box_bbox:
            self.extractor.compute_interaction(detections, hands, box_bbox, h, w)
        self.hand_detector.select_active_hand(hands, target_object_bbox=None,
                                              box_bbox=box_bbox)

        # Update tracked objects
        if box_bbox:
            for det in detections:
                if det.track_id >= 0 and det.cls_name in INTERACT_OBJS:
                    cx, cy = det.center
                    bx1, by1, bx2, by2 = box_bbox
                    in_box = (bx1 <= cx <= bx2 and by1 <= cy <= by2)
                    in_box_ratio = self.yolo.compute_in_box_ratio(det.bbox, box_bbox)
                    self.yolo.update_in_box_status(det.track_id,
                                                   in_box or in_box_ratio > 0.3)

        # ── Feature extraction at target FPS ──
        if timestamp - self._last_feat_time >= self._feat_interval or self._last_feat_time < 0:
            self._last_feat_time = timestamp
            self.feat_count += 1

            feat = self.extractor.extract(frame, detections, hands, box_bbox,
                                          box_state_str)
            feat_idx = self.feat_count - 1

            # ── TemporalPredictorV2 ──
            self.predictor.predict(feat)
            mature = self.predictor.get_latest_mature_prediction()

            trace_entry = {
                "feature_idx": feat_idx,
                "frame": self.frame_count,
                "timestamp": round(timestamp, 3),
                "mature_frame_idx": self.predictor.latest_mature_frame,
                "prediction_available": mature is not None,
                "physical_state_enabled": False,
            }

            if mature is None:
                self._none_count += 1
                self.disp_none = self._none_count
                trace_entry["fsm_step"] = self.fsm.current_step.value
                trace_entry["fsm_step_name"] = STEP_NAMES[self.fsm.current_step]
                trace_entry["fsm_updated"] = False
                self.event_trace.append(trace_entry)
            else:
                pred_step = mature["step"]
                pred_conf = mature["confidence"]
                top3 = mature["top3"]

                self.disp_pred = str(pred_step)
                self.disp_conf = pred_conf
                self.disp_mature = self.predictor.latest_mature_frame

                self.temporal_trace.append({
                    "feature_idx": feat_idx,
                    "mature_frame_idx": self.predictor.latest_mature_frame,
                    "frame": self.frame_count,
                    "timestamp": round(timestamp, 3),
                    "step": pred_step,
                    "confidence": round(pred_conf, 4),
                    "step_name": STEP_NAMES.get(pred_step, "?"),
                    "top3": [(int(s), round(float(c), 4)) for s, c in top3],
                })

                # ── FSM logic ──
                # S0 = idle/no action. Don't feed to FSM (avoids spurious
                # ERROR on S5→S0). Post-S5 completion is handled by timer below.
                if pred_step != 0:
                    fs_result = self.fsm.validate(pred_step, pred_conf,
                                                  physical_state_ok=True,
                                                  timestamp=timestamp)
                    if fs_result.step_id != self._prev_fsm_step:
                        self.fsm_events.append({
                            "feature_idx": feat_idx, "frame": self.frame_count,
                            "timestamp": round(timestamp, 3),
                            "step": fs_result.step_id,
                            "step_name": STEP_NAMES.get(fs_result.step_id, "?"),
                            "model_pred": pred_step,
                            "model_confidence": round(pred_conf, 4),
                            "is_correct": fs_result.is_correct,
                            "has_error": fs_result.has_error,
                            "error_type": fs_result.error_type,
                            "message": fs_result.message,
                            "top3": [(int(s), round(float(c), 4)) for s, c in top3],
                            "physical_state_enabled": False,
                        })
                        self._prev_fsm_step = fs_result.step_id
                        # Track when S5 was first reached
                        if fs_result.step_id == 5:  # S5_CLOSE
                            self._s5_reached_time = timestamp

                # ── Post-S5 auto-completion ──
                # After S5_CLOSE has been stable for _post_s5_delay, complete.
                # This replaces the old "idle after S5" hack — S0 is no longer
                # required to trigger COMPLETE; only time-in-S5 matters.
                if (self.fsm.current_step == SOPStep.S5_CLOSE
                        and self._s5_reached_time is not None):
                    if timestamp - self._s5_reached_time >= self._post_s5_delay:
                        self.fsm.current_step = SOPStep.COMPLETE
                        if self._prev_fsm_step != 6:
                            self.fsm_events.append({
                                "feature_idx": feat_idx, "frame": self.frame_count,
                                "timestamp": round(timestamp, 3),
                                "step": 6, "step_name": "COMPLETE",
                                "model_pred": pred_step,
                                "is_correct": True,
                                "has_error": False, "error_type": "",
                                "message": (f"Sequence complete "
                                            f"(S5 stable {self._post_s5_delay:.1f}s)"),
                                "top3": [(int(s), round(float(c), 4))
                                         for s, c in top3],
                                "physical_state_enabled": False,
                            })
                            self._prev_fsm_step = 6

                self.disp_step = STEP_NAMES.get(self.fsm.current_step, "?")

                trace_entry["fsm_step"] = self.fsm.current_step.value
                trace_entry["fsm_step_name"] = STEP_NAMES[self.fsm.current_step]
                trace_entry["fsm_updated"] = True
                trace_entry["model_pred"] = pred_step
                trace_entry["model_conf"] = round(pred_conf, 4)
                self.event_trace.append(trace_entry)

        # ── Draw overlay ──
        return self._draw_overlay(frame, detections, hands, box_bbox, box_state_str)

    def _draw_overlay(self, frame, detections, hands, box_bbox, box_state):
        h, w = frame.shape[:2]

        # Detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = GREEN if det.cls_name in ("box_open", "box_closed") else YELLOW
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.cls_name} {det.confidence:.2f}"
            if det.track_id >= 0:
                label += f" id:{det.track_id}"
            cv2.putText(frame, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Box bbox
        if box_bbox:
            bx1, by1, bx2, by2 = box_bbox
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), CYAN, 1)
            cv2.putText(frame, f"box:{box_state}", (bx1, by1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)

        # Hands
        for hand in hands:
            if hand.bbox is not None:
                hx1, hy1, hx2, hy2 = hand.bbox
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 150, 0), 1)

        # ── Status panel ──
        px, py = 10, 28
        lh = 22

        # FPS
        cv2.putText(frame, f"FPS:{self.disp_fps:.1f}", (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Step
        step_color = GREEN if self.fsm.current_step == SOPStep.COMPLETE else YELLOW
        cv2.putText(frame, f"STEP:{self.disp_step}", (px, py + lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, step_color, 2)

        # Prediction
        cv2.putText(frame, f"Pred:{self.disp_pred} conf:{self.disp_conf:.3f}",
                    (px, py + lh * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        # Mature / None count
        cv2.putText(frame, f"Mature:{self.disp_mature} None:{self.disp_none}",
                    (px, py + lh * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

        # Physical state indicator
        cv2.putText(frame, "PhysState: OFF", (px, py + lh * 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)

        # Test name
        test_str = f"Test:{self.test_name}" if self.active else "Test: IDLE"
        cv2.putText(frame, test_str, (px, py + lh * 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)

        # ── Bottom bar ──
        bar_y = h - 40
        cv2.rectangle(frame, (0, bar_y), (w, h), (0, 0, 0), -1)
        if self.active:
            elapsed = time.time() - self.start_time
            info = (f"[s]Start [r]Reset [q]Quit | "
                    f"Frames:{self.frame_count} Feats:{self.feat_count} "
                    f"Elapsed:{elapsed:.1f}s")
        else:
            info = "[s] Start new test  [q] Quit"
        cv2.putText(frame, info, (10, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        # Error indicator
        if any(e.get("has_error") for e in self.fsm_events):
            cv2.putText(frame, "ERROR", (w - 110, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        return frame

    def finish(self) -> Dict:
        """Finalize test and return result dict."""
        self.active = False
        elapsed = time.time() - self.start_time

        # If test ended while at S5 (user pressed 'r' or 'q'), complete if timer elapsed
        if self.fsm.current_step == SOPStep.S5_CLOSE:
            post_s5_elapsed = elapsed - (self._s5_reached_time or elapsed)
            if post_s5_elapsed >= self._post_s5_delay:
                self.fsm.current_step = SOPStep.COMPLETE
                if self._prev_fsm_step != 6:
                    self.fsm_events.append({
                        "feature_idx": self.feat_count - 1,
                        "frame": self.frame_count,
                        "timestamp": round(elapsed, 3),
                        "step": 6, "step_name": "COMPLETE",
                        "model_pred": -1, "is_correct": True,
                        "has_error": False, "error_type": "",
                        "message": (f"Sequence complete (S5 stable "
                                    f"{post_s5_elapsed:.1f}s, end of test)"),
                        "physical_state_enabled": False,
                    })

        fsm_complete = self.fsm.current_step == SOPStep.COMPLETE
        fsm_path = [e["step"] for e in self.fsm_events if e["step"] > 0]
        fsm_errors = [e for e in self.fsm_events if e.get("has_error")]

        return {
            "test_name": self.test_name,
            "timestamp": datetime.now().isoformat(),
            "duration_sec": round(elapsed, 1),
            "total_frames": self.frame_count,
            "total_features": self.feat_count,
            "n_mature_predictions": len(self.temporal_trace),
            "none_prediction_count": self._none_count,
            "physical_state_enabled": False,
            "none_consumed_by_fsm": 0,
            "fsm_path": fsm_path,
            "final_step": self.fsm.current_step.value,
            "final_step_name": STEP_NAMES.get(self.fsm.current_step, "?"),
            "fsm_complete": fsm_complete,
            "fsm_errors": [{"type": e["error_type"], "msg": e["message"]}
                           for e in fsm_errors],
            "first_error": fsm_errors[0] if fsm_errors else None,
            "first_mature_frame": self.predictor.first_mature_frame,
            "event_trace": self.event_trace,
            "fsm_events": self.fsm_events,
            "temporal_trace": self.temporal_trace,
        }


def save_test_result(result: Dict, output_dir: Path):
    test_name = result["test_name"]
    test_dir = output_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # event_trace.json
    with open(test_dir / "event_trace.json", "w", encoding="utf-8") as f:
        json.dump(result.pop("event_trace"), f, indent=2, ensure_ascii=False)

    # fsm_trace.json
    fsm_out = {
        "test_name": test_name,
        "fsm_path": result["fsm_path"],
        "fsm_complete": result["fsm_complete"],
        "final_step": result["final_step_name"],
        "events": result.pop("fsm_events"),
        "errors": result["fsm_errors"],
    }
    with open(test_dir / "fsm_trace.json", "w", encoding="utf-8") as f:
        json.dump(fsm_out, f, indent=2, ensure_ascii=False)

    # temporal_trace.json
    with open(test_dir / "temporal_trace.json", "w", encoding="utf-8") as f:
        json.dump(result.pop("temporal_trace"), f, indent=2, ensure_ascii=False)

    # result.json
    with open(test_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  → Saved to {test_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=480)  # smaller for CPU
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    yolo_path = args.yolo or str(ROOT / "models" / "yolo_final_v1.pt")
    model_path = args.model or str(ROOT / "models" / "temporal" /
                                   "v2_90_tcn_bigru" / "checkpoints" / "best.pt")
    out_dir = Path(args.output_dir) if args.output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("相机实时 SOP 检测烟雾测试 — v2_90 全链路")
    print(f"  YOLO: {yolo_path}")
    print(f"  Temporal: {model_path}")
    print(f"  Output: {out_dir}")
    print(f"  imgsz: {args.imgsz} (CPU mode)")
    print(f"  physical_state_enabled: False")
    print(f"  S5 completion: timer-based ({POST_S5_DELAY}s post-S5)")
    print("=" * 60)

    # ── Init pipeline components ──
    print("\n[Init] YOLO + MediaPipe...")
    yolo = YOLODetector(yolo_path, conf_thresh=args.conf, device="cpu",
                        imgsz=args.imgsz)
    hand_detector = HandDetector()
    extractor = FeatureExtractorV2Standalone(yolo, hand_detector,
                                             ema_alpha=0.5, hold_frames=5)

    print("[Init] TemporalPredictorV2...")
    predictor = TemporalPredictorV2(model_path, device="cpu", T=T, stride=STRIDE)
    print(f"  input_dim={predictor.input_dim}, first mature ~{predictor.half_w / 15:.1f}s")

    session = SmokeTestSession(yolo, hand_detector, extractor, predictor)

    # ── Open camera ──
    print("\n⚠  请确认 MVS 软件已关闭")
    cam, cam_w, cam_h = open_hik_camera()
    if cam is None:
        print("ERROR: 无法打开相机")
        return 1

    grabbing = True
    all_results = []
    test_counter = [0]
    start_time = time.time()

    print("\n" + "=" * 60)
    print("控制: [s] 开始新测试  [r] 重置当前测试  [q] 退出")
    print("烟雾测试计划: 3 OK + 2 乱序 + 2 漏步 + 1 重复 = 8 次")
    print("=" * 60)

    cv2.namedWindow("SOP Realtime Detection - v2_90", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SOP Realtime Detection - v2_90", 960, 720)

    try:
        while True:
            # ── Get camera frame ──
            st_frame = MV_FRAME_OUT()
            memset(byref(st_frame), 0, sizeof(MV_FRAME_OUT))

            ret = cam.MV_CC_GetImageBuffer(st_frame, 1000)
            if ret != MV_OK or st_frame.pBufAddr is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                if key == ord('s'):
                    if session.active:
                        print(f"\n[Finishing] '{session.test_name}'...")
                        result = session.finish()
                        save_test_result(result, out_dir)
                        all_results.append(result)
                    test_counter[0] += 1
                    test_name = f"test_{test_counter[0]:03d}"
                    print(f"\n[Starting] '{test_name}' — 请执行 SOP 动作...")
                    session.reset(test_name)
                    start_time = time.time()
                if key == ord('r') and session.active:
                    print(f"\n[Reset] '{session.test_name}' cancelled")
                    result = session.finish()
                    save_test_result(result, out_dir)
                    all_results.append(result)
                continue

            try:
                frame = convert_frame_to_bgr(st_frame)
                timestamp = time.time() - start_time

                # ── Process through pipeline ──
                annotated = session.process_frame(frame, timestamp)
                if annotated is None:
                    annotated = frame

                cv2.imshow("SOP Realtime Detection - v2_90", annotated)

            finally:
                cam.MV_CC_FreeImageBuffer(st_frame)

            # ── Auto-complete check ──
            if session.active and session.fsm.current_step == SOPStep.COMPLETE:
                print(f"\n[Auto] '{session.test_name}' COMPLETE!")
                result = session.finish()
                save_test_result(result, out_dir)
                all_results.append(result)
                print(" 按 [s] 开始下一次测试")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                if session.active:
                    print(f"\n[Finishing] '{session.test_name}'...")
                    result = session.finish()
                    save_test_result(result, out_dir)
                    all_results.append(result)
                test_counter[0] += 1
                test_name = f"test_{test_counter[0]:03d}"
                print(f"\n[Starting] '{test_name}' — 请执行 SOP 动作...")
                session.reset(test_name)
                start_time = time.time()
            elif key == ord('r') and session.active:
                print(f"\n[Reset] '{session.test_name}' cancelled")
                result = session.finish()
                save_test_result(result, out_dir)
                all_results.append(result)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

        if session.active:
            result = session.finish()
            save_test_result(result, out_dir)
            all_results.append(result)

        close_hik_camera(cam, grabbing)

        # ── Save summary ──
        if all_results:
            ok_tests = [r for r in all_results if r["fsm_complete"]
                        and len(r["fsm_errors"]) == 0]
            err_tests = [r for r in all_results if not r["fsm_complete"]
                         or len(r["fsm_errors"]) > 0]

            summary = {
                "test_date": datetime.now().strftime("%Y-%m-%d"),
                "test_time": datetime.now().isoformat(),
                "method": "相机实时 SOP 检测烟雾测试 — v2_90 全链路",
                "physical_state_enabled": False,
                "n_tests": len(all_results),
                "n_complete_no_error": len(ok_tests),
                "n_error": len(err_tests),
                "per_test": [],
            }
            for r in all_results:
                summary["per_test"].append({
                    "test_name": r["test_name"],
                    "duration_sec": r["duration_sec"],
                    "fsm_path": r["fsm_path"],
                    "fsm_complete": r["fsm_complete"],
                    "final_step": r["final_step_name"],
                    "n_mature": r["n_mature_predictions"],
                    "none_count": r["none_prediction_count"],
                    "n_errors": len(r["fsm_errors"]),
                    "first_error": r["first_error"],
                })

            with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"\n{'=' * 60}")
            print(f"烟雾测试总结 — {len(all_results)} 次测试")
            print(f"{'=' * 60}")
            for s in summary["per_test"]:
                icon = "✓" if s["fsm_complete"] and s["n_errors"] == 0 else "✗"
                print(f"  [{icon}] {s['test_name']}: path={s['fsm_path']} "
                      f"final={s['final_step']} errors={s['n_errors']} "
                      f"dur={s['duration_sec']}s")
            print(f"\n  Report: {out_dir / 'summary.json'}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
