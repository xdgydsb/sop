"""
本地海康相机图像采集 + WebSocket 发送到推理服务器

Usage:
  C:/Users/HP/AppData/Local/Programs/Python/Python310/python.exe local_client/hik_stream_client.py

前提:
  1. MVS 软件已关闭
  2. 相机 USB3 线已连接
  3. 服务器端 inference_server.py 已启动
"""
import sys
import os
import json
import time
import struct
import ctypes
import argparse
import numpy as np
import cv2
from pathlib import Path
from ctypes import c_uint, c_void_p, c_bool, POINTER, byref, cast, memset, sizeof

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── MVS SDK setup ──
_MVS_ROOT = r"D:\develop\MVS"
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

_MVS_SDK_PYTHON = os.path.join(_MVS_ROOT, "Development", "Samples", "Python")
_MVS_MV_IMPORT = os.path.join(_MVS_SDK_PYTHON, "MvImport")
sys.path.insert(0, _MVS_SDK_PYTHON)
sys.path.insert(0, _MVS_MV_IMPORT)

from MvCameraControl_class import (MvCamera, MV_CC_DEVICE_INFO_LIST,
                                    MVCC_INTVALUE, MVCC_FLOATVALUE,
                                    MVCC_ENUMVALUE, MV_FRAME_OUT)
from CameraParams_header import (MV_CC_DEVICE_INFO, MV_USB_DEVICE,
                                  MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE)
from MvErrorDefine_const import MV_OK
import PixelType_header as px

# ── WebSocket ──
from websockets.sync.client import connect as ws_connect

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


def open_hik_camera(target_w: int = 1280, target_h: int = 720):
    """Open MV-CS050-10UC, set target resolution, return (cam, w, h) or (None, 0, 0)."""
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

    cam.MV_CC_SetEnumValue("TriggerMode", 0)

    # ── Set target resolution BEFORE StartGrabbing ──
    print(f"  目标分辨率: {target_w}x{target_h}")
    ret_w = cam.MV_CC_SetIntValue("Width", target_w)
    ret_h = cam.MV_CC_SetIntValue("Height", target_h)
    if ret_w == MV_OK and ret_h == MV_OK:
        print(f"    设置成功")
    else:
        print(f"    设置失败 ret_w={to_hex_str(ret_w)} ret_h={to_hex_str(ret_h)}")

    st_exp = MVCC_FLOATVALUE()
    cam.MV_CC_GetFloatValue("ExposureTime", st_exp)
    if st_exp.fCurValue < 15000:
        cam.MV_CC_SetFloatValue("ExposureTime", 35000.0)
        print(f"  曝光: {st_exp.fCurValue:.0f} → 35000 us")
    else:
        print(f"  曝光: {st_exp.fCurValue:.0f} us (保持)")

    # Read actual resolution
    st_w = MVCC_INTVALUE()
    st_h = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("Width", st_w)
    cam.MV_CC_GetIntValue("Height", st_h)
    w, h = st_w.nCurValue, st_h.nCurValue
    print(f"  实际分辨率: {w}x{h}")

    ret = cam.MV_CC_StartGrabbing()
    if ret != MV_OK:
        print(f"  StartGrabbing FAILED: {to_hex_str(ret)}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        return None, 0, 0

    print("  ✓ 取流已开始")
    return cam, w, h


def close_hik_camera(cam):
    if cam is None:
        return
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


def draw_status(frame, local_fps, send_fps, server_info, out_w, out_h,
                connected, sequence_active=False):
    """Overlay status panel + render server detection boxes on frame.

    PREVIEW mode: camera + stable boxes + FPS only. NO STEP, NO ALARM, NO S1-S5.
    RUNNING mode: full SOP display with progress bar.
    """
    h, w = frame.shape[:2]
    px, py = 10, 30
    lh = 24

    # Determine mode from server response
    mode = server_info.get("mode", "PREVIEW") if server_info else "PREVIEW"
    is_yolo_debug = mode == "YOLO_DEBUG"
    is_running = mode in ("RUNNING", "COMPLETE", "ERROR")
    is_armed = mode == "ARMED"
    # ── YOLO_DEBUG mode display ──
    if is_yolo_debug:
        detections = server_info.get("detections", [])
        rejected_count = server_info.get("rejected_count", 0)
        rejected_summary = server_info.get("rejected_summary", {})
        raw_count = server_info.get("raw_count", 0)
        box_roi = server_info.get("box_roi")

        # Draw box_roi outline
        if box_roi and len(box_roi) == 4:
            rx1, ry1, rx2, ry2 = map(int, box_roi)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
            cv2.putText(frame, "box_roi", (rx1, ry1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw stable detections
        for det in detections:
            try:
                cls_name = det.get("class", "")
                conf = det.get("conf", 0)
                bbox = det.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                if "box" in cls_name:
                    color = (0, 255, 0)
                elif cls_name == "charger":
                    color = (255, 200, 0)
                elif cls_name == "green_bag":
                    color = (0, 200, 100)
                elif cls_name == "earphone":
                    color = (255, 255, 0)
                else:
                    color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}",
                            (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            except Exception:
                continue

        # ── YOLO_DEBUG info panel ──
        y_off = py + lh * 4
        cv2.putText(frame, f"MODE: YOLO_DEBUG  raw:{raw_count}  "
                    f"stable:{len(detections)}  rej:{rejected_count}",
                    (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 200), 2)
        y_off += lh
        if rejected_summary:
            rej_text = "Rej: " + " ".join(
                f"{k}={v}" for k, v in sorted(rejected_summary.items()))
            cv2.putText(frame, rej_text[:100],
                        (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (100, 100, 255), 1)
            y_off += lh
        # Box state hint
        box_state = server_info.get("box_state", "")
        cv2.putText(frame, f"Box: {box_state}",
                    (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0) if box_state == "closed" else
                    (0, 200, 255) if box_state == "open" else (150, 150, 150), 1)

        # Center message
        cv2.putText(frame, "YOLO DEBUG MODE — No SOP, No FSM, No Temporal",
                    (w // 2 - 250, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        cv2.putText(frame, "SPACE=back to SOP  R=Reset",
                    (w // 2 - 130, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # No S1-S5 bar in YOLO_DEBUG mode — return early
        return frame

    # ── Render server detection boxes ──
    if server_info:
        detections = server_info.get("detections", [])
        box_bbox = server_info.get("box_bbox")
        box_state = server_info.get("box_state", "")

        for det in detections:
            try:
                cls_name = det.get("class", "")
                conf = det.get("conf", 0)
                bbox = det.get("bbox", [])
                method = det.get("method", "")
                if len(bbox) != 4:
                    continue

                # ── Filter unreliable detections ──
                # Skip hsv_fallback green_bag (too many false positives)
                if cls_name == "green_bag" and method == "hsv_fallback":
                    continue
                # Skip very low confidence detections
                if cls_name in ("box_open", "box_closed") and conf < 0.45:
                    continue
                if cls_name in ("earphone", "charger", "green_bag") and conf < 0.25:
                    continue

                x1, y1, x2, y2 = map(int, bbox)

                # Color by class/source
                if "box" in cls_name:
                    color = (0, 255, 0)  # green for box
                elif cls_name == "charger" and method == "pink_marker":
                    color = (255, 0, 255)  # magenta — pink expansion
                elif cls_name == "charger" and method == "pink_guided":
                    color = (180, 100, 255)  # light purple
                elif cls_name == "charger" and method == "yolo+pink_marker":
                    color = (0, 165, 255)  # orange
                elif cls_name == "charger":
                    color = (255, 200, 0)  # gold — YOLO charger
                elif cls_name == "green_bag":
                    color = (0, 200, 100)  # teal
                elif cls_name == "earphone":
                    color = (255, 255, 0)  # cyan-yellow
                else:
                    color = (0, 255, 255)  # yellow

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                if method and method not in ("yolo",):
                    label += f" [{method}]"
                cv2.putText(frame, label,
                            (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            except Exception:
                continue

        # Box bbox highlight — color-coded by state
        if box_bbox and isinstance(box_bbox, list) and len(box_bbox) == 4:
            bx1, by1, bx2, by2 = map(int, box_bbox)
            state_colors = {
                "open": (0, 255, 0),
                "closed": (0, 165, 255),
                "transition": (0, 200, 255),
                "unknown": (100, 100, 100),
            }
            box_color = state_colors.get(box_state, (200, 200, 200))
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
            cv2.putText(frame, f"box:{box_state}", (bx1, by1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # ── Safety net: when box is CLOSED, suppress interior objects ──
        if box_state == "closed" and box_bbox and isinstance(box_bbox, list) and len(box_bbox) == 4:
            interior_objects = {"earphone", "charger", "green_bag"}
            bx1, by1, bx2, by2 = box_bbox
            margin = 20
            for det in detections:
                cls = det.get("class", "")
                if cls not in interior_objects:
                    continue
                dbbox = det.get("bbox", [])
                if len(dbbox) != 4:
                    continue
                cx = (dbbox[0] + dbbox[2]) / 2
                cy = (dbbox[1] + dbbox[3]) / 2
                if bx1 - margin <= cx <= bx2 + margin and by1 - margin <= cy <= by2 + margin:
                    # Object center inside closed box — should not be visible
                    # Draw over with background color to hide the box
                    ex1, ey1, ex2, ey2 = map(int, dbbox)
                    cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 0), -1)

    # ── Connection status ──
    conn_color = (0, 255, 0) if connected else (0, 0, 255)
    conn_text = "CONNECTED" if connected else "DISCONNECTED"
    cv2.putText(frame, conn_text, (w - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, conn_color, 2)

    # ── FPS panel ──
    cv2.putText(frame, f"Local FPS:{local_fps:.1f}", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Send FPS:{send_fps:.1f}", (px, py + lh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ── Server info ──
    if server_info:
        stage = server_info.get("stage", 0)
        s_fps = server_info.get("server_fps", 0)
        s_lat = server_info.get("latency_ms", 0)
        hands = server_info.get("hands_detected", 0)
        n_dets = len(server_info.get("detections", []))
        charger_src = server_info.get("charger_source", "")

        cv2.putText(frame, f"Svr FPS:{s_fps:.1f} Lat:{s_lat:.0f}ms "
                    f"Stage:{stage} Dets:{n_dets} Hands:{hands}",
                    (px, py + lh * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 0), 1)

        # ── MODE line ──
        mode_colors = {
            "PREVIEW": (100, 200, 100),
            "ARMED": (255, 200, 0),
            "RUNNING": (0, 255, 255),
            "COMPLETE": (0, 255, 0),
            "ERROR": (0, 0, 255),
        }
        mode_color = mode_colors.get(mode, (200, 200, 200))
        cv2.putText(frame, f"MODE: {mode}",
                    (px, py + lh * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    mode_color, 2)

        # ── Step info (available in all modes) ──
        s_step = server_info.get("current_step", "-")
        s_step_id = server_info.get("current_step_id", 0)
        s_conf = server_info.get("confidence", 0)
        s_alarm = server_info.get("alarm")
        is_complete = server_info.get("is_complete", False)
        fsm_path = server_info.get("fsm_path", [])
        model_pred = server_info.get("model_pred", -1)
        step_probs = server_info.get("step_probs", [])
        event_name = server_info.get("event_name")
        event_conf = server_info.get("event_confidence", 0)
        event_rejected = server_info.get("event_rejected", "")
        expected_event = server_info.get("expected_event", "")
        accepted_events = server_info.get("accepted_events", [])
        current_action = server_info.get("current_action", "none")
        action_phase = server_info.get("action_phase", "WAITING")

        # ── RUNNING/COMPLETE/ERROR: FSM-driven display ──
        if is_running or is_armed:
            y_off = py + lh * 4

            # Expected event
            if expected_event:
                exp_color = (0, 255, 200) if event_name else (200, 200, 100)
                cv2.putText(frame, f"Expect: {expected_event}",
                            (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, exp_color, 2)
                y_off += lh

            # Current action
            if current_action and current_action != "none":
                phase_color = {
                    "ACTIVE": (0, 200, 255),
                    "COMPLETING": (0, 255, 200),
                    "DONE": (0, 255, 0),
                    "WRONG_ACTIVE": (0, 0, 255),
                }.get(action_phase, (200, 200, 200))
                cv2.putText(frame, f"Action: {current_action} [{action_phase}]",
                            (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, phase_color, 1)
                y_off += lh

            # FSM step
            if is_running or (is_armed and s_step_id > 0):
                cv2.putText(frame, f"Step: {s_step} conf:{s_conf:.2f}",
                            (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                y_off += lh

            # Event
            if event_name:
                cv2.putText(frame, f"Event: {event_name} ({event_conf:.2f})",
                            (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)
                y_off += lh
            elif event_rejected:
                cv2.putText(frame, f"Rej: {event_rejected[:60]}",
                            (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)
                y_off += lh

            # Accepted events
            if accepted_events:
                ae_text = "OK: " + " -> ".join(accepted_events)
                cv2.putText(frame, ae_text[:80],
                            (px, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 100), 1)
                y_off += lh

            # Alarm
            if s_alarm:
                cv2.putText(frame, f"ALARM: {s_alarm}", (px, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_off += lh

        # ── PREVIEW: minimal info only ──
        elif mode == "PREVIEW":
            if model_pred > 0:
                cv2.putText(frame, f"Predict: S{model_pred} conf:{s_conf:.2f}",
                            (px, py + lh * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (150, 150, 80), 1)

        # ── S1-S5 Progress Bar (RUNNING/ARMED/COMPLETE/ERROR only, NOT PREVIEW) ──
        if stage >= 3 and mode != "PREVIEW":
            _draw_sop_progress(frame, s_step_id, fsm_path, is_complete, s_alarm,
                              mode, model_pred, step_probs, action_phase, current_action)

        # Charger source
        if charger_src:
            src_color = {"yolo": (0, 255, 0),
                         "pink_marker": (255, 0, 255),
                         "white_guided": (0, 200, 255),
                         "pink_guided": (180, 100, 255)}.get(charger_src, (200, 200, 200))
            y_row = py + lh * (6 if is_running and server_info.get("alarm") else
                               6 if is_running else 4)
            cv2.putText(frame, f"Charger: {charger_src}",
                        (px, y_row), cv2.FONT_HERSHEY_SIMPLEX, 0.4, src_color, 1)

        # Start blocked reason
        start_blocked = server_info.get("start_blocked")
        if start_blocked:
            cv2.putText(frame, str(start_blocked),
                        (w // 2 - 250, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # Send resolution
    cv2.putText(frame, f"Send: {out_w}x{out_h} JPEG",
                (px, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1)

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(frame, "Waiting for box_opened...",
                    (w // 2 - 140, h // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    elif mode == "RUNNING":
        cv2.putText(frame, "SOP RUNNING...", (w // 2 - 100, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif mode == "COMPLETE":
        cv2.putText(frame, "SOP COMPLETE!", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(frame, "Press SPACE to restart",
                    (w // 2 - 130, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif mode == "ERROR":
        cv2.putText(frame, "ERROR DETECTED", (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    return frame


def _draw_sop_progress(frame, current_step_id, fsm_path, is_complete, alarm,
                      mode="PREVIEW", model_pred=-1, step_probs=None,
                      action_phase="WAITING", current_action="none"):
    """Draw S1-S5 progress bar at top-center of frame.

    PREVIEW/ARMED: temporal predictions only (dim colors)
    RUNNING: FSM-driven — next step only lights up when action is ACTIVE/COMPLETING
    COMPLETE: all green
    ERROR: red on error step
    """
    # Map action name to step id
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

    # For PREVIEW/ARMED, use model_pred as the "current" step
    display_step = current_step_id if is_running else model_pred
    # Build path from FSM for running, or from model_pred for preview
    display_path = fsm_path if is_running else (
        list(range(1, model_pred)) if model_pred > 1 else [])

    for i in range(total_steps):
        sid = i + 1
        x1 = margin_left + i * (step_w + gap)
        x2 = x1 + step_w
        y1 = bar_y
        y2 = bar_y + bar_h

        # Determine color based on mode
        if alarm and is_running:
            if sid == display_step:
                color = (0, 0, 255)  # red — error on this step
            elif sid in display_path:
                color = (0, 180, 0)  # green — completed
            else:
                color = (80, 80, 80)  # dark gray
        elif is_complete and is_running:
            color = (0, 220, 0)  # all green — done
        elif is_running:
            # FSM-driven: current_step_id = LAST COMPLETED step
            # Steps <= current_step_id are DONE (green)
            # Step current_step_id+1 is NEXT — BUT only highlight if user is actually doing it
            # Steps > current_step_id+1 are FUTURE (dark gray)
            next_step = display_step + 1
            next_active = (
                action_phase in ("ACTIVE", "COMPLETING", "WRONG_ACTIVE")
                and _action_to_step.get(current_action, -1) == next_step
            )
            if sid <= display_step:
                color = (0, 180, 0)  # green — completed
            elif sid == next_step and next_active:
                color = (0, 240, 255)  # bright yellow-cyan — currently doing this step
            elif sid == next_step:
                color = (60, 60, 90)  # dim blue-gray — expected next, not yet started
            else:
                color = (80, 80, 80)  # dark gray — future
        elif is_preview and display_step > 0:
            # Temporal-only: dim colors to indicate "prediction, not confirmed"
            if sid == display_step:
                # Use step_probs to adjust brightness
                prob = 0.5
                if step_probs and sid < len(step_probs):
                    prob = step_probs[sid]
                b = int(100 + 80 * prob)  # dim to medium brightness
                color = (b, b, 60)  # dim yellow — predicted current
            elif sid < display_step:
                color = (60, 100, 60)  # dim green — likely completed
            else:
                color = (50, 50, 50)  # very dark gray — not reached
        else:
            # PREVIEW with no prediction yet
            color = (50, 50, 50)  # all dark gray

        # Draw filled rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # Step label
        text = step_names[i]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        tx = x1 + (step_w - tw) // 2
        ty = y1 + (bar_h + th) // 2
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="海康相机 → WebSocket 图像流")
    parser.add_argument("--server", default="192.168.31.19",
                        help="推理服务器地址")
    parser.add_argument("--port", type=int, default=8765,
                        help="服务器端口")
    parser.add_argument("--width", type=int, default=1280,
                        help="发送宽度")
    parser.add_argument("--height", type=int, default=720,
                        help="发送高度")
    parser.add_argument("--quality", type=int, default=70,
                        help="JPEG 质量 (1-100)")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="目标发送帧率")
    parser.add_argument("--no-display", action="store_true",
                        help="不显示本地画面")
    args = parser.parse_args()

    print("=" * 60)
    print("海康相机 WebSocket 图像流客户端")
    print(f"  服务器: ws://{args.server}:{args.port}")
    print(f"  发送分辨率: {args.width}x{args.height}")
    print(f"  JPEG 质量: {args.quality}")
    print(f"  目标帧率: {args.fps} FPS")
    print("=" * 60)

    # ── Open camera ──
    print("\n⚠  请确认 MVS 软件已关闭")
    cam, cam_w, cam_h = open_hik_camera()
    if cam is None:
        print("ERROR: 无法打开相机")
        return 1

    out_w, out_h = args.width, args.height
    frame_interval = 1.0 / args.fps
    server_url = f"ws://{args.server}:{args.port}"

    print(f"\n[Client] 连接服务器 {server_url}...")

    frame_id = 0
    send_count = 0
    recv_count = 0
    local_fps = 0.0
    send_fps = 0.0
    server_info = None
    connected = False
    sequence_active = False   # True when server is in RUNNING mode

    fps_times = []
    send_times = []

    try:
        with ws_connect(server_url, max_size=5 * 1024 * 1024) as ws:
            connected = True
            print("[Client] ✓ 已连接")
            print("[Client] 开始发送图像流...")

            last_send = time.time()

            while True:
                loop_start = time.time()

                # ── Grab camera frame ──
                st_frame = MV_FRAME_OUT()
                memset(byref(st_frame), 0, sizeof(MV_FRAME_OUT))
                ret = cam.MV_CC_GetImageBuffer(st_frame, 1000)

                if ret != MV_OK or st_frame.pBufAddr is None:
                    # No frame available, just wait
                    time.sleep(0.005)
                    continue

                try:
                    frame = convert_frame_to_bgr(st_frame)
                finally:
                    cam.MV_CC_FreeImageBuffer(st_frame)

                # ── FPS tracking ──
                now = time.time()
                fps_times.append(now)
                if len(fps_times) > 30:
                    fps_times = fps_times[-30:]
                if len(fps_times) >= 2:
                    local_fps = (len(fps_times) - 1) / (
                        fps_times[-1] - fps_times[0])

                # ── FPS throttle ──
                if now - last_send < frame_interval:
                    # Display only (no send this frame)
                    if not args.no_display:
                        display = cv2.resize(frame, (out_w, out_h))
                        display = draw_status(display, local_fps, send_fps,
                                              server_info, out_w, out_h,
                                              connected, sequence_active)
                        cv2.imshow("Camera Client", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue

                last_send = now
                frame_id += 1

                # ── Resize + JPEG encode ──
                send_frame = cv2.resize(frame, (out_w, out_h))
                _, jpeg = cv2.imencode('.jpg', send_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                jpeg_bytes = jpeg.tobytes()

                # ── Send: [frame_id: uint32 LE] [timestamp: float64 LE] [jpeg] ──
                try:
                    header = struct.pack('<Id', frame_id, now)
                    ws.send(header + jpeg_bytes)
                    send_count += 1

                    send_times.append(now)
                    if len(send_times) > 30:
                        send_times = send_times[-30:]
                    if len(send_times) >= 2:
                        send_fps = (len(send_times) - 1) / (
                            send_times[-1] - send_times[0])

                except Exception as e:
                    print(f"\n[Client] 发送失败: {e}")
                    connected = False
                    break

                # ── Receive latest server response (non-blocking) ──
                try:
                    response_bytes = ws.recv(timeout=0.001)
                    if response_bytes:
                        msg = json.loads(response_bytes)

                        # Handle control acknowledgements
                        if "control_ack" in msg:
                            ack = msg["control_ack"]
                            if ack == "start":
                                if msg.get("success"):
                                    sequence_active = True
                                    server_info = {"mode": "ARMED", **msg}
                                    print(f"[Client] START accepted: {msg.get('message', '')}")
                                else:
                                    sequence_active = False
                                    print(f"[Client] START blocked: {msg.get('reason', '')}")
                            elif ack == "stop":
                                sequence_active = False
                                server_info = {"mode": "PREVIEW"}
                                print(f"[Client] STOPPED")
                            elif ack == "reset":
                                sequence_active = False
                                server_info = {"mode": "PREVIEW"}
                                print(f"[Client] RESET complete")
                            recv_count += 1
                        else:
                            server_info = msg
                            # Track mode from server
                            svr_mode = server_info.get("mode", "PREVIEW")
                            sequence_active = svr_mode in ("ARMED", "RUNNING", "COMPLETE", "ERROR")
                            if recv_count == 0:
                                print(f"[Client] First response: {json.dumps(server_info, ensure_ascii=False, default=str)[:300]}")
                            recv_count += 1
                except TimeoutError:
                    pass
                except Exception:
                    pass

                # ── Display ──
                if not args.no_display:
                    display = cv2.resize(frame, (out_w, out_h))
                    display = draw_status(display, local_fps, send_fps,
                                          server_info, out_w, out_h,
                                          connected, sequence_active)
                    cv2.imshow("Camera Client", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Client] quit")
                        break
                    elif key == ord('s'):
                        save_path = f"frame_{frame_id:06d}.jpg"
                        cv2.imwrite(save_path, display)
                        print(f"[Client] Saved: {save_path}")
                    elif key == ord('r'):  # R = reset to PREVIEW
                        try:
                            ws.send("reset")
                            sequence_active = False
                            server_info = None
                            print("[Client] RESET → PREVIEW")
                        except Exception as e:
                            print(f"[Client] reset failed: {e}")
                    elif key == ord(' '):  # SPACE = start SOP detection
                        svr_mode = server_info.get("mode", "PREVIEW") if server_info else "PREVIEW"
                        if svr_mode in ("RUNNING",):
                            # Already running — stop first
                            try:
                                ws.send("stop")
                                sequence_active = False
                                print("[Client] STOP → PREVIEW")
                            except Exception as e:
                                print(f"[Client] stop failed: {e}")
                        elif svr_mode in ("COMPLETE", "ERROR"):
                            # Reset then start fresh
                            try:
                                ws.send("reset")
                                time.sleep(0.2)
                                ws.send("start")
                                print("[Client] RESET + START")
                            except Exception as e:
                                print(f"[Client] reset+start failed: {e}")
                        else:
                            # PREVIEW → try to start
                            try:
                                ws.send("start")
                                print("[Client] START sent")
                            except Exception as e:
                                print(f"[Client] start failed: {e}")

    except ConnectionRefusedError:
        print(f"\n[Client] 无法连接 {server_url} — 服务器启动了吗?")
    except Exception as e:
        print(f"\n[Client] 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        close_hik_camera(cam)
        print(f"\n[Client] 统计: 发送 {send_count} 帧, 接收 {recv_count} 个响应")
        print("[Client] 退出")

    return 0


if __name__ == "__main__":
    sys.exit(main())
