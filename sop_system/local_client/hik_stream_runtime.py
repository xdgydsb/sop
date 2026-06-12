import ctypes
import os
import sys
from ctypes import POINTER, byref, cast, memset, sizeof

import cv2
import numpy as np
from websockets.sync.client import connect as ws_connect

_MVS_ROOT = os.environ.get("MVS_ROOT", "")
if not _MVS_ROOT or not os.path.isdir(_MVS_ROOT):
    for candidate in (
        r"D:\develop\MVS",
        r"C:\Program Files (x86)\MVS",
        r"C:\Program Files\MVS",
        r"D:\MVS",
    ):
        if os.path.isdir(candidate):
            _MVS_ROOT = candidate
            break

if not _MVS_ROOT:
    raise RuntimeError("MVS SDK not found. Set MVS_ROOT or install Hikvision MVS.")

_MVS_SDK_PYTHON = os.path.join(_MVS_ROOT, "Development", "Samples", "Python")
_MVS_MV_IMPORT = os.path.join(_MVS_SDK_PYTHON, "MvImport")
for dll_dir in (
    os.path.join(_MVS_ROOT, "Development", "Libraries", "win64"),
    os.path.join(_MVS_ROOT, "Runtime", "win64"),
):
    if os.path.isdir(dll_dir):
        try:
            os.add_dll_directory(dll_dir)
        except Exception:
            pass
sys.path.insert(0, _MVS_SDK_PYTHON)
sys.path.insert(0, _MVS_MV_IMPORT)

from MvCameraControl_class import MvCamera, MV_CC_DEVICE_INFO_LIST, MVCC_FLOATVALUE, MVCC_INTVALUE, MV_FRAME_OUT
from CameraParams_header import MV_CC_DEVICE_INFO, MV_GIGE_DEVICE, MV_USB_DEVICE
from MvErrorDefine_const import MV_OK
import PixelType_header as px


def open_hik_camera(target_w: int = 1280, target_h: int = 720):
    print("[Camera] Initialize SDK...")
    try:
        MvCamera.MV_CC_Finalize()
    except Exception:
        pass
    ret = MvCamera.MV_CC_Initialize()
    if ret != MV_OK:
        print(f"[Camera] MV_CC_Initialize failed: {ret:#x}")
        return None, 0, 0

    print("[Camera] Enumerate devices...")
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if ret != MV_OK or device_list.nDeviceNum == 0:
        print(f"[Camera] No device found ret={ret:#x} n={device_list.nDeviceNum}")
        return None, 0, 0

    print(f"[Camera] Found {device_list.nDeviceNum} device(s)")
    target_serial = "DA9562204"
    selected_idx = -1
    for i in range(device_list.nDeviceNum):
        mvcc_dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType != MV_USB_DEVICE:
            continue
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
        elif selected_idx < 0:
            selected_idx = i

    if selected_idx < 0:
        print("[Camera] USB camera not found")
        return None, 0, 0

    print(f"[Camera] Select [{selected_idx}]")
    st_device = cast(device_list.pDeviceInfo[selected_idx], POINTER(MV_CC_DEVICE_INFO)).contents
    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(st_device)
    if ret != MV_OK:
        print(f"[Camera] CreateHandle failed: {ret:#x}")
        return None, 0, 0

    ret = cam.MV_CC_OpenDevice()
    if ret != MV_OK:
        print(f"[Camera] OpenDevice failed: {ret:#x}")
        cam.MV_CC_DestroyHandle()
        return None, 0, 0

    cam.MV_CC_SetEnumValue("TriggerMode", 0)
    print(f"[Camera] Target resolution: {target_w}x{target_h}")
    ret_w = cam.MV_CC_SetIntValue("Width", target_w)
    ret_h = cam.MV_CC_SetIntValue("Height", target_h)
    if ret_w == MV_OK and ret_h == MV_OK:
        print("[Camera] Resolution set")
    else:
        print(f"[Camera] Resolution set failed ret_w={ret_w:#x} ret_h={ret_h:#x}")

    st_exp = MVCC_FLOATVALUE()
    cam.MV_CC_GetFloatValue("ExposureTime", st_exp)
    if st_exp.fCurValue < 15000:
        cam.MV_CC_SetFloatValue("ExposureTime", 35000.0)
        print(f"[Camera] Exposure: {st_exp.fCurValue:.0f} -> 35000 us")
    else:
        print(f"[Camera] Exposure: {st_exp.fCurValue:.0f} us (keep)")

    st_w = MVCC_INTVALUE()
    st_h = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("Width", st_w)
    cam.MV_CC_GetIntValue("Height", st_h)
    w, h = st_w.nCurValue, st_h.nCurValue
    print(f"[Camera] Actual resolution: {w}x{h}")

    ret = cam.MV_CC_StartGrabbing()
    if ret != MV_OK:
        print(f"[Camera] StartGrabbing failed: {ret:#x}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        return None, 0, 0

    print("[Camera] Stream started")
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


def convert_frame_to_bgr(st_frame: MV_FRAME_OUT):
    fi = st_frame.stFrameInfo
    w, h, fmt = fi.nWidth, fi.nHeight, fi.enPixelType
    buf = st_frame.pBufAddr

    if fmt == px.PixelType_Gvsp_Mono8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents
        )
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_GRAY2BGR)
    if fmt == px.PixelType_Gvsp_RGB8_Packed:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h * 3))).contents
        )
        return cv2.cvtColor(raw.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
    if fmt == px.PixelType_Gvsp_BGR8_Packed:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h * 3))).contents
        )
        return raw.reshape((h, w, 3))
    if fmt == px.PixelType_Gvsp_BayerRG8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents
        )
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerRG2BGR)
    if fmt == px.PixelType_Gvsp_BayerGB8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents
        )
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerGB2BGR)
    if fmt == px.PixelType_Gvsp_BayerGR8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents
        )
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerGR2BGR)
    if fmt == px.PixelType_Gvsp_BayerBG8:
        raw = np.ctypeslib.as_array(
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents
        )
        return cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerBG2BGR)

    raw = np.ctypeslib.as_array(
        ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * fi.nFrameLen)).contents
    )
    return raw.reshape((h, w, 3))


def draw_status(frame, local_fps, send_fps, server_info, out_w, out_h, show_overlay=True, sequence_active=True):
    if not show_overlay:
        return frame

    server_info = server_info or {}
    mode = server_info.get("mode", "PREVIEW")
    expected = server_info.get("expected_event", "-")
    expected_step_map = {
        "box_opened": "S1",
        "earphone_in_box": "S2",
        "charger_in_box": "S3",
        "green_bag_in_box": "S4",
        "box_closed": "S5",
    }
    current_step = expected_step_map.get(expected, str(server_info.get("current_step_id") or server_info.get("current_step") or "-"))
    alarm = server_info.get("alarm")

    cv2.putText(frame, f"Local FPS:{local_fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Send FPS:{send_fps:.1f}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Mode: {mode}", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Step: {current_step}", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
    cv2.putText(frame, f"Expect: {expected}", (12, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)

    if alarm:
        cv2.putText(
            frame,
            f"ALARM: {alarm.get('type', 'ALARM')} {alarm.get('message', '')}",
            (12, 176),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    detections = server_info.get("detections") or []
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cls_name = det.get("class", "obj")
        conf = float(det.get("conf", 0.0) or 0.0)
        color = (0, 255, 255)
        if cls_name in ("box_open", "box_closed"):
            color = (0, 200, 255) if cls_name == "box_closed" else (0, 255, 0)
        elif cls_name == "earphone":
            color = (255, 255, 0)
        elif cls_name == "charger":
            color = (255, 200, 0)
        elif cls_name == "green_bag":
            color = (0, 200, 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return frame


def decoding_char(raw_bytes):
    c_char_p = ctypes.cast(raw_bytes, ctypes.c_char_p)
    value = c_char_p.value
    if not value:
        return ""
    for encoding in ("utf-8", "gbk", "latin1"):
        try:
            return value.decode(encoding)
        except Exception:
            continue
    return str(value)
