"""
海康威视USB工业相机 — MVS SDK封装
"""
import sys
import os
import ctypes
import numpy as np
from typing import Tuple, Optional
from .camera_base import CameraBase

# SDK路径
_SDK_PATHS = [
    r"D:\develop\MVS\Development\Samples\Python",
    r"C:\Program Files (x86)\MVS\Development\Samples\Python",
    r"C:\Program Files\MVS\Development\Samples\Python",
]

_MV_IMPORT_PATH = ""
_INITIALIZED = False


def _setup_path() -> bool:
    global _MV_IMPORT_PATH
    if _MV_IMPORT_PATH:
        return True
    for base in _SDK_PATHS:
        mv = os.path.join(base, "MvImport")
        if os.path.isdir(mv):
            _MV_IMPORT_PATH = mv
            if base not in sys.path:
                sys.path.insert(0, base)
            if mv not in sys.path:
                sys.path.insert(0, mv)
            return True
    return False


def is_hikvision_sdk_available() -> bool:
    for base in _SDK_PATHS:
        if os.path.isdir(os.path.join(base, "MvImport")):
            return True
    return False


class HikvisionCamera(CameraBase):
    """海康威视USB/GigE工业相机"""

    def __init__(self, camera_index: int = 0):
        self._index = camera_index
        self._cam = None
        self._w = 0
        self._h = 0
        self._payload = 0
        self._grabbing = False

    def open(self) -> bool:
        global _INITIALIZED

        if not _setup_path():
            print("[HikCamera] SDK not found")
            return False

        from MvCameraControl_class import MvCamera, MV_CC_DEVICE_INFO_LIST, MV_FRAME_OUT, MVCC_INTVALUE
        from CameraParams_header import MV_CC_DEVICE_INFO, MV_USB_DEVICE, MV_GIGE_DEVICE
        from MvErrorDefine_const import MV_OK

        # 初始化（每次进程只调一次）
        if not _INITIALIZED:
            MvCamera.MV_CC_Initialize()
            _INITIALIZED = True

        # 枚举
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, device_list)
        if ret != MV_OK or device_list.nDeviceNum == 0:
            print("[HikCamera] No device (ret=0x%x, n=%d)" % (ret, device_list.nDeviceNum))
            return False

        n = device_list.nDeviceNum
        print("[HikCamera] Found %d device(s)" % n)
        if self._index >= n:
            print("[HikCamera] Index %d >= %d" % (self._index, n))
            return False

        # 获取设备信息结构体
        st_device = ctypes.cast(
            device_list.pDeviceInfo[self._index],
            ctypes.POINTER(MV_CC_DEVICE_INFO)
        ).contents

        # 创建句柄 + 打开
        self._cam = MvCamera()
        ret = self._cam.MV_CC_CreateHandle(st_device)
        if ret != MV_OK:
            print("[HikCamera] CreateHandle failed: 0x%x" % ret)
            return False

        ret = self._cam.MV_CC_OpenDevice()
        if ret != MV_OK:
            print("[HikCamera] OpenDevice failed: 0x%x (close MVS client!)" % ret)
            self._cam.MV_CC_DestroyHandle()
            self._cam = None
            return False

        # 关闭触发模式
        self._cam.MV_CC_SetEnumValue("TriggerMode", 0)

        # 获取参数
        st = MVCC_INTVALUE()
        self._cam.MV_CC_GetIntValue("Width", st)
        self._w = st.nCurValue
        self._cam.MV_CC_GetIntValue("Height", st)
        self._h = st.nCurValue
        self._cam.MV_CC_GetIntValue("PayloadSize", st)
        self._payload = st.nCurValue

        # 开始取流
        ret = self._cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            print("[HikCamera] StartGrabbing failed: 0x%x" % ret)
            self._cam.MV_CC_CloseDevice()
            self._cam.MV_CC_DestroyHandle()
            self._cam = None
            return False

        self._grabbing = True
        print("[HikCamera] Opened %dx%d payload=%d" % (self._w, self._h, self._payload))
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cam is None or not self._grabbing:
            return False, None

        from MvCameraControl_class import MV_FRAME_OUT

        st_frame = MV_FRAME_OUT()
        ret = self._cam.MV_CC_GetImageBuffer(st_frame, 1000)
        if ret != 0 or st_frame.pBufAddr is None:
            return False, None

        try:
            fi = st_frame.stFrameInfo
            w, h = fi.nWidth, fi.nHeight
            fmt = fi.enPixelType

            if fmt == 0x02180014:  # BayerRG8
                import cv2
                raw = np.ctypeslib.as_array(
                    ctypes.cast(st_frame.pBufAddr,
                                ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
                frame = cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_BayerRG2BGR)

            elif fmt == 0x0210001f:  # RGB8Packed
                raw = np.ctypeslib.as_array(
                    ctypes.cast(st_frame.pBufAddr,
                                ctypes.POINTER(ctypes.c_ubyte * (w * h * 3))).contents)
                import cv2
                frame = cv2.cvtColor(raw.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)

            elif fmt in (0x01080001,):  # Mono8
                import cv2
                raw = np.ctypeslib.as_array(
                    ctypes.cast(st_frame.pBufAddr,
                                ctypes.POINTER(ctypes.c_ubyte * (w * h))).contents)
                frame = cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_GRAY2BGR)

            else:
                raw = np.ctypeslib.as_array(
                    ctypes.cast(st_frame.pBufAddr,
                                ctypes.POINTER(ctypes.c_ubyte * self._payload)).contents)
                try:
                    frame = raw.reshape((h, w, 3))
                except Exception:
                    import cv2
                    frame = cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_GRAY2BGR)
        finally:
            self._cam.MV_CC_FreeImageBuffer(st_frame)

        return True, frame

    def release(self):
        if self._cam is not None:
            if self._grabbing:
                self._cam.MV_CC_StopGrabbing()
                self._grabbing = False
            self._cam.MV_CC_CloseDevice()
            self._cam.MV_CC_DestroyHandle()
            self._cam = None
            print("[HikCamera] Released")

    def get_fps(self) -> float:
        if self._cam is None:
            return 30.0
        from MvCameraControl_class import MVCC_FLOATVALUE
        st_fps = MVCC_FLOATVALUE()
        ret = self._cam.MV_CC_GetFloatValue("ResultingFrameRate", st_fps)
        return st_fps.fCurValue if ret == 0 else 30.0


def open_camera(camera_id: int = 0, prefer_hikvision: bool = True,
                width: int = 1280, height: int = 720) -> Tuple[bool, CameraBase]:
    """智能打开：海康SDK优先，OpenCV回退"""
    if prefer_hikvision and is_hikvision_sdk_available():
        cam = HikvisionCamera(camera_index=camera_id)
        if cam.open():
            return True, cam
        print("[Camera] Hikvision failed, try OpenCV...")

    from .opencv_camera import OpenCVCamera
    cam = OpenCVCamera(camera_id=camera_id, width=width, height=height)
    if cam.open():
        return True, cam
    return False, None
