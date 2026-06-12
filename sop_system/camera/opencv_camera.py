"""OpenCV camera — standard USB/UVC cameras including Hikvision USB in UVC mode"""
import cv2
import numpy as np
from typing import Tuple, Optional
from .camera_base import CameraBase


class OpenCVCamera(CameraBase):
    """OpenCV-based camera capture"""

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, use_dshow: bool = True):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.use_dshow = use_dshow
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if self.use_dshow:
            self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.camera_id)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_fps(self) -> float:
        if self._cap is not None:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
        return 30.0
