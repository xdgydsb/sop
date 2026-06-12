"""Camera module — OpenCV + Hikvision SDK support"""
from .camera_base import CameraBase
from .opencv_camera import OpenCVCamera
from .hikvision_camera import HikvisionCamera, is_hikvision_sdk_available
