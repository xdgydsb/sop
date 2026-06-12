"""Camera base class"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class CameraBase(ABC):
    """Abstract camera interface"""

    @abstractmethod
    def open(self) -> bool:
        """Open camera, return success"""
        ...

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame, return (success, frame)"""
        ...

    @abstractmethod
    def release(self):
        """Release camera"""
        ...

    @abstractmethod
    def get_fps(self) -> float:
        """Get camera FPS"""
        ...
