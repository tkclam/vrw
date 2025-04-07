from abc import ABC, abstractmethod
import numpy as np

class Backend(ABC):
    def __init__(self, filename: str, fps: float):
        self._filename = filename
        self._fps = fps
        self._n_frames = 0

    @staticmethod
    def from_name(name: str):
        if name == "cv2":
            from .cv2_backend import Cv2Backend
            return Cv2Backend
        raise ValueError(f"Unknown backend: {name}. Available backends: cv2")

    @abstractmethod
    def write(self, frame: np.ndarray):
        pass

    @abstractmethod
    def _init(self, frame_shape: tuple):
        pass

    def close(self):
        pass

    def __len__(self):
        return self._n_frames

    def __del__(self):
        self.close()
