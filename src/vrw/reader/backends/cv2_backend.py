from .backend import Backend
from os import PathLike
import numpy as np
import cv2

class Cv2Backend(Backend):
    def __init__(self, path: str | PathLike, to_gray=False):
        super().__init__(path, to_gray)

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {self._path}")
        
        if self._to_gray:
            self._color_mode = cv2.COLOR_BGR2GRAY
        else:
            self._color_mode = cv2.COLOR_BGR2RGB
        
    def get_frame(self, frame_id: int) -> np.ndarray:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self._cap.read()
        if not ret:
            raise IndexError(f"Frame {frame_id} not found")
        return cv2.cvtColor(frame, self._color_mode)

    @property
    def n_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)
    
    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
