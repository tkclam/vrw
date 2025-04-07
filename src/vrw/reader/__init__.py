from functools import cache, cached_property
import numpy as np
from os import PathLike
import cv2

class VideoReader:
    def __init__(self, path: str | PathLike, to_gray=False):
        self._cap = cv2.VideoCapture(str(path))
        if to_gray:
            self._color_mode = cv2.COLOR_BGR2GRAY
        else:
            self._color_mode = cv2.COLOR_BGR2RGB

    def __getitem__(self, key) -> np.ndarray:
        if np.issubdtype(type(key), np.integer):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, key)
            ret, frame = self._cap.read()
            if not ret:
                raise IndexError(f"Frame {key} not found")
            return cv2.cvtColor(frame, self._color_mode)
        else:
            raise NotImplementedError(f"Indexing with {type(key)} is not supported")
    
    @cache
    def __len__(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @cached_property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)
    
    @cached_property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @cached_property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @cached_property
    def shape(self) -> tuple[int, int, int, int] | tuple[int, int, int]:
        if self._color_mode == cv2.COLOR_BGR2GRAY:
            return (len(self), self.height, self.width)
        else:
            return (len(self), self.height, self.width, 3)

    def __del__(self):
        if self._cap.isOpened():
            self._cap.release()
