from abc import ABC, abstractmethod
import numpy as np
from os import PathLike

class Backend(ABC):
    def __init__(self, path: str | PathLike, to_gray: bool):
        self._path = str(path)
        self._to_gray = to_gray

    @staticmethod
    def from_name(name: str) -> type["Backend"]:
        """Create a Backend instance from a name."""
        if name == "cv2":
            from .cv2_backend import Cv2Backend
            return Cv2Backend
        else:
            raise NotImplementedError(f"Backend '{name}' is not implemented.")

    @abstractmethod
    def get_frame(self, frame_id: int) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """Number of frames in the video."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Frames per second of the video."""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Width of each frame."""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Height of each frame."""
        pass

    @property
    def shape(self) -> tuple[int, int, int] | tuple[int, int, int, int]:
        """Shape of each frame."""
        if self._to_gray:
            return self.n_frames, self.height, self.width
        else:
            return self.n_frames, self.height, self.width, 3
