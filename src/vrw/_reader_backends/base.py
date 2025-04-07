from abc import ABC, abstractmethod
import numpy as np
from os import PathLike


class ReaderBackend(ABC):
    def __init__(self, path: str | PathLike, to_gray: bool):
        self._path = str(path)
        self._to_gray = to_gray

    @staticmethod
    def from_name(name: str) -> type["ReaderBackend"]:
        """Create a Backend instance from a name."""
        if name == "cv2":
            from .cv2_backend import Cv2Backend

            return Cv2Backend

        raise NotImplementedError(
            f"Backend '{name}' is not implemented. Available backends: cv2"
        )

    @abstractmethod
    def get_frame(self, frame_id: int) -> np.ndarray:
        pass

    def iter_frames(self, frame_ids):
        for i in frame_ids:
            yield self.get_frame(i)

    def iter_slice(self, slice_: slice):
        return self.iter_frames(range(*slice_.indices(self.n_frames)))

    def iter_continuous_slice(self, start: int, stop: int):
        return self.iter_slice(slice(start, stop))

    def iter_all_frames(self):
        return self.iter_continuous_slice(0, self.n_frames)

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

    def close(self):
        """Close the video file."""
        pass

    @property
    def dtype(self) -> np.dtype:
        """Data type of the frames."""
        return np.uint8
