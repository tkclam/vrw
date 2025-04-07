from .backends import Backend
import numpy as np


class VideoWriter:
    def __init__(self, filename: str, fps: float, backend: str = "cv2", **kwargs):
        self._backend = Backend.from_name(backend)(filename, fps, **kwargs)

    def write(self, frame: np.ndarray):
        """Write a frame to the video file."""
        self._backend.write(frame)

    def close(self):
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self._backend)
