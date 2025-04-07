from functools import cache, cached_property
import numpy as np
from os import PathLike
from .backends import Backend


class VideoReader:
    def __init__(self, path: str | PathLike, to_gray=False, backend="cv2"):
        self._backend = Backend.from_name(backend)(path, to_gray=to_gray)

    def __getitem__(self, key) -> np.ndarray:
        if np.issubdtype(type(key), np.integer):
            return self._backend.get_frame(key)
        else:
            raise NotImplementedError(f"Indexing with {type(key)} is not supported")

    @cache
    def __len__(self):
        return self._backend.n_frames

    @cached_property
    def fps(self) -> float:
        return self._backend.fps

    @cached_property
    def width(self) -> int:
        return self._backend.width

    @cached_property
    def height(self) -> int:
        return self._backend.height

    @cached_property
    def shape(self) -> tuple[int, int, int, int] | tuple[int, int, int]:
        return self._backend.shape
