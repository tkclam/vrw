from functools import cache, cached_property
import numpy as np
from os import PathLike
from ._reader_backends import ReaderBackend

__all__ = ["VideoReader"]


class VideoReader:
    def __init__(self, path: str | PathLike, to_gray=False, backend="cv2"):
        self._backend = ReaderBackend.from_name(backend)(path, to_gray=to_gray)

    def _get_frames_iter(self, key):
        if key is Ellipsis:
            key = slice(None)
        if isinstance(key, slice):
            if key.step is None or key.step == 1:
                start, stop = key.indices(self._backend.n_frames)[:2]
                return self._backend.iter_continuous_slice(start, stop)
            return self._backend.iter_slice(key)
        key = np.asarray(key)
        if key.ndim == 1:
            return self._backend.iter_frames(key)
        if key.ndim > 1:
            return (np.asarray(tuple(self._get_frames_iter(key_))) for key_ in key)
        return self._backend.get_frame(key)

    def __getitem__(self, key) -> np.ndarray:
        if np.issubdtype(type(key), np.integer):
            return self._backend.get_frame(key)

        if key is None:
            return self[:][None]

        if isinstance(key, tuple):
            if len(key) == 0:
                # a[()] is equivalent to a[:]
                return self[:]

            key0 = key[0]
            key1 = key[1:]

            if np.issubdtype(type(key0), np.integer):
                return self._backend.get_frame(key0)[key1]

            if key0 is None:
                return self[key1][None]

            if key0 is Ellipsis:
                if Ellipsis in key1:
                    raise IndexError("an index can only have a single ellipsis ('...')")
                key1 = (..., *key1)  # need to propage ellipsis

            frames = np.asarray([im[key1] for im in self._get_frames_iter(key0)])
            if frames.shape == (0,):
                # handle empty cases like a[:0, ...]
                key1 = (slice(None), *key1)
                frames = np.empty((0, *self.shape[1:]), dtype=self.dtype)[key1]
            return frames

        # handle slice, list, np.ndarray, etc.
        frames = np.asarray([im for im in self._get_frames_iter(key)])
        if frames.shape == (0,):
            # handle empty cases like a[:0] and a[[]]
            frames = np.empty((0, *self.shape[1:]), dtype=self.dtype)
        return frames

    def __iter__(self):
        return self._backend.iter_all_frames()

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            dtype = self.dtype
        return self[:].astype(dtype)

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

    @cached_property
    def dtype(self) -> np.dtype:
        return self._backend.dtype

    def close(self):
        self._backend.close()
        self._backend = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()
