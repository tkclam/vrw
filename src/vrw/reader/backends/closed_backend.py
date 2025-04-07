from .backend import Backend


class ClosedBackend(Backend):
    def __init__(self):
        super().__init__(path="", to_gray=False)

    def get_frame(self, frame_id: int):
        raise ValueError("Cannot get frame: backend is closed.")

    @property
    def n_frames(self) -> int:
        raise ValueError("Cannot get number of frames: backend is closed.")

    @property
    def fps(self) -> float:
        raise ValueError("Cannot get fps: backend is closed.")

    @property
    def width(self) -> int:
        raise ValueError("Cannot get width: backend is closed.")

    @property
    def height(self) -> int:
        raise ValueError("Cannot get height: backend is closed.")

    @property
    def shape(self) -> tuple[int, int, int] | tuple[int, int, int, int]:
        raise ValueError("Cannot get shape: backend is closed.")
