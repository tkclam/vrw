from .backend import Backend
import av


class PyAvBackend(Backend):
    def __init__(
        self,
        filename: str,
        fps: float,
        codec="libx264",
        crf=23,
        preset="slow",
        verbose=0,
        **kwargs,
    ):
        super().__init__(filename=filename, fps=fps)
        kwargs["crf"] = str(crf)
        kwargs["preset"] = preset
        self._codec = codec
        if not verbose and codec == "libx265":
            kwargs["x265-params"] = "log-level=none"
        self._kwargs = kwargs
        self._container = None
        self._stream = None

    def _init(self, frame_shape):
        assert len(frame_shape) == 2
        height, width = frame_shape[:2]
        self._container = av.open(self._filename, "w")
        self._stream = self._container.add_stream(self._codec, rate=self._fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "gray"
        self._stream.options = self._kwargs
        if self._codec == "libx265":
            self._stream.codec_context.codec_tag = "hvc1"

    def write(self, frame):
        if self._container is None or self._container is None:
            self._init(frame.shape)

        im = av.VideoFrame.from_ndarray(frame, format="gray")
        for packet in self._stream.encode(im):
            self._container.mux(packet)
        self._n_frames += 1

    def close(self):
        if self._container and self._stream:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
            self._container = None
            self._stream = None
