# vrw: Library for simple video reading and writing

## Usage
### Reading videos
```python
from vrw import VideoReader

vr = VideoReader("video.mp4")
print(vr.shape) # (N, H, W, C), N frames, H height, W width, C channels
print(vr[:, 0].shape) # (N, W, C), all frames, first row
print(vr[::2, :, -1].shape) # (N // 2, H, C), every second frame, last column
print(vr[..., 0].shape) # (N, H, W), all frames, first channel
print(vr[[1, 3, 3, 7], :42].shape) # (4, 42, W, C)

# reading grayscale frames
vr_gray = VideoReader("video.mp4", to_gray=True)
print(vr_gray.shape) # (N, H, W)
```

### Writing videos
```python
from vrw import VideoWriter

# writing rgb frames
with VideoWriter("video.mp4", fps=30) as vw:
    for i in range(256):
        im = np.full((256, 256, 3), i, dtype=np.uint8)
        im[..., 0], im[..., 1] = np.mgrid[:256, :256]
        vw.write(im)

# writing grayscale frames
with VideoWriter("video.mp4", fps=30) as vw:
    for i in range(256):
        vw.write(np.full((256, 256), i, dtype=np.uint8))

# use pyav backend (requires ffmpeg built with the codec you want to use)
# instead of opencv (default). 
with VideoWriter("video.mp4", fps=30, backend="pyav", codec="libx264") as vw:
    for i in range(256):
        im = np.full((256, 256, 3), i, dtype=np.uint8)
        im[..., 0], im[..., 1] = np.mgrid[:256, :256]
        vw.write(im)

```
