from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Iterable

from PIL import Image
from natsort import natsorted

from ..lib.pipeline import Pipeline
from ..lib.parse_opt import Sources
from ..lib.load_images import FrameInformation, load_frames_from_source


def _load_rgb_image(frame_path: Path) -> Image.Image:
    """Open and decode one frame as RGB (pixel load happens here, not later)."""
    with Image.open(frame_path) as img:
        return img.convert("RGB")


def _load_rgb_batch(frames_info: list[FrameInformation]) -> list[Image.Image]:
    """Decode one CaptureImages batch. Raises on I/O or decode errors."""
    return [_load_rgb_image(frame_info.frame_path) for frame_info in frames_info]


class CaptureImages(Pipeline):
    """Pipeline task to extract image from source"""

    def __init__(self, batch_size: int, prefetch_workers: int = 4):
        super(CaptureImages).__init__()
        self.frames_information = []
        self.frame_count = len(self.frames_information)
        self.frame_id = None
        self.batch_size = batch_size
        self.prefetch_workers = max(1, int(prefetch_workers))
    
    def setup_new_session(self, src: Path, mode: Sources):
        """ Reset image loaded """
        self.frames_information = natsorted(load_frames_from_source(src, mode))
        self.frame_count = len(self.frames_information)
        self.frame_id = 0

    def generator(self):
        if self.prefetch_workers <= 1:
            yield from self._generator_sequential()
        else:
            yield from self._generator_prefetch()

        yield None # Need to yield None to avoid an extra turn

    def _batch_frames_info(self) -> Iterable[list[FrameInformation]]:
        """Yield frame-info batches; same leftover drop as before when n % bs != 0."""
        while self.frame_id + self.batch_size <= self.frame_count:
            frames_info = [
                self.frames_information[id]
                for id in range(self.frame_id, self.frame_id + self.batch_size)
            ]
            self.frame_id = self.frame_id + self.batch_size
            yield frames_info

    def _generator_sequential(self):
        """Original single-threaded path (workers=1): decode on the pipeline thread."""
        for frames_info in self._batch_frames_info():
            try:
                data = {
                    "frames": _load_rgb_batch(frames_info),
                    "frames_info": frames_info,
                }

                if self.filter(data):
                    yield self.map(data)

            except StopIteration:
                return

    def _generator_prefetch(self):
        """Decode upcoming batches on a thread pool while the GPU pipeline runs."""
        max_inflight = self.prefetch_workers
        batch_iter = iter(self._batch_frames_info())
        pending: deque[tuple[Future, list[FrameInformation]]] = deque()

        def fill() -> None:
            while len(pending) < max_inflight:
                try:
                    frames_info = next(batch_iter)
                except StopIteration:
                    return
                future = pool.submit(_load_rgb_batch, frames_info)
                pending.append((future, frames_info))

        with ThreadPoolExecutor(max_workers=self.prefetch_workers) as pool:
            fill()
            while pending:
                future, frames_info = pending.popleft()
                try:
                    frames = future.result()
                except Exception:
                    for fut, _ in pending:
                        fut.cancel()
                    raise

                data = {
                    "frames": frames,
                    "frames_info": frames_info,
                }
                if self.filter(data):
                    yield self.map(data)
                fill()
    
    def cleanup(self):
        """ nothing to release """
        pass
