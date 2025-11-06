from PIL import Image
from pathlib import Path
from natsort import natsorted

from ..lib.pipeline import Pipeline
from ..lib.parse_opt import Sources
from ..lib.load_images import load_frames_from_source

class CaptureImages(Pipeline):
    """Pipeline task to extract image from source"""

    def __init__(self, batch_size: int):
        super(CaptureImages).__init__()
        self.frames_information = []
        self.frame_count = len(self.frames_information)
        self.frame_id = None
        self.batch_size = batch_size
    
    def setup(self, src: Path, mode: Sources):
        """ Reset image loaded """
        self.frames_information = natsorted(load_frames_from_source(src, mode))
        self.frame_count = len(self.frames_information)
        self.frame_id = 0

    def generator(self):
        while self.frame_id + self.batch_size <= self.frame_count:
            frames_info = [self.frames_information[id] for id in range(self.frame_id, self.frame_id + self.batch_size)]
            try:
                data = {
                    "frames": [Image.open(frame_info.frame_path) for frame_info in frames_info],
                    "frames_info": frames_info
                }

                if self.filter(data):
                    self.frame_id = self.frame_id + self.batch_size
                    yield self.map(data)

            except StopIteration:
                return
        
        yield None # Need to yield None to avoid an extra turn 
    
    def cleanup(self):
        """ nothing to release """
        pass