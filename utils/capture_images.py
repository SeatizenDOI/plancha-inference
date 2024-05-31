from PIL import Image
from natsort import natsorted

from .pipeline import Pipeline
from .libs.load_images import load_frames_from_source

class CaptureImages(Pipeline):
    """Pipeline task to extract image from source"""

    def __init__(self, batch_size):
        super(CaptureImages).__init__()
        self.frames_path = []
        self.frame_count = len(self.frames_path)
        self.frame_id = None
        self.batch_size = batch_size
    
    def setup(self, src, mode):
        """ Reset image loaded """
        self.frames_path = natsorted(load_frames_from_source(src, mode))
        self.frame_count = len(self.frames_path)
        self.frame_id = 0

    def generator(self):
        
        while self.frame_id + self.batch_size <= self.frame_count:
            frame_paths = [self.frames_path[id] for id in range(self.frame_id, self.frame_id + self.batch_size)]
            try:
                data = {
                    "frames": [Image.open(frame_path) for frame_path in frame_paths],
                    "frame_paths": frame_paths
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