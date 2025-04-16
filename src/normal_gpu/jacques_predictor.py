import torch
import pandas as pd
from pathlib import Path

from ..base.pipeline import Pipeline
from ..base.jacques_model_base import JACQUES_THRESHOLD, build_jacques_model
from ..base.tools import get_image_transformation


class JacquesPredictor(Pipeline):
    """Pipeline for jacques predictor. Jacques sort image in useless/useful classes"""

    def __init__(self, checkpoint: str, batch_size: int):
        super(JacquesPredictor).__init__()
        self.model = build_jacques_model(checkpoint)
        self.transform = get_image_transformation()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
    
    def generator(self):
        """Yields the image enriched with jacques predictions"""
        self.model.eval() # Put model in inference mode

        data = None
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream.
                data = next(self.source)
            except StopIteration:
                stop = True

            if not stop and data:
                data["Useless"], data["prob_jacques"] = [], []
                # Pass image to GPU
                for frame in data["frames"]:
                    image = self.transform(frame)[None, :]
                    image.to(self.device) 

                    with torch.no_grad():
                        logits = self.model(image)
                        prob = torch.sigmoid(logits).data
                        prob_round = round(prob.cpu().numpy()[0][0], 3)
                        data["Useless"].append(1 if prob_round > JACQUES_THRESHOLD else 0)
                        data["prob_jacques"].append(prob_round)

                yield data

    def cleanup(self):
        """ nothing to release """
        pass


class JacquesCSV(Pipeline):
    """Pipeline for jacques csv file. Used csv file to get annotate"""

    def __init__(self):
        super(JacquesCSV).__init__()
        self.filename, self.df = None, None       
    
    def setup(self, filename: Path):
        self.filename = filename
        if not self.filename.exists():
            raise "Jacques csv file doesn't exist"
        self.df = pd.read_csv(self.filename)

    def generator(self):
        """Yields the image enriched with jacques predictions"""

        data = None
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream.
                data = next(self.source)
            except StopIteration:
                stop = True

            if not stop and data:
                data["Useless"] = []
                for frame in data["frame_paths"]:
                    row = self.df[self.df["FileName"] == frame.name]
                    if len(row) != 0:
                        data["Useless"].append(row["Useless"].iloc[0])
                        
                yield data
    
    def cleanup(self):
        """ nothing to release """
        pass