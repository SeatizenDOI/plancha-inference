import torch
import numpy as np
import pandas as pd
from pathlib import Path

from .pipeline import Pipeline
from .libs.engine_tools import NeuralNetworkGPU
from .libs.tools import sigmoid, get_image_transformation
from .libs.jacques_model import build_jacques_model, get_jacques_engine_name

JACQUES_THRESHOLD = 0.306

class JacquesPredictor(Pipeline):
    """Pipeline for jacques predictor. Jacques sort image in useless/useful classes"""

    def __init__(self, checkpoint, batch_size):
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

class JacquesPredictorGPU(Pipeline):
    """Pipeline for jacques predictor. Jacques sort image in useless/use classes"""

    def __init__(self, checkpoint, batch_size):
        super(JacquesPredictorGPU).__init__()
        
        self.batch_size = batch_size
        self.model = NeuralNetworkGPU(get_jacques_engine_name(checkpoint, batch_size))
        self.transform = get_image_transformation()
    
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
                images = [(self.transform(frame)[None, :]).numpy() for frame in data["frames"]]

                outputs = np.split(self.model.detect(np.stack(images))[0], self.batch_size)
                data["Useless"], data["prob_jacques"] = [], []
                for output in outputs:
                    prob = sigmoid(output)[0]
                    data["Useless"].append(1 if prob > JACQUES_THRESHOLD else 0)
                    data["prob_jacques"].append(prob)

            yield data

    def cleanup(self):
        """ nothing to release """
        pass

class JacquesCSV(Pipeline):
    """Pipeline for jacques csv file. Used csv file to get annotate"""

    def __init__(self):
        super(JacquesCSV).__init__()
        self.filename, self.df = None, None       
    
    def setup(self, filename):
        self.filename = Path(filename)
        if not Path.exists(self.filename):
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