import torch
import numpy as np
from pathlib import Path

from ..base.pipeline import Pipeline
from ..base.tools import get_image_transformation, sigmoid
from ..base.jacques_model_base import JACQUES_THRESHOLD, PATH_TO_JACQUES_MODEL_DIRECTORY, build_jacques_model

from .engine_tools import NeuralNetworkGPU, build_and_save_engine_from_onnx

class JacquesPredictor(Pipeline):
    """Pipeline for jacques predictor. Jacques sort image in useless/use classes"""

    def __init__(self, checkpoint: str, batch_size: int):
        super(JacquesPredictor).__init__()
        
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


def get_jacques_engine_name(checkpoint, batch_size):
    path_to_jacques_engine = Path(Path.cwd(), PATH_TO_JACQUES_MODEL_DIRECTORY, checkpoint.replace("/", "_"), f"jacques_bs_{batch_size}.engine")
    if Path.exists(path_to_jacques_engine):
        return str(path_to_jacques_engine)

    # Build Jacques model.
    model = build_jacques_model(checkpoint)
    model.eval()

    path_to_jacques_onnx = Path(Path.cwd(), PATH_TO_JACQUES_MODEL_DIRECTORY, checkpoint.replace("/", "_"), f"jacques_bs_{batch_size}.onnx")
    # Convert to onnx format if needed.
    if not Path.exists(path_to_jacques_onnx):
        print("-- Building jacques onnx file")
        build_jacques_onnx_file(model, path_to_jacques_onnx, batch_size)
    
    # Build engine from onnx file.
    print("-- Building jacques engine file")
    build_and_save_engine_from_onnx(str(path_to_jacques_onnx), str(path_to_jacques_engine))
    
    # Return loaded engine.
    return str(path_to_jacques_engine)

def build_jacques_onnx_file(model, path_to_onnx, batch_size):
    torch_input = torch.randn(batch_size, 3, 224, 224) #! FIXME Fixed value
    torch.onnx.export(model, torch_input, path_to_onnx)