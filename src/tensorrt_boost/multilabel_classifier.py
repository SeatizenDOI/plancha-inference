import torch
import numpy as np
from PIL import Image
from pathlib import Path
from itertools import compress
from transformers import AutoImageProcessor

from ..base.tools import sigmoid
from ..base.multilabel_model_base import MultiLabelClassifierBase, PATH_TO_MULTILABEL_DIRECTORY, NewHeadDinoV2ForImageClassification

from .engine_tools import NeuralNetworkGPU, build_and_save_engine_from_onnx


class MultiLabelClassifier(MultiLabelClassifierBase):
    """Multilabel classifier with TensorRt"""

    def __init__(self, repo_name: str, batch_size: int):
        super().__init__(repo_name, batch_size)

        self.model = NeuralNetworkGPU(get_multilabel_engine(repo_name, batch_size))

    def generator(self):

        data = None
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream.
                data = next(self.source)
            except StopIteration:
                stop = True

            if not stop and data:
                # Check if image is not useless
                if "Useless" not in data or 0 in data["Useless"]:
                    inputs = self.image_processor(data["frames"], return_tensors="pt")["pixel_values"]
                    outputs = np.split(self.model.detect(np.stack(inputs))[0], self.batch_size)

                    predicted_labels, scores = [], []
                    for output in outputs:
                        score = sigmoid(output)
                        scores.append([str(s) for s in score]) # Save score
    
                        # Save predicted label
                        score = self.applyThreshold(score)  
                        predicted_labels.append(list(compress(self.classes_name, score)))

                    data["multilabel_scores"] = scores
                    data["multilabel_labels"] = predicted_labels
                
                yield data
    

def get_multilabel_engine(repo_name: str, batch_size: int) -> Path:
    """ """
    path_to_multilabel_engine = Path(Path.cwd(), PATH_TO_MULTILABEL_DIRECTORY, repo_name, f"multilabel_bs_{batch_size}.engine")
    # Check for engine file.
    if Path.exists(path_to_multilabel_engine):
        return str(path_to_multilabel_engine)

    # If engine not found, build model and next build onnx and finally build engine.
    path_to_multilabel_onnx = Path(Path.cwd(), PATH_TO_MULTILABEL_DIRECTORY, repo_name, f"multilabel_bs_{batch_size}.onnx")
    if not Path.exists(path_to_multilabel_onnx):
        print("-- Building multilabel onnx file")
        build_onnx_file_for_multilabel(repo_name, path_to_multilabel_onnx, batch_size)

    print("-- Building multilabel engine file")
    build_and_save_engine_from_onnx(path_to_multilabel_onnx, path_to_multilabel_engine)

    return path_to_multilabel_engine


def build_onnx_file_for_multilabel(repo_name: str, path_to_multilabel_onnx: Path, batch_size: int) -> None:
    model = NewHeadDinoV2ForImageClassification.from_pretrained(repo_name)
    image = Image.open(Path(Path.cwd(), "inputs/image_mutilabel_setup.jpeg"))
    image_processor = AutoImageProcessor.from_pretrained(repo_name)
    inputs = image_processor([image for _ in range(batch_size)], return_tensors="pt")

    torch.onnx.export(
        model,
        tuple(inputs.values()),
        f=path_to_multilabel_onnx,
        input_names=['pixel_values'],
        output_names=['logits'],
        do_constant_folding=True,
    )