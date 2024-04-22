import torch
import numpy as np
from itertools import compress
from transformers import AutoImageProcessor

from .pipeline import Pipeline
from .libs.tools import sigmoid
from .libs.engine_tools import NeuralNetworkGPU
from .libs.multilabel_model import NewHeadDinoV2ForImageClassification, getDynoConfig, getThreshold, get_multilabel_engine


class MultiLabelClassifier(Pipeline):
    """Pipeline to identify mulitple class in image"""
    def __init__(self, repo_name, batch_size):
        super(MultiLabelClassifier).__init__()

        self.image_processor = AutoImageProcessor.from_pretrained(repo_name)
        self.config = getDynoConfig(repo_name)
        self.classes_name = list(self.config["label2id"].keys())
        self.threshold = getThreshold(repo_name)
        self.batch_size = batch_size

    def applyThreshold(self, scores):
        if self.threshold.shape == scores.shape:
            return scores > self.threshold
        else:
            return scores > 0.5

    def cleanup(self):
        """ nothing to release """
        pass

class MultiLabelClassifierCUDA(MultiLabelClassifier):
    """Multilabel classifier with cuda"""

    def __init__(self, repo_name, batch_size):
        super().__init__(repo_name, batch_size)

        self.model = NewHeadDinoV2ForImageClassification.from_pretrained(repo_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    def generator(self):
        # Pass model to gpu
        self.model = self.model.to(self.device)

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
                    inputs = self.image_processor(data["frames"], return_tensors="pt")
                    inputs = inputs.to(self.device)
                    with torch.no_grad():
                        model_outputs = self.model(**inputs)
                    
                    data["multilabel_scores"], data["multilabel_labels"] = [], []
                    for logit in model_outputs["logits"]:
                        scores = sigmoid(logit.cpu().numpy())
                    
                        data["multilabel_scores"].append([str(s) for s in scores])

                        # Apply threshold. 
                        scores = self.applyThreshold(scores)

                        # Send predicted label
                        predicted_label = list(compress(self.classes_name,  scores))
                        data["multilabel_labels"].append(predicted_label)
                
                yield data

class MultiLabelClassifierTRT(MultiLabelClassifier):
    """Multilabel classifier with TensorRt"""

    def __init__(self, repo_name, batch_size):
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
    