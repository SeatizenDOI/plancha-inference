import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import Dinov2Config, Dinov2ForImageClassification, AutoImageProcessor
from ..base.model_base import ModelBase


from PIL import Image
from itertools import compress


from ..lib.tools import sigmoid
from .registry import register_model

try:
    from ..lib.engine_tools import NeuralNetworkGPU, build_and_save_engine_from_onnx
    HAS_TENSORRT = True
except ImportError:
    NeuralNetworkGPU = None
    build_and_save_engine_from_onnx = None
    HAS_TENSORRT = False


PATH_TO_MULTILABEL_DIRECTORY = "models/multilabel"

@register_model("dinovdeau", default_weights="weights/dinovdeau.pt")
class DinoVdeau(ModelBase):

    def __init__(self, use_tensorrt=True, batch_size=8, **kwargs):
        super().__init__(**kwargs)
        self.repo_name = "facebook/dinov2-base"
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_tensorrt = use_tensorrt and HAS_TENSORRT
        self.init_model()

    def init_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.repo_name, use_fast=True)
        self.config = get_dyno_config(self.repo_name)
        self.classes_name = list(self.config["label2id"].keys())
        self.threshold = get_threshold(self.repo_name)

        if self.use_tensorrt:
            print("[INFO] Using TensorRT engine.")
            self.model = NeuralNetworkGPU(get_multilabel_engine(self.repo_name, self.batch_size))
        else:
            print("[INFO] Using standard PyTorch model.")
            self.model = NewHeadDinoV2ForImageClassification.from_pretrained(self.repo_name).to(self.device)

    def setup_new_session(self, session: Path):
        self.filename_pred = Path(session, "PROCESSED_DATA/IA/pred.csv")
        self.filename_scores = Path(session, "PROCESSED_DATA/IA/score.csv")
        self.csv_connector_classes = open(self.filename_pred, "w")
        self.csv_connector_scores = open(self.filename_scores, "w")

        classe_to_write = ",".join(self.classes_name)
        self.csv_connector_classes.write(f"FileName,{classe_to_write}\n")
        self.csv_connector_scores.write(f"FileName,{classe_to_write}\n")
    
    
    def generator(self):
        """Unified generator dispatch"""
        if self.use_tensorrt:
            base_gen = self._generator_tensorrt()
        else:
            base_gen = self._generator_pytorch()
        
        # Wrap with CSV writing
        yield from self._generator_with_csv(base_gen)

    def _generator_pytorch(self):
        for data in self._data_stream():
            inputs = self.image_processor(data["frames"], return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs)["logits"]
            yield self._postprocess(data, logits)

    def _generator_tensorrt(self):
        for data in self._data_stream():
            inputs = self.image_processor(data["frames"], return_tensors="pt")["pixel_values"]
            outputs = np.split(self.model.detect(np.stack(inputs))[0], self.batch_size)
            yield self._postprocess(data, outputs)

    def _generator_with_csv(self, base_generator):
        """CSV-writing wrapper around another generator."""
        header = ",".join(self.classes_name)
        self.csv_connector_classes.write(f"FileName,{header}\n")
        self.csv_connector_scores.write(f"FileName,{header}\n")

        for data in base_generator:
            if "multilabel_labels" in data:
                for i, frame_info in enumerate(data["frames_info"]):
                    frame_name = frame_info.filename
                    labels = ['1' if lab in data["multilabel_labels"][i] else '0' for lab in self.classes_name]
                    self.csv_connector_classes.write(f"{frame_name},{','.join(labels)}\n")
                    self.csv_connector_scores.write(f"{frame_name},{','.join(data['multilabel_scores'][i])}\n")
            yield data


    # ---------------------------------------------------------
    # DATA FLOW HELPERS
    # ---------------------------------------------------------
    def _data_stream(self):
        """Abstracts your data source handling."""
        stop = False
        while self.has_next() and not stop:
            try:
                data = next(self.source)
                if data and ("Useless" not in data or 0 in data["Useless"]):
                    yield data
            except StopIteration:
                stop = True

    def _postprocess(self, data, logits):
        """Common postprocessing for both backends."""
        scores_list, labels_list = [], []

        for logit in logits:
            scores = sigmoid(logit if isinstance(logit, np.ndarray) else logit.cpu().numpy())
            scores_list.append([str(s) for s in scores])
            mask = self.applyThreshold(scores)
            labels_list.append(list(compress(self.classes_name, mask)))

        data["multilabel_scores"] = scores_list
        data["multilabel_labels"] = labels_list
        return data

    def cleanup(self):
        self.csv_connector_classes.close()
        self.csv_connector_scores.close()

    # Method specific to Dinovdeau
    def applyThreshold(self, scores: np.ndarray):
        if self.threshold.shape == scores.shape:
            return scores > self.threshold
        else:
            return scores > 0.5
        

class NewHeadDinoV2ForImageClassification(Dinov2ForImageClassification):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)

        # Classifier head
        self.classifier = self.create_head(config.hidden_size * 2, config.num_labels)
    
    # CREATE CUSTOM MODEL
    def create_head(self, num_features: int, number_classes: int, dropout_prob: float = 0.5, activation_func = nn.ReLU) -> nn.Sequential:
        features_lst = [num_features , num_features//2 , num_features//4]
        layers = []
        for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
            layers.append(nn.Linear(in_f , out_f))
            layers.append(activation_func())
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob != 0 : layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1] , number_classes))
        return nn.Sequential(*layers)


def get_dyno_config(repo_name: str) -> dict:
    repo_path = Path(Path.cwd(), PATH_TO_MULTILABEL_DIRECTORY, repo_name)
    if not Path.exists(repo_path):
        snapshot_download(repo_id=repo_name, local_dir=Path(Path.cwd(), PATH_TO_MULTILABEL_DIRECTORY, repo_name))

    config = None
    with open(Path(repo_path, "config.json")) as f:
        config = json.load(f)
    
    return config


def get_threshold(repo_name: str) -> np.ndarray:
    threshold_file = Path(Path.cwd(), PATH_TO_MULTILABEL_DIRECTORY, repo_name, "threshold.json")
    threshold = np.array([])
    if Path.exists(threshold_file):
        with open(threshold_file) as f:
            threshold = np.array(list(json.load(f).values()))
    return threshold


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


