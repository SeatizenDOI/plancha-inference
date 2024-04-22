import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import Dinov2Config, Dinov2ForImageClassification, AutoImageProcessor

from .engine_tools import build_and_save_engine_from_onnx

PATH_TO_MULTILABEL_DIRECTORY = "models/multilabel"

class NewHeadDinoV2ForImageClassification(Dinov2ForImageClassification):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)

        # Classifier head
        self.classifier = self.create_head(config.hidden_size * 2, config.num_labels)
    
    # CREATE CUSTOM MODEL
    def create_head(self, num_features , number_classes ,dropout_prob=0.5 ,activation_func = nn.ReLU):
        features_lst = [num_features , num_features//2 , num_features//4]
        layers = []
        for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
            layers.append(nn.Linear(in_f , out_f))
            layers.append(activation_func())
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob != 0 : layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1] , number_classes))
        return nn.Sequential(*layers)

def getDynoConfig(repo_name):
    repo_path = Path(Path.cwd(), PATH_TO_MULTILABEL_DIRECTORY, repo_name)
    if not Path.exists(repo_path):
        snapshot_download(repo_id=repo_name, local_dir=Path(Path.cwd(), PATH_TO_MULTILABEL_DIRECTORY, repo_name))

    config = None
    with open(Path(repo_path, "config.json")) as f:
        config = json.load(f)
    
    return config

def getThreshold(repo_name):
    threshold_file = Path(Path.cwd(), PATH_TO_MULTILABEL_DIRECTORY, repo_name, "threshold.json")
    threshold = np.array([])
    if Path.exists(threshold_file):
        with open(threshold_file) as f:
            threshold = np.array(list(json.load(f).values()))
    return threshold

def get_multilabel_engine(repo_name, batch_size):
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
    build_and_save_engine_from_onnx(str(path_to_multilabel_onnx), str(path_to_multilabel_engine))

    return str(path_to_multilabel_engine)

def build_onnx_file_for_multilabel(repo_name, path_to_multilabel_onnx, batch_size):
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
        opset_version=13,
    )