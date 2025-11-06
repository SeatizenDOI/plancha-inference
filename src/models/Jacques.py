import torch
import pandas as pd
from pathlib import Path

from ..lib.tools import get_image_transformation
from ..base.model_base import ModelBase

import torch
from torch import nn
from pathlib import Path
from collections import OrderedDict
from torchvision.models.resnet import ResNet, resnet50
from subprocess import Popen, PIPE, CalledProcessError

try:
    from ..lib.engine_tools import NeuralNetworkGPU, build_and_save_engine_from_onnx
    HAS_TENSORRT = True
except ImportError:
    NeuralNetworkGPU = None
    build_and_save_engine_from_onnx = None
    HAS_TENSORRT = False


PATH_TO_JACQUES_MODEL_DIRECTORY = "./models/jacques/"
JACQUES_THRESHOLD = 0.306


class Jacques(ModelBase):
    """Pipeline for jacques predictor. Jacques sort image in useless/useful classes"""

    def __init__(self, checkpoint: str, batch_size: int):
        super(JacquesPredictor).__init__()
        self.model = build_jacques_model(checkpoint)
        self.transform = get_image_transformation()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
    
    def __init__(self, use_tensorrt=True, batch_size=8, **kwargs):
        pass
    
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

import torch
import numpy as np
from pathlib import Path
redictor(Pipeline):
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

class JacquesPredictions(Pipeline):
    """Pipeline task to save Jacques predictions"""

    def __init__(self):
        self.filename, self.csv_connector = None, None
        super(JacquesPredictions, self).__init__()

    def setup(self, filename: Path):
        self.filename = filename
        self.csv_connector = open(self.filename, "w")

    def generator(self):
        """ Write in csv file"""
        self.csv_connector.write("FileName,Useless,Score\n")

        data, stop = None, False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream.
                data = next(self.source)
            except StopIteration:
                stop = True

            if not stop and data:
                for i, frame_info in enumerate(data["frames_info"]):
                    frame_name = frame_info.filename
                    self.csv_connector.write(f"{frame_name},{data['Useless'][i]},{data['prob_jacques'][i]}\n")
            
                yield data
    
    def cleanup(self):
        self.csv_connector.close()


class HeadNet():
    def __init__(self, bodynet_features_out: int):

        self.bodynet_features_out = bodynet_features_out
        self.nb_hidden_head_layers = 2
        self.nb_classes = 1 # Class named Useless.

    def create_one_hidden_head_layer(self, nb_features_in: int, nb_features_out: int) -> nn.Sequential:
        hidden_head_layer = [
            nn.Linear(nb_features_in, nb_features_out),
            nn.Sigmoid(),
            nn.BatchNorm1d(nb_features_out),
            nn.Dropout(0.5)
        ]
        return nn.Sequential(*hidden_head_layer)

    def create_output_layer(self, nb_features_last_hidden_layer: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(nb_features_last_hidden_layer, self.nb_classes))

    def create_head_layers(self) -> nn.Sequential:
        nb_features_headnet_hidden_layer_in = self.bodynet_features_out
        nb_features_hidden_layer_out = self.bodynet_features_out//2
        head_layers = []
        for _ in range(self.nb_hidden_head_layers):
            hidden_head_layer = self.create_one_hidden_head_layer(nb_features_headnet_hidden_layer_in, nb_features_hidden_layer_out)
            nb_features_headnet_hidden_layer_in //= 2
            nb_features_hidden_layer_out //= 2
            head_layers.append(hidden_head_layer)
        output_layer = self.create_output_layer(nb_features_hidden_layer_out*2)
        head_layers.append(output_layer)
        return nn.Sequential(*head_layers)


def build_jacques_model(checkpoint: str) -> ResNet:
    backbone = resnet50(weights='ResNet50_Weights.DEFAULT')
    headnet = HeadNet(bodynet_features_out = backbone.fc.in_features)

    # Init a pretrained model & freeze the backbone parameters with requires_grad = False.
    for param in backbone.parameters():
        param.requires_grad = False
        
    # Replace the fc layer by head layers.
    backbone.fc = headnet.create_head_layers()

    # Load last checkpoint.
    model = load_checkpoint(backbone, checkpoint)

    return model

def load_checkpoint(model: ResNet, checkpoint_name: str) -> ResNet:
    # Create path to get jacques model.
    folder_checkpoint = Path(Path.cwd(), PATH_TO_JACQUES_MODEL_DIRECTORY, checkpoint_name.replace("/", "_"))

    # If we don't have download the checkpoint go ahead.    
    if not folder_checkpoint.exists() or not folder_checkpoint.is_dir():
        download_checkpoint(checkpoint_name, folder_checkpoint)

    checkpoint_loaded = torch.load(Path(folder_checkpoint, "epoch.pth"), map_location='cuda:0' if torch.cuda.is_available() else 'cpu', weights_only=True)
    
    new_state_dict = OrderedDict()
    for k, v in checkpoint_loaded['state_dict'].items():
        name = k[6:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    return model


def download_checkpoint(checkpoint_name: str, folder_checkpoint: Path):
    print("\n-- Download checkpoint for jacques model")

    with Popen(["zenodo_get", "-o", str(folder_checkpoint), checkpoint_name], stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')

        p.wait() # Wait because sometimes Python is too fast.
        if p.returncode != 0:
            raise CalledProcessError(p.returncode, p.args)
    
    # Rename file to epoch.pth
    for file in folder_checkpoint.iterdir():
        if file.suffix.lower() == ".pth":
            file.rename(Path(file.parent, "epoch.pth"))
            break
