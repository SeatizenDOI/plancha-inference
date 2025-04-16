import torch
from torch import nn
from pathlib import Path
from collections import OrderedDict
import torchvision.models as models
from subprocess import Popen, PIPE, CalledProcessError

from .engine_tools import build_and_save_engine_from_onnx

PATH_TO_JACQUES_MODEL_DIRECTORY = "./models/jacques/"

class HeadNet():
    def __init__(self, bodynet_features_out):

        self.bodynet_features_out = bodynet_features_out
        self.nb_hidden_head_layers = 2
        self.nb_classes = 1 # Class named Useless.

    def create_one_hidden_head_layer(self, nb_features_in, nb_features_out):
        hidden_head_layer = [
            nn.Linear(nb_features_in, nb_features_out),
            nn.Sigmoid(),
            nn.BatchNorm1d(nb_features_out),
            nn.Dropout(0.5)
        ]
        return nn.Sequential(*hidden_head_layer)

    def create_output_layer(self, nb_features_last_hidden_layer):
        return nn.Sequential(nn.Linear(nb_features_last_hidden_layer, self.nb_classes))

    def create_head_layers(self):
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

def build_jacques_model(checkpoint):
    backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    headnet = HeadNet(bodynet_features_out = backbone.fc.in_features)

    # Init a pretrained model & freeze the backbone parameters with requires_grad = False.
    for param in backbone.parameters():
        param.requires_grad = False
        
    # Replace the fc layer by head layers.
    backbone.fc = headnet.create_head_layers()

    # Load last checkpoint.
    model = load_checkpoint(backbone, checkpoint)

    return model

def load_checkpoint(model, checkpoint_name):
    # Create path to get jacques model.
    folder_checkpoint = Path(Path.cwd(), PATH_TO_JACQUES_MODEL_DIRECTORY, checkpoint_name.replace("/", "_"))

    # If we don't have download the checkpoint go ahead.    
    if not Path.exists(folder_checkpoint) or not Path.is_dir(folder_checkpoint):
        download_checkpoint(checkpoint_name, folder_checkpoint)

    checkpoint_loaded = torch.load(Path(folder_checkpoint, "epoch"), map_location='cuda:0' if torch.cuda.is_available() else 'cpu', weights_only=True)
    
    new_state_dict = OrderedDict()
    for k, v in checkpoint_loaded['state_dict'].items():
        name = k[6:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    return model

def download_checkpoint(checkpoint_name, folder_checkpoint: Path):
    print("\n-- Download checkpoint for jacques model")

    with Popen(["zenodo_get", "-o", str(folder_checkpoint), checkpoint_name], stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')

        p.wait() # Wait because sometimes Python is too fast.
        if p.returncode != 0:
            raise CalledProcessError(p.returncode, p.args)

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