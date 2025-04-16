import torch
from torch import nn
from pathlib import Path
from collections import OrderedDict
from torchvision.models.resnet import ResNet, resnet50
from subprocess import Popen, PIPE, CalledProcessError

PATH_TO_JACQUES_MODEL_DIRECTORY = "./models/jacques/"
JACQUES_THRESHOLD = 0.306


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