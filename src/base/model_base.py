import json
import numpy as np
import torch.nn as nn
from pathlib import Path

from .pipeline import Pipeline

PATH_TO_MULTILABEL_DIRECTORY = "models/multilabel"


class ModelBase(Pipeline):
    """Pipeline to identify mulitple class in image"""
    def __init__(self, repo_name: str, batch_size: int):
        super(ModelBase).__init__()


    def init_model():
        pass
    

    def cleanup():
        pass

