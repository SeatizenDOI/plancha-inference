import json
import numpy as np
import torch.nn as nn
from pathlib import Path
from abc import ABC, abstractmethod
from ..lib.pipeline import Pipeline


# Abstract class to manage models
class ModelBase(Pipeline, ABC):
    """Pipeline to identify mulitple class in image"""
    def __init__(self):
        super(ModelBase).__init__()


    @abstractmethod
    def init_model(self):
        """Initialize the model (to be implemented by subclasses)."""
        pass


    @abstractmethod
    def setup_new_session(self, session: Path):
        """Initialize the model (to be implemented by subclasses)."""
        pass

    @abstractmethod
    def generator(self):
        """Initialize the model (to be implemented by subclasses)."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources (to be implemented by subclasses)."""
        pass