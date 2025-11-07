from pathlib import Path
from abc import ABC, abstractmethod
from ..lib.pipeline import Pipeline


# Abstract class to manage models
class ModelBase(Pipeline, ABC):
    def __init__(self, weights, use_tensorr, batch_size):
        super(ModelBase).__init__()

    @property
    @abstractmethod
    def folder_name(self):
        pass
    
    @property
    def weight_folder(self):
        return Path("./weights", self.folder_name)

    @abstractmethod
    def init_model(self):
        pass


    @abstractmethod
    def setup_new_session(self, session: Path):
        pass

    @abstractmethod
    def generator(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass