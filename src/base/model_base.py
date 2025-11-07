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


    # Optional method
    def add_gps_position(self, metadata_path: Path):
        """Optional: subclasses can override this to add GPS info."""
        raise NotImplementedError("add_gps_position not implemented")

    # Generic checker
    @classmethod
    def has_method(cls, method_name: str) -> bool:
        """
        Check whether `method_name` is implemented in the subclass (not inherited from BaseModel).
        """
        method = cls.__dict__.get(method_name, None)
        base_method = getattr(ModelBase, method_name, None)
        return callable(method) and method is not base_method

    # Optional syntactic sugar
    def need(self, method_name: str) -> bool:
        """Instance-level version."""
        return self.__class__.has_method(method_name)


    def __repr__(self):
        return f"{self.folder_name}, with_gps_position: {self.has_method('add_gps_position')}"