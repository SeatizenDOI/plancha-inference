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
        raise NotImplementedError("Please implement me")

    @abstractmethod
    def setup_new_session(self, session: Path):
        raise NotImplementedError("Please implement me")

    @abstractmethod
    def generator(self):
        raise NotImplementedError("Please implement me")

    @abstractmethod
    def cleanup(self):
        raise NotImplementedError("Please implement me")
    
    @abstractmethod
    def files_generate_by_model():
        raise NotImplementedError("Please implement me")


    # Optional method
    def add_gps_position(self, metadata_path: Path):
        """Optional: subclasses can override this to add GPS info."""
        raise NotImplementedError("add_gps_position not implemented")

    def add_pdf_pages(self):
        """Optional: subclasses can override this to add pdf pages."""
        raise NotImplementedError("add_pdf_pages not implemented")


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