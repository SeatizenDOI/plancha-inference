from pathlib import Path
from .model_base import ModelBase
from ..models.registry import MODEL_REGISTRY

class ModelsManager:

    def __init__(self, model_names: list[str], weight_paths: list | None, use_tensorrt: bool, batch_size: int):
        
        self.models: list[ModelBase] = []

        self.load_models(model_names, weight_paths, use_tensorrt, batch_size)
    
    def load_models(self, model_names: list[str], weight_paths: list | None, use_tensorrt: bool, batch_size: int) -> list[ModelBase]:

        for i, name in enumerate(model_names):
            model_info = MODEL_REGISTRY[name]
            ModelClass = model_info["class"]

            # choose user-provided weight or default
            weights = (
                weight_paths[i]
                if weight_paths and i < len(weight_paths)
                else model_info["default_weights"]
            )

            print(f"→ Loading {name} with weights: {weights}")
            model = ModelClass(weights, use_tensorrt, batch_size)
            self.models.append(model)


    def setup_new_session(self, session: Path) -> None:
        for model in self.models:
            model.setup_new_session(session)

    def add_pdf_pages(self, pdf_folder_tmp: Path, alpha3_code: int):
        
        for i, model in enumerate(self.models):
            model.add_pdf_pages(i, pdf_folder_tmp, alpha3_code)

    def add_gps_position(self, metadata_path: Path) -> None:
        for model in self.models:
            if model.need("add_gps_position"):
                model.add_gps_position(metadata_path)


    def cleanup(self) -> None:
        for model in self.models:
            model.cleanup()
    
    def files_generate_by_model(self) -> list[Path]:
        files = []
        for model in self.models:
            files += model.files_generate_by_model()
        
        return files


    def __repr__(self):
        return f"Model Manager: {self.models}"
    
    def __ror__(self, source):
        """Allow chaining with | operator."""
        self.source = source
        return self

    def __iter__(self):
        """Iterate through all models sequentially."""
        iterator = iter(self.source)
        for model in self.models:
            model.source = iterator
            iterator = model.generator()
        return iterator
