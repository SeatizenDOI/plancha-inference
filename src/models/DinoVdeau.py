import json
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import Dinov2Config, Dinov2ForImageClassification, AutoImageProcessor
from ..base.model_base import ModelBase
from ..base.seatizen_tools import join_GPS_metadata
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
import pandas as pd
from PIL import Image
from pathlib import Path
from textwrap import wrap
from tqdm import tqdm



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table
from reportlab.lib.pagesizes import letter, landscape

from PIL import Image
from itertools import compress


from ..lib.tools import sigmoid
from .registry import register_model
from ..base.seatizen_tools import get_cmap, COUNTRY_CODE_FOR_HIGH_ZOOM_LEVEL

try:
    from ..lib.engine_tools import NeuralNetworkGPU, build_and_save_engine_from_onnx
    HAS_TENSORRT = True
except ImportError:
    NeuralNetworkGPU = None
    build_and_save_engine_from_onnx = None
    HAS_TENSORRT = False

@register_model("dinovdeau", default_weights="lombardata/DinoVdeau-large-2024_04_03-with_data_aug_batch-size32_epochs150_freeze")
class DinoVdeau(ModelBase):

    folder_name = "multilabel"

    def __init__(self, weights, use_tensorrt: bool, batch_size: int):
        super().__init__(weights, use_tensorrt, batch_size)
        self.repo_name = weights
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_tensorrt = use_tensorrt and HAS_TENSORRT
        self.init_model()
    

    def init_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.repo_name, use_fast=True)
        self.config = get_dyno_config(self.weight_folder, self.repo_name)
        self.classes_name = list(self.config["label2id"].keys())
        self.threshold = get_threshold(self.weight_folder,  self.repo_name)

        if self.use_tensorrt:
            print("[INFO] Using TensorRT engine for dinov2.")
            self.model = NeuralNetworkGPU(get_multilabel_engine(self.weight_folder, self.repo_name, self.batch_size))
        else:
            print("[INFO] Using standard PyTorch model for dinov2.")
            self.model = NewHeadDinoV2ForImageClassification.from_pretrained(self.repo_name).to(self.device)

    def setup_new_session(self, session: Path):
        self.filename_pred = Path(session, "PROCESSED_DATA/IA", f"{session.name}_{self.repo_name.replace("/", "_")}.csv")
        self.filename_scores = Path(session, "PROCESSED_DATA/IA", f"{session.name}_{self.repo_name.replace("/", "_")}_scores.csv")
        self.csv_connector_classes = open(self.filename_pred, "w")
        self.csv_connector_scores = open(self.filename_scores, "w")

        classe_to_write = ",".join(self.classes_name)
        self.csv_connector_classes.write(f"FileName,{classe_to_write}\n")
        self.csv_connector_scores.write(f"FileName,{classe_to_write}\n")
    
    
    def generator(self):
        """Unified generator dispatch"""
        if self.use_tensorrt:
            base_gen = self._generator_tensorrt()
        else:
            base_gen = self._generator_pytorch()
        
        # Wrap with CSV writing
        yield from self._generator_with_csv(base_gen)

    def _generator_pytorch(self):
        for data in self._data_stream():
            inputs = self.image_processor(data["frames"], return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs)["logits"]
            yield self._postprocess(data, logits)

    def _generator_tensorrt(self):
        for data in self._data_stream():
            inputs = self.image_processor(data["frames"], return_tensors="pt")["pixel_values"]
            outputs = np.split(self.model.detect(np.stack(inputs))[0], self.batch_size)
            yield self._postprocess(data, outputs)

    def _generator_with_csv(self, base_generator):
        """CSV-writing wrapper around another generator."""

        for data in base_generator:
            if "multilabel_labels" in data:
                for i, frame_info in enumerate(data["frames_info"]):
                    frame_name = frame_info.filename
                    labels = ['1' if lab in data["multilabel_labels"][i] else '0' for lab in self.classes_name]
                    self.csv_connector_classes.write(f"{frame_name},{','.join(labels)}\n")
                    self.csv_connector_scores.write(f"{frame_name},{','.join(data['multilabel_scores'][i])}\n")
            yield data


    def _data_stream(self):
        """Abstracts your data source handling."""
        stop = False
        while self.has_next() and not stop:
            try:
                data = next(self.source)
                if data and ("Useless" not in data or 0 in data["Useless"]):
                    yield data
            except StopIteration:
                stop = True


    def _postprocess(self, data, logits):
        """Common postprocessing for both backends."""
        scores_list, labels_list = [], []

        for logit in logits:
            scores = sigmoid(logit if isinstance(logit, np.ndarray) else logit.cpu().numpy())
            scores_list.append([str(s) for s in scores])
            mask = self.applyThreshold(scores)
            labels_list.append(list(compress(self.classes_name, mask)))

        data["multilabel_scores"] = scores_list
        data["multilabel_labels"] = labels_list
        return data


    def cleanup(self):
        self.csv_connector_classes.close()
        self.csv_connector_scores.close()


    def applyThreshold(self, scores: np.ndarray):
        if self.threshold.shape == scores.shape:
            return scores > self.threshold
        else:
            return scores > 0.5
    

    def add_gps_position(self, metadata_path: Path) -> None:
        self.predictions_gps = Path(metadata_path.parent, "predictions_gps.csv")
        self.predictions_scores_gps = Path(metadata_path.parent, "predictions_scores_gps.csv")
        
        join_GPS_metadata(self.filename_pred, metadata_path, self.predictions_gps)
        join_GPS_metadata(self.filename_scores, metadata_path, self.predictions_scores_gps)
    

    def files_generate_by_model(self) -> list[Path]:
        return [self.predictions_gps, self.predictions_scores_gps, self.filename_pred, self.filename_scores]



    def add_pdf_pages(self, prefix: int, pdf_folder_tmp: Path, alpha3_code: int) -> Path:
        """ Create a folder of map for each predictions. """

        df = pd.read_csv(self.predictions_gps)
        if len(df) == 0: return None # No predictions
        if "GPSLongitude" not in df or "GPSLatitude" not in df: return None # No GPS coordinate
        if round(df["GPSLatitude"].std(), 10) == 0.0 or round(df["GPSLongitude"].std(), 10) == 0.0: return None # All frames have the same gps coordinate

        imagery = GoogleTiles(url='https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}')

        # Create temp directory     
        cmap = get_cmap(len(self.classes_name))
        for i, category in tqdm(enumerate(self.classes_name)):
            fig = plt.figure(figsize=(8, 6), dpi=300)
            ax = fig.add_subplot(projection=ccrs.PlateCarree())
            ax.set_extent([df.GPSLongitude.min()-0.0003, df.GPSLongitude.max()+0.0003, df.GPSLatitude.min()-0.0003, df.GPSLatitude.max()+0.0003])
            ax.add_image(imagery, 19 if alpha3_code in COUNTRY_CODE_FOR_HIGH_ZOOM_LEVEL else 17)
            ax.plot(df[df[category] == 1].GPSLongitude, df[df[category] == 1].GPSLatitude, '.', color=cmap(i), markersize=2.5, markeredgewidth=0)
            ax.plot(df[df[category] == 0].GPSLongitude, df[df[category] == 0].GPSLatitude, '.', color='tab:gray', markersize=2.0, markeredgewidth=0)
            ax.set_title(category)
            path_to_save_img = Path(pdf_folder_tmp, f"{prefix}_multiple_page_{category.replace('/', '')}_subplots.jpg")
            plt.savefig(str(path_to_save_img), dpi=300)
            plt.close()


#-------------------------------
# -- TOOLS
#-------------------------------

class NewHeadDinoV2ForImageClassification(Dinov2ForImageClassification):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)

        # Classifier head
        self.classifier = self.create_head(config.hidden_size * 2, config.num_labels)
    
    # CREATE CUSTOM MODEL
    def create_head(self, num_features: int, number_classes: int, dropout_prob: float = 0.5, activation_func = nn.ReLU) -> nn.Sequential:
        features_lst = [num_features , num_features//2 , num_features//4]
        layers = []
        for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
            layers.append(nn.Linear(in_f , out_f))
            layers.append(activation_func())
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob != 0 : layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1] , number_classes))
        return nn.Sequential(*layers)


def get_dyno_config(weight_folder: Path, repo_name: str) -> dict:
    repo_path = Path(weight_folder, repo_name)
    if not repo_path.exists():
        snapshot_download(repo_id=repo_name, local_dir=Path(weight_folder, repo_name))

    config = None
    with open(Path(repo_path, "config.json")) as f:
        config = json.load(f)
    
    return config


def get_threshold(weight_folder: Path, repo_name: str) -> np.ndarray:
    threshold_file = Path(weight_folder, repo_name, "threshold.json")
    threshold = np.array([])
    if threshold_file.exists():
        with open(threshold_file) as f:
            threshold = np.array(list(json.load(f).values()))
    return threshold


def get_multilabel_engine(weight_folder: Path, repo_name: str, batch_size: int) -> Path:
    """ """
    path_to_multilabel_engine = Path(weight_folder, repo_name, f"multilabel_bs_{batch_size}.engine")
    # Check for engine file.
    if path_to_multilabel_engine.exists():
        return str(path_to_multilabel_engine)

    # If engine not found, build model and next build onnx and finally build engine.
    path_to_multilabel_onnx = Path(weight_folder, repo_name, f"multilabel_bs_{batch_size}.onnx")
    if not path_to_multilabel_onnx.exists():
        print("-- Building multilabel onnx file")
        build_onnx_file_for_multilabel(repo_name, path_to_multilabel_onnx, batch_size)

    print("-- Building multilabel engine file")
    build_and_save_engine_from_onnx(path_to_multilabel_onnx, path_to_multilabel_engine)

    return path_to_multilabel_engine


def build_onnx_file_for_multilabel(repo_name: str, path_to_multilabel_onnx: Path, batch_size: int) -> None:
    model = NewHeadDinoV2ForImageClassification.from_pretrained(repo_name)
    image = Image.open(Path("./inputs/image_mutilabel_setup.jpeg"))
    image_processor = AutoImageProcessor.from_pretrained(repo_name)
    inputs = image_processor([image for _ in range(batch_size)], return_tensors="pt")

    torch.onnx.export(
        model,
        tuple(inputs.values()),
        f=path_to_multilabel_onnx,
        input_names=['pixel_values'],
        output_names=['logits'],
        do_constant_folding=True,
    )


