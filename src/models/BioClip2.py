import json
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from open_clip import create_model, get_tokenizer

import torch
import torch.nn.functional as F
from torchvision import transforms

from .registry import register_model
from ..base.model_base import ModelBase
from ..base.seatizen_tools import join_GPS_metadata

try:
    from ..lib.engine_tools import NeuralNetworkGPU, build_and_save_engine_from_onnx
    HAS_TENSORRT = True
except ImportError:
    NeuralNetworkGPU = None
    build_and_save_engine_from_onnx = None
    HAS_TENSORRT = False

TOKENIZER_STR = "ViT-L-14"
HF_DATA_STR = "imageomics/TreeOfLife-200M"
NB_CLASS_TO_KEEP = 5
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
ranks = [a.title() for a in RANKS]

@register_model("bioclip2", default_weights="hf-hub:imageomics/bioclip-2")
class BioClip2(ModelBase):
    
    folder_name = "bioclip2"
    
    def __init__(self, weights, use_tensorrt: bool, batch_size: int):
        super().__init__(weights, use_tensorrt, batch_size)
        self.repo_name = weights
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_tensorrt = use_tensorrt and HAS_TENSORRT
        self.init_model()
    
    def init_model(self):
        self.image_processor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.tokenizer = get_tokenizer(TOKENIZER_STR)

        self.txt_emb = torch.from_numpy(np.load(hf_hub_download(
            repo_id=HF_DATA_STR,
            filename="embeddings/txt_emb_species.npy",
            repo_type="dataset",
        ))).to(self.device)
        
        with open(hf_hub_download(
            repo_id=HF_DATA_STR,
            filename="embeddings/txt_emb_species.json",
            repo_type="dataset",
        )) as fd:
            self.txt_names = json.load(fd)

        with open(hf_hub_download(
                repo_id="imageomics/bioclip-2-demo",
                filename="components/metadata.parquet",
                repo_type="space",
            )) as fd:

                self.metadata_df = pl.read_parquet(fd, low_memory = False)
                self.metadata_df = self.metadata_df.with_columns(pl.col(["eol_page_id", "gbif_id"]).cast(pl.Int64))


        if self.use_tensorrt:
            print("[INFO] Using TensorRT engine for bioclip2.")
            # self.model = NeuralNetworkGPU(get_multilabel_engine(self.weight_folder, self.repo_name, self.batch_size))
        else:
            print("[INFO] Using standard PyTorch model for bioclip2.")
            self.model = create_model(self.repo_name, output_dict=True, require_pretrained=True)
            self.model = self.model.to(self.device)


    def setup_new_session(self, session: Path):

        self.filename_pred = Path(session, "PROCESSED_DATA/IA", f"{session.name}_bioclip2.csv")
        self.csv_connector = open(self.filename_pred, "w")
        self.csv_connector.write(f"FileName,{','.join(ranks)},score,gbif_id,eol_id\n")


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
            for frame, frame_info in zip(data["frames"], data["frames_info"]):

                open_domain_output = self.predict(frame)
                key_with_max_value = max(open_domain_output, key=lambda k: open_domain_output[k])
                _, gbif_id, eol_id, _, _ = self.get_sample_data(key_with_max_value)
                self.csv_connector.write(f"{frame_info.filename},{key_with_max_value.split(' (')[0].replace(' ', ',')},{open_domain_output[key_with_max_value]},{gbif_id},{eol_id}\n")
  

    def _generator_tensorrt(self):
        for data in self._data_stream():
            continue
    

    def _generator_with_csv(self, base_generator):
        """CSV-writing wrapper around another generator."""

        for data in base_generator:
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


    def cleanup(self):
        self.csv_connector.close()


    def files_generate_by_model(self) -> list[Path]:
        return [self.csv_connector]
    

    def format_name(self, taxon, common):
        taxon = " ".join(taxon)
        if not common:
            return taxon
        return f"{taxon} ({common})"
    

    @torch.no_grad()
    def predict(self, img):
   
        img = self.image_processor(img).to(self.device)
        img_features = self.model.encode_image(img.unsqueeze(0))
        img_features = F.normalize(img_features, dim=-1)

        logits = (self.model.logit_scale.exp() * img_features @ self.txt_emb).squeeze()
        probs = F.softmax(logits, dim=0).to("cpu")
        topk = probs.topk(NB_CLASS_TO_KEEP)
        prediction_dict = {
            self.format_name(*self.txt_names[i]): prob for i, prob in zip(topk.indices, topk.values)
        }

        return prediction_dict


    def get_sample_data(self, pred_taxon, rank = 6):
        for idx in range(rank + 1):
            taxon = RANKS[idx]
            target_taxon = pred_taxon.split(" ")[idx]
            self.metadata_df = self.metadata_df.filter(pl.col(taxon) == target_taxon)

        if self.metadata_df.shape[0] == 0:
            return None, np.nan, np.nan, "", False

        # First, try to find entries with empty lower ranks
        exact_df = self.metadata_df
        for lower_rank in RANKS[rank + 1:]:
            exact_df = exact_df.filter((pl.col(lower_rank).is_null()) | (pl.col(lower_rank) == ""))

        if exact_df.shape[0] > 0:
            df_filtered = exact_df.sample()
            full_name = " ".join(df_filtered.select(RANKS[:rank+1]).row(0))
            return df_filtered["file_path"][0], df_filtered["gbif_taxon_id"].cast(pl.String)[0], df_filtered["eol_page_id"].cast(pl.String)[0], full_name, True

        # If no exact matches, return any entry with the specified rank
        df_filtered = self.metadata_df.sample()
        full_name = " ".join(df_filtered.select(RANKS[:rank+1]).row(0)) + " " + " ".join(df_filtered.select(RANKS[rank+1:]).row(0))
        return df_filtered["file_path"][0], df_filtered["gbif_taxon_id"].cast(pl.String)[0], df_filtered["eol_page_id"].cast(pl.String)[0], full_name, False


    def add_gps_position(self, metadata_path: Path) -> None:
        self.bioclip_gps = Path(metadata_path.parent, "bioclip_gps.csv")
        join_GPS_metadata(self.filename_pred, metadata_path, self.bioclip_gps)
    

    def add_pdf_pages(self, prefix: int, pdf_folder_tmp: Path, alpha3_code: int) -> Path:

        df = pd.read_csv(self.bioclip_gps)
        if len(df) == 0: return None # No predictions
        if "GPSLongitude" not in df or "GPSLatitude" not in df: return None # No GPS coordinate
        if round(df["GPSLatitude"].std(), 10) == 0.0 or round(df["GPSLongitude"].std(), 10) == 0.0: return None # All frames have the same gps coordinate

        summary = (
            df.groupby(ranks + ["gbif_id", "eol_id"])["FileName"]
            .nunique()  # counts unique filenames
            .reset_index(name="n_frames")
        )

        path_to_save_img = Path(pdf_folder_tmp, f"{prefix}_bioclipclass.jpg")

        # Take the first 20 rows
        top20 = summary.head(20)

        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')

        # Add title
        plt.title("BioClip2 — Summary of the 20 Most Frequent Taxa", fontsize=14, fontweight='bold', pad=20)

        table = ax.table(
            cellText=top20.values,
            colLabels=top20.columns,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)

        plt.tight_layout()

        # Save as image
        plt.savefig(path_to_save_img, dpi=300, bbox_inches="tight")
        plt.close()