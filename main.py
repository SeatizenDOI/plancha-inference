import collections
import heapq
import json
import logging

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from open_clip import create_model, get_tokenizer
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()


# For sample images
METADATA_PATH = "components/metadata.parquet"
# Read page IDs as int
metadata_df = pl.read_parquet(METADATA_PATH, low_memory = False)
metadata_df = metadata_df.with_columns(pl.col(["eol_page_id", "gbif_id"]).cast(pl.Int64))

model_str = "hf-hub:imageomics/bioclip-2"
tokenizer_str = "ViT-L-14"
HF_DATA_STR = "imageomics/TreeOfLife-200M"

min_prob = 1e-9
k = 5

device = torch.device("cpu")

preprocess_img = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

ranks = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")

def format_name(taxon, common):
    taxon = " ".join(taxon)
    if not common:
        return taxon
    return f"{taxon} ({common})"


@torch.no_grad()
def open_domain_classification(img, rank: int, return_all=False):
    """
    Predicts from the entire tree of life.
    If targeting a higher rank than species, then this function predicts among all
    species, then sums up species-level probabilities for the given rank.
    """

    logger.info(f"Starting open domain classification for rank: {rank}")
    img = preprocess_img(img).to(device)
    img_features = model.encode_image(img.unsqueeze(0))
    img_features = F.normalize(img_features, dim=-1)

    logits = (model.logit_scale.exp() * img_features @ txt_emb).squeeze()
    probs = F.softmax(logits, dim=0)

    if rank + 1 == len(ranks):
        topk = probs.topk(k)
        prediction_dict = {
            format_name(*txt_names[i]): prob for i, prob in zip(topk.indices, topk.values)
        }
        logger.info(f"Top K predictions: {prediction_dict}")
        top_prediction_name = format_name(*txt_names[topk.indices[0]]).split("(")[0]
        # logger.info(f"Top prediction name: {top_prediction_name}")
        # sample_img, taxon_url = get_sample(metadata_df, top_prediction_name, rank)
        if return_all:
            return prediction_dict, sample_img, taxon_url
        return prediction_dict

    output = collections.defaultdict(float)
    for i in torch.nonzero(probs > min_prob).squeeze():
        output[" ".join(txt_names[i][0][: rank + 1])] += probs[i]

    topk_names = heapq.nlargest(k, output, key=output.get)
    prediction_dict = {name: output[name] for name in topk_names}
    logger.info(f"Top K names for output: {topk_names}")
    logger.info(f"Prediction dictionary: {prediction_dict}")

    top_prediction_name = topk_names[0]
    logger.info(f"Top prediction name: {top_prediction_name}")
    # sample_img, taxon_url = get_sample(metadata_df, top_prediction_name, rank)
    # logger.info(f"Sample image and taxon URL: {sample_img}, {taxon_url}")

    if return_all:
        return prediction_dict, sample_img, taxon_url
    return prediction_dict


if __name__ == "__main__":
    logger.info("Starting.")
    model = create_model(model_str, output_dict=True, require_pretrained=True)
    model = model.to(device)
    logger.info("Created model.")

    model = torch.compile(model)
    logger.info("Compiled model.")

    tokenizer = get_tokenizer(tokenizer_str)

    txt_emb = torch.from_numpy(np.load(hf_hub_download(
        repo_id=HF_DATA_STR,
        filename="embeddings/txt_emb_species.npy",
        repo_type="dataset",
    )))
    with open(hf_hub_download(
        repo_id=HF_DATA_STR,
        filename="embeddings/txt_emb_species.json",
        repo_type="dataset",
    )) as fd:
        txt_names = json.load(fd)

    done = txt_emb.any(axis=0).sum().item()
    total = txt_emb.shape[1]
    status_msg = ""
    if done != total:
        status_msg = f"{done}/{total} ({done / total * 100:.1f}%) indexed"

    img_input = Image.open("inputs/image_multilabel_setup.jpeg")
    open_domain_output, sample_img, taxon_url = open_domain_classification(img_input, len(ranks)-1, return_all=True)


    
