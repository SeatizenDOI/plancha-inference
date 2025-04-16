import torch
from itertools import compress

from ..base.tools import sigmoid
from ..base.multilabel_model_base import MultiLabelClassifierBase, NewHeadDinoV2ForImageClassification

class MultiLabelClassifier(MultiLabelClassifierBase):
    """Multilabel classifier with cuda"""

    def __init__(self, repo_name, batch_size):
        super().__init__(repo_name, batch_size)

        self.model = NewHeadDinoV2ForImageClassification.from_pretrained(repo_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    def generator(self):
        # Pass model to gpu
        self.model = self.model.to(self.device)

        data = None
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream.
                data = next(self.source)
            except StopIteration:
                stop = True

            if not stop and data:
                # Check if image is not useless
                if "Useless" not in data or 0 in data["Useless"]:
                    inputs = self.image_processor(data["frames"], return_tensors="pt")
                    inputs = inputs.to(self.device)
                    with torch.no_grad():
                        model_outputs = self.model(**inputs)
                    
                    data["multilabel_scores"], data["multilabel_labels"] = [], []
                    for logit in model_outputs["logits"]:
                        scores = sigmoid(logit.cpu().numpy())
                    
                        data["multilabel_scores"].append([str(s) for s in scores])

                        # Apply threshold. 
                        scores = self.applyThreshold(scores)

                        # Send predicted label
                        predicted_label = list(compress(self.classes_name,  scores))
                        data["multilabel_labels"].append(predicted_label)
                
                yield data