from pathlib import Path
from .pipeline import Pipeline
class JacquesPredictions(Pipeline):
    """Pipeline task to save Jacques predictions"""

    def __init__(self):
        self.filename, self.csv_connector = None, None
        super(JacquesPredictions, self).__init__()

    def setup(self, filename: Path):
        self.filename = filename
        self.csv_connector = open(self.filename, "w")

    def generator(self):
        """ Write in csv file"""
        self.csv_connector.write("FileName,Useless,Score\n")

        data, stop = None, False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream.
                data = next(self.source)
            except StopIteration:
                stop = True

            if not stop and data:
                for i, frame_info in enumerate(data["frames_info"]):
                    frame_name = frame_info.filename
                    self.csv_connector.write(f"{frame_name},{data['Useless'][i]},{data['prob_jacques'][i]}\n")
            
                yield data
    
    def cleanup(self):
        self.csv_connector.close()

class  MultilabelPredictions(Pipeline):
    """Pipeline task to save Jacques predictions"""

    def __init__(self, classes: list):
        self.filename_pred, self.filename_scores, self.csv_connector_classes, self.csv_connector_scores = None, None, None, None
        self.classes = classes
        super(MultilabelPredictions, self).__init__()
    
    def setup(self, filename_pred: Path, filename_scores: Path):
        self.filename_pred = filename_pred
        self.filename_scores = filename_scores
        self.csv_connector_classes = open(self.filename_pred, "w")
        self.csv_connector_scores = open(self.filename_scores, "w")

    def generator(self):
        """ Write in csv file"""
        classe_to_write = ",".join(self.classes)
        self.csv_connector_classes.write(f"FileName,{classe_to_write}\n")
        self.csv_connector_scores.write(f"FileName,{classe_to_write}\n")


        data = None
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream.
                data = next(self.source)
            except StopIteration:
                stop = True

            if not stop and data:
                if "multilabel_labels" in data:
                    for i, frame_info in enumerate(data["frames_info"]):
                        frame_name = frame_info.filename
                        labels = ['1' if lab in data["multilabel_labels"][i] else '0' for lab in self.classes] 
                        self.csv_connector_classes.write(f"{frame_name},{','.join(labels)}\n")
                        self.csv_connector_scores.write(f"{frame_name},{','.join(data['multilabel_scores'][i])}\n")
            
                yield data
    
    def cleanup(self):
        self.csv_connector_classes.close()
        self.csv_connector_scores.close()