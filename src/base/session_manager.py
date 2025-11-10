import shutil 
import pandas as pd
from pathlib import Path


class SessionManager:

    def __init__(self, session: Path, clean_session: bool) -> None:
        self.session = session

        self._metadata_path = Path(self.session, "METADATA/metadata.csv")
        self._ia_path = Path(self.session, "PROCESSED_DATA/IA")
        self._tmp_folder_pdf = Path(self.session, "tmp_folder")

        self.clean_session(clean_session)
    
    
    @property
    def metadata_path(self) -> Path:  
        return self._metadata_path
    
    @property
    def alpha3_code(self) -> str:
        return self.session.name.split("_")[1].split("-")[0]

    @property
    def tmp_folder_pdf(self) -> Path:
        return self._tmp_folder_pdf
    
    @property
    def ia_path(self) -> Path:
        return self._ia_path

    @property
    def pdf_file(self) -> Path:
        return Path(self.session, f"000_{self.session.name}_preview.pdf")

    def verify_metadata_csv(self) -> bool:
        return self._metadata_path.exists()
    
    def __repr__(self):
        return f"{self.session.name}"
        

    def clean_session(self, need_cleaning: bool):
        # Clean sessions if needed
        if need_cleaning:
            print("\t-- Clean session \n\n")
            # Clean PROCESSED_DATA/IA folder
            if self._ia_path.exists():
                shutil.rmtree(self._ia_path)

            if self._tmp_folder_pdf.exists():
                shutil.rmtree(self._tmp_folder_pdf)

            # Delete preview file
            for file in self.session.iterdir():
                if file.is_file() and file.suffix.lower() == ".pdf":
                    file.unlink()

        self._ia_path.mkdir(exist_ok=True, parents=True)
    

    def check_and_remove_predictions_files_if_necessary(self, min_prediction: int, jacque_pred_path: Path, files_generated_by_models: list[Path]) -> bool: 
        """ Remove all predictions stuff if we don't have enough predictions """

        if not jacque_pred_path.exists(): return False

        # Get all files who are not useless.
        jacques_df = pd.read_csv(jacque_pred_path)
        jacques_df = jacques_df[jacques_df["Useless"] == 0]

        # If the limit is higher. Return
        if len(jacques_df) >= min_prediction: return False

        # If we reached this point, we don't have enough predictions so we delete all.
        jacque_pred_path.unlink()

        for file in files_generated_by_models:
            file.unlink()

        return True
    
    
    def create_tmp_folder(self) -> None:
        if self._tmp_folder_pdf.exists():
            shutil.rmtree(self._tmp_folder_pdf)
        self._tmp_folder_pdf.mkdir(exist_ok=True, parents=True)