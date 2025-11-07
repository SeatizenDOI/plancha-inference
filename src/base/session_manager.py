from pathlib import Path
import shutil 
class SessionManager:

    def __init__(self, session: Path, clean_session: bool) -> None:
        self.session = session

        self._metadata_path = Path(self.session, "METADATA/metadata.csv")

        self.clean_session(clean_session)
    
    
    @property
    def metadata_path(self) -> Path:  
        return self._metadata_path


    def verify_metadata_csv(self) -> bool:
        return self._metadata_path.exists()
    
    def __repr__(self):
        return f"{self.session.name}"
        

    def clean_session(self, need_cleaning: bool):
        # Clean sessions if needed
        path_IA = Path(self.session, "PROCESSED_DATA/IA")
        if need_cleaning:
            print("\t-- Clean session \n\n")
            # Clean PROCESSED_DATA/IA folder
            if path_IA.exists():
                shutil.rmtree(path_IA)

            # Delete preview file
            for file in self.session.iterdir():
                if file.is_file() and file.suffix.lower() == ".pdf":
                    file.unlink()

        path_IA.mkdir(exist_ok=True, parents=True)
    
    def check_and_remove_predictions_files_if_necessary(self) -> bool: 
        """ Remove all predictions stuff if we don't have enough predictions """

        pass