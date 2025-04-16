import pandas as pd
from pathlib import Path

from .parse_opt import Sources

def load_frames_from_source(src, mode):
    """ Returns to the correct import function"""
    frames = None
    
    if mode == Sources.CSV_SESSION:
        frames = load_frames_from_csv_session(src)
    elif mode == Sources.FOLDER:
        frames = load_frames_from_folder(src)
    elif mode == Sources.SESSION:
        frames = load_frames_from_session(src)
    
    return frames 

def load_frames_from_csv_session(src):
    """ Import frames from a csv listing sessions """
    src = Path(src)

    # Check if csv file exists.
    if not Path.exists(src) or not Path.is_file(src):
        print(f"Path to file {src} doesn't exist.")
        return []

    file = pd.read_csv(src)
    frames = []
    for row in file.itertuples(index=False):
        folder_path = Path(row.root_folder, row.session_name)
        frames += load_frames_from_session(folder_path)

    print(f"Successfully load {len(frames)} images")
    return frames

def load_frames_from_folder(src):
    """ Import frames from a folder of sessions """
    src = Path(src)

    # Check if folder of sessions exists.
    if not Path.exists(src) or not Path.is_dir(src):
        print(f"Path to folder {src} doesn't exist.")
        return []

    frames = []
    for folder in Path.iterdir(src):
        folder_path = Path(src, folder)
        frames += load_frames_from_session(folder_path)

    print(f"Successfully load {len(frames)} images")
    return frames

def load_frames_from_session(src):
    """ Import frames from a single session """
    frames_path = []

    # Get metadata_file csv
    path_metadata = Path(src, "METADATA", "metadata.csv")
    if not Path.exists(path_metadata) or not path_metadata.is_file():
        print(f"No metadata.csv for session {src}, cannot extract file.")
        return frames_path
    
    metadata_df = pd.read_csv(path_metadata)
    if len(metadata_df) == 0: return frames_path

    try:
        relative_path_key = [key for key in list(metadata_df) if "relative_file_path" in key][0]
    except Exception:
        raise NameError(f"Cannot find relative path key for {src}")

    # Iter on each file
    cpt_image, cpt_error = 0, 0
    for _, row in metadata_df.iterrows():
        path_img = Path(Path(src).parent, *[x for x in row[relative_path_key].split("/") if x]) # Sometimes relative path start with /
        cpt_image += 1
        # Check if it's a file and if ended with image extension
        if not path_img.exists() or not path_img.is_file() or not path_img.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            cpt_error += 1
            continue
        frames_path.append(path_img)
    print(f"Folder {src}, number of files: {cpt_image}, number of errors: {cpt_error}")
    return frames_path