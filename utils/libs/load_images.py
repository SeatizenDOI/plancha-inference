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
    path_to_frames_folder = Path(src, "PROCESSED_DATA", "FRAMES")
    
    # Check if session exists.
    if not Path.exists(path_to_frames_folder) or not Path.is_dir(path_to_frames_folder):
        print(f"Path to folder {path_to_frames_folder} doesn't exist.")
        return []

    # Iter on each file
    cpt_image, cpt_error = 0, 0
    frames_path = []
    for file in Path.iterdir(path_to_frames_folder):
        cpt_image += 1
        # Check if it's a file and if ended with image extension
        if not file.is_file() or not file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            cpt_error += 1
            continue
        frames_path.append(Path(path_to_frames_folder, file))

    print(f"Folder {path_to_frames_folder}, number of files: {cpt_image}, number of errors: {cpt_error}")

    return frames_path