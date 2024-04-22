import traceback
from pathlib import Path

from utils.libs.seatizen_tools import create_pdf_preview

root_folders = {
    # "202210_plancha_session": "/media/bioeos/F/202210_plancha_session",
    # "202301-07_plancha_session": "/media/bioeos/F/202301-07_plancha_session",
    # "202305_plancha_session": "/media/bioeos/F/202305_plancha_session"

    # "2015_plancha_session": "/media/bioeos/E/2015_plancha_session",
    "2021_plancha_session": "/media/bioeos/E/2021_plancha_session",
    # "202211_plancha_session": "/media/bioeos/E/202211_plancha_session"
    # "202309_plancha_session": "/media/bioeos/E/202309_plancha_session",
    # "202310_plancha_session": "/media/bioeos/E/202310_plancha_session"

    # "202311_plancha_session": "/media/bioeos/D/202311_plancha_session",
    # "202312_plancha_session": "/media/bioeos/D/202312_plancha_session"
}

def main():
    sessions_fail, nb_session = [], 0
    for root_name in root_folders:
            
        # Get root folder
        root_folder = Path(root_folders[root_name])
        if not Path.exists(root_folder) or not root_folder.is_dir():
            print(f"\n\n[ERROR ROOT FOLDER] Cannot find root folder or root folder isn't a directory for: {root_folder}\n\n")
            continue
        
        for session in sorted(list(root_folder.iterdir())):
            nb_session += 1
            session_name = session.name

            if not session.is_dir():
                print(f"\n\n[ERROR] Session {session} isn't a directory")
                continue
            print(f"\n\n\nWorking with {session_name}")
            
            metadata = Path(session, "METADATA/metadata.csv")
            frames = Path(session, "PROCESSED_DATA/FRAMES")
            metadata_gps = Path(session, "METADATA/metadata_gps.csv")

            if not Path.exists(metadata):
                print(f"No metadata file for {session_name}")
                continue

            if not Path.exists(metadata_gps):
                print(f"No metadata file for {session_name}")
                continue

            if not Path.exists(frames) or len(list(frames.iterdir())) == 0:
                print(f"No image for session {session_name}")
                continue 
            
            try:
                create_pdf_preview(session, session, Path(session).name, sorted(list(Path(session, "PROCESSED_DATA/FRAMES").iterdir())))
            
            except Exception:
                print(traceback.format_exc(), end="\n\n")

                sessions_fail.append(session_name)

    # Stat
    print("\nEnd of process. On {} sessions, {} fails. ".format(nb_session, len(sessions_fail)))
    if (len(sessions_fail)):
        [print("\t* " + session_name) for session_name in sessions_fail]

main()