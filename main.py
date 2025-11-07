import shutil
import traceback
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace

from src.base.capture_images import CaptureImages
from src.models.Jacques import Jacques
from src.models.registry import MODEL_REGISTRY
from src.base.model_manager import ModelsManager
from src.lib.parse_opt import Sources, get_list_sessions
from src.base.session_manager import SessionManager

def parse_args() -> Namespace:

    # Parse command line arguments.
    ap = ArgumentParser(description="Seatizen inference", epilog="Thanks to use it!")

    # Input.
    arg_input = ap.add_mutually_exclusive_group(required=True)
    arg_input.add_argument("-efol", "--enable_folder", action="store_true", help="Take all images from a folder of a seatizen session")
    arg_input.add_argument("-eses", "--enable_session", action="store_true", help="Take all images from a single seatizen session")
    arg_input.add_argument("-ecsv", "--enable_csv", action="store_true", help="Take all images from session in csv file")

    # Path of input.
    ap.add_argument("-pfol", "--path_folder", default="/media/bioeos/D/202311_plancha_session/", help="Load all images from a folder of sessions")
    ap.add_argument("-pses", "--path_session", default="/home/bioeos/Documents/Bioeos/annotations_some_image/20240524_REU-LE-PORT_HUMAN-1_01/", help="Load all images from a single session")
    ap.add_argument("-pcsv", "--path_csv_file", default="./csv_inputs/temp.csv", help="Load all images from session write in the provided csv file")

    # Choose how to used jacques model.
    ap.add_argument("-jcku", "--jacques_checkpoint_url", default="20240513_v20.0", help="Specified which checkpoint file to used, if checkpoint file is not found we downloaded it")
    ap.add_argument("-nj", "--no_jacques", action="store_true", help="Didn't used jacques model")

    ap.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_REGISTRY.keys(),
        help="List of model names to load.",
        default=["dinovdeau"]
    )
    ap.add_argument(
        "--weights",
        nargs="+",
        help=(
            "Optional list of weight paths corresponding to the models. "
            "If omitted, defaults from MODEL_REGISTRY will be used."
        )
    )

    # Optional arguments.
    ap.add_argument("-trt", "--tensorrt", action="store_true", help="Try to use TensorRT")
    ap.add_argument("-np", "--no-progress", action="store_true", help="Hide display progress")
    ap.add_argument("-ns", "--no-save", action="store_true", help="Don't save annotations")
    ap.add_argument("-npr", "--no_prediction_raster", action="store_true", help="Don't produce predictions rasters")
    ap.add_argument("-c", "--clean", action="store_true", help="Clean pdf preview and predictions files")
    ap.add_argument("-is", "--index_start", default="0", help="Choose from which index to start")
    ap.add_argument("-ip", "--index_position", default="-1", help="if != -1, take only session at selected index")
    ap.add_argument("-bs", "--batch_size", default="1", help="Numbers of frames processed in one time")
    ap.add_argument("-minp", "--min_prediction", default="100", help="Minimum for keeping predictions after inference.")

    return ap.parse_args()

def main(opt: Namespace):
    print("\n-- Parse input options", end="\n\n")

    batch_size = int(opt.batch_size) if opt.batch_size.isnumeric() else 1
    min_prediction = int(opt.min_prediction) if opt.min_prediction.isnumeric() else 100

    print("\n-- Load the pipeline ...", end="\n\n")

    # Image input (csv with session path, folder_path_to_sessions, path_to_sessions)
    capture_images = CaptureImages(batch_size)

    # Jacques. Not used in model manager because jacques is mandatory in a seatizen session.
    jacques_model = Jacques(opt.jacques_checkpoint_url, opt.tensorrt, batch_size)


    # Model manager to deal with all kind of model.
    models_manager = ModelsManager(opt.models, opt.weights, opt.tensorrt, batch_size)


    # Stat
    sessions_fail = []
    list_session = get_list_sessions(opt)
    index_start = int(opt.index_start) if opt.index_start.isnumeric() and int(opt.index_start) < len(list_session) else 0
    index_position = int(opt.index_position)-1 if opt.index_position.isnumeric() and \
                                            int(opt.index_position) > 0 and \
                                            int(opt.index_position) <= len(list_session) else -1
    sessions = list_session[index_start:] if index_position == -1 else [list_session[index_position]]
    print("\n-- Start inference !", end="\n\n")

    for session in sessions:

        print(f"\nLaunched session {session.name}\n\n")
        session_manager = SessionManager(session, opt.clean)

        if not session_manager.verify_metadata_csv():
            print(f"[ERROR] Session {session.name} doesn't have a metadata file.")
            sessions_fail.append(session.name)
            continue

        # Setup pipeline for current session
        capture_images.setup_new_session(session, Sources.SESSION)
        jacques_model.setup_new_session(session)
        models_manager.setup_new_session(session)

        pipeline = (
            capture_images |
            jacques_model |
            models_manager
        )

        # Iterate through pipeline
        start_t = datetime.now()
        print("\t-- Start prediction session \n\n")
        progress = tqdm(total=capture_images.frame_count//batch_size,
                        disable=opt.no_progress)
        
        try:
            for _ in pipeline:
                progress.update(1)
        except StopIteration:
            return
        except KeyboardInterrupt:
            return
        finally:
            progress.close()

        jacques_model.cleanup()
        models_manager.cleanup()

        print(f"\n -- Elapsed time: {datetime.now() - start_t} seconds\n\n")

        try:

            # Add gpsposition to predictions.
            
            # Remove predictions if minimal number of predictions is not achieve. We don't remove frame because sometimes it's also the raw data.

            # Create pdf preview.

            # Create raster predictions.
            
            print(f"\nSession {session.name} end succesfully ! ", end="\n\n\n")

        except Exception:
            print(traceback.format_exc(), end="\n\n")

            sessions_fail.append(session.name)
    
    # Stat
    print("\nEnd of process. On {} sessions, {} fails. ".format(len(sessions), len(sessions_fail)))
    if (len(sessions_fail)):
        [print("\t* " + session_name) for session_name in sessions_fail]
    
if __name__ == "__main__":
    args = parse_args()
    main(args)