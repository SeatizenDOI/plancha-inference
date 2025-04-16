import shutil
import traceback
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace

from utils.capture_images import CaptureImages
from utils.savers import JacquesPredictions, MultilabelPredictions
from utils.libs.predictions_raster_tools import create_rasters_for_classes
from utils.jacques_predictor import JacquesPredictor, JacquesPredictorGPU, JacquesCSV
from utils.multilabel_classifier import MultiLabelClassifierCUDA, MultiLabelClassifierTRT

from utils.libs.parse_opt import Sources, get_list_sessions
from utils.libs.seatizen_tools import create_pdf_preview, join_GPS_metadata, get_uselful_images, check_and_remove_predictions_files_if_necessary

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
    ap.add_argument("-pcsv", "--path_csv_file", default="./csv_inputs/stleu.csv", help="Load all images from session write in the provided csv file")

    # Choose how to used jacques model.
    ap.add_argument("-jcku", "--jacques_checkpoint_url", default="20240513_v20.0", help="Specified which checkpoint file to used, if checkpoint file is not found we downloaded it")
    ap.add_argument("-jgpu", "--jacques_gpu", action="store_true", help="Build an engine from jacques_checkpoint_url, use tensorrt to speedup inference")
    ap.add_argument("-jcsv", "--jacques_csv", action="store_true", help="Used csv file of jacques predictions")
    ap.add_argument("-nj", "--no_jacques", action="store_true", help="Didn't used jacques model")
    
    # Choose how to used multilabel model.
    ap.add_argument("-mlu", "--multilabel_url", default="lombardata/DinoVdeau-large-2024_04_03-with_data_aug_batch-size32_epochs150_freeze", help="Hugging face repository")
    ap.add_argument("-mlgpu", "--multilabel_gpu", action="store_true", help="Speedup inference with tensorrt")
    ap.add_argument("-nml", "--no_multilabel", action="store_true", help="Didn't used multilabel model")

    # Optional arguments.
    ap.add_argument("-np", "--no-progress", action="store_true", help="Hide display progress")
    ap.add_argument("-ns", "--no-save", action="store_true", help="Don't save annotations")
    ap.add_argument("-npr", "--no_prediction_raster", action="store_true", help="Don't produce predictions rasters")
    ap.add_argument("-c", "--clean", action="store_true", help="Clean pdf preview and predictions files")
    ap.add_argument("-is", "--index_start", default="0", help="Choose from which index to start")
    ap.add_argument("-ip", "--index_position", default="-1", help="if != -1, take only session at selected index")
    ap.add_argument("-bs", "--batch_size", default="1", help="Numbers of frames processed in one time")
    ap.add_argument("-minp", "--min_prediction", default="100", help="Minimum for keeping predictions after inference.")

    return ap.parse_args()

def pipeline_seatizen(opt: Namespace):
    print("\n-- Parse input options", end="\n\n")
    
    batch_size = int(opt.batch_size) if opt.batch_size.isnumeric() else 1
    min_prediction = int(opt.min_prediction) if opt.min_prediction.isnumeric() else 100

    print("\n-- Load the pipeline ...", end="\n\n")

    # Image input (csv with session path, folder_path_to_sessions, path_to_sessions)
    capture_images = CaptureImages(batch_size)

    # Load jacques model.
    jacques_model = None 
    if opt.no_jacques:
        jacques_model = None
    elif opt.jacques_csv:
        jacques_model = JacquesCSV()
    elif opt.jacques_gpu:
        jacques_model = JacquesPredictorGPU(opt.jacques_checkpoint_url, batch_size)
    else:
        jacques_model = JacquesPredictor(opt.jacques_checkpoint_url, batch_size)

    # Load Hugging face model.
    multilabel_model = None 
    if opt.no_multilabel:
        multilabel_model = None 
    elif opt.multilabel_gpu:
        multilabel_model = MultiLabelClassifierTRT(opt.multilabel_url, batch_size)
    else:
        multilabel_model = MultiLabelClassifierCUDA(opt.multilabel_url, batch_size)

    # Annotage images.
    jacques_savers = None if opt.no_jacques or opt.jacques_csv else JacquesPredictions()
    multilabel_savers = None if opt.no_multilabel else MultilabelPredictions(multilabel_model.classes_name)

    if opt.no_save:
        jacques_savers, multilabel_savers = None, None
 
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

        session_name = Path(session).name
        jacques_model_name = opt.jacques_checkpoint_url.replace("/", "_")
        print(f"\nLaunched session {session_name}\n\n")

        # Clean sessions if needed
        if opt.clean:
            print("\t-- Clean session \n\n")
            # Clean PROCESSED_DATA/IA folder
            path_IA = Path(session, "PROCESSED_DATA/IA")
            if Path.exists(path_IA):
                shutil.rmtree(path_IA)
            path_IA.mkdir(exist_ok=True, parents=True)

            # Delete preview file
            for file in Path(session).iterdir():
                if file.is_file() and file.suffix.lower() == ".pdf":
                    file.unlink()
        else:
            path_IA = Path(session, "PROCESSED_DATA/IA")
            path_IA.mkdir(exist_ok=True, parents=True)

        jacques_csv_name = Path(session, "PROCESSED_DATA/IA", f"{session_name}_jacques-v0.1.0_model-{jacques_model_name}.csv")
        multilabel_pred_csv_name = Path(session, "PROCESSED_DATA/IA", f"{session_name}_{opt.multilabel_url.replace('/', '_')}.csv")
        multilabel_scores_csv_name = Path(session, "PROCESSED_DATA/IA", f"{session_name}_{opt.multilabel_url.replace('/', '_')}_scores.csv")

        metadata_csv_name = Path(session, "METADATA/metadata.csv")
        if not Path.exists(metadata_csv_name):
            print(f"[ERROR] Session {session_name} doesn't have a metadata file.")
            sessions_fail.append(session_name)
            continue

        # Setup pipeline for current session
        capture_images.setup(session, Sources.SESSION)
        if opt.jacques_csv:
            jacques_model.setup(jacques_csv_name)
        if jacques_savers:
            jacques_savers.setup(jacques_csv_name)
        if multilabel_savers:
            multilabel_savers.setup(multilabel_pred_csv_name, multilabel_scores_csv_name)      

        pipeline = (
            capture_images |
            jacques_model  |
            multilabel_model |
            jacques_savers |
            multilabel_savers
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

        # Pipeline cleanup
        if jacques_savers:
            jacques_savers.cleanup()
        if multilabel_savers:
            multilabel_savers.cleanup()

        print(f"\n -- Elapsed time: {datetime.now() - start_t} seconds\n\n")

        if opt.no_multilabel: continue
        try:
            predictions_gps = Path(session, "METADATA", "predictions_gps.csv")
            predictions_scores_gps = Path(session, "METADATA", "predictions_scores_gps.csv")
            useful_images = get_uselful_images(capture_images.frames_path, jacques_csv_name)

            # Create predictions_gps.csv
            print("\t-- Join metadata GPS")
            join_GPS_metadata(multilabel_pred_csv_name, metadata_csv_name, str(predictions_gps))
            join_GPS_metadata(multilabel_scores_csv_name, metadata_csv_name, str(predictions_scores_gps))

            # Remove predictions if minimal number of predictions is not achieve. We don't remove frame because sometimes it's also the raw data.
            if check_and_remove_predictions_files_if_necessary(session, predictions_gps, predictions_scores_gps, min_prediction):
                print("[WARNING] All predictions have been removed due to lack of multilabel predictions.")
                continue
            
            # Add preview pdf
            print("\t-- Create pdf preview \n\n") # TODO Add bathy preview in global pdf
            create_pdf_preview(session, session_name, useful_images, metadata_csv_name, predictions_gps, multilabel_model.classes_name)

            # Create raster predictions
            print("\t-- Creating raster for each class \n\n")
            if not opt.no_prediction_raster:
                create_rasters_for_classes(predictions_scores_gps, multilabel_model.classes_name, path_IA, session_name, 'linear')
            
            print(f"\nSession {session_name} end succesfully ! ", end="\n\n\n")

        except Exception:
            print(traceback.format_exc(), end="\n\n")

            sessions_fail.append(session_name)
    
    # Stat
    print("\nEnd of process. On {} sessions, {} fails. ".format(len(list_session), len(sessions_fail)))
    if (len(sessions_fail)):
        [print("\t* " + session_name) for session_name in sessions_fail]
    
if __name__ == "__main__":
    args = parse_args()
    pipeline_seatizen(args)