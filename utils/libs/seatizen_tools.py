"""
    All functions is taken from seatizen-to-zenodo repository. Except predictions map
    https://github.com/IRDG2OI/seatizen-to-zenodo
"""

import os
import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from textwrap import wrap
import cartopy.crs as ccrs
from pypdf import PdfMerger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table
from reportlab.lib.pagesizes import letter, landscape
from cartopy.io.img_tiles import GoogleTiles

def list_directories(path):
    '''
    Function that only list the directories located at the specified path.
    '''
    directories = [os.path.join(path, entry) for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]
    return directories

def join_GPS_metadata(annotation_csv_path, metadata_path, merged_csv_path):
    '''
    Function to merge multilabel annotations csv with GPS metadata (latitude, longitude and date)
    '''
    annot_df = pd.read_csv(annotation_csv_path)
    gps_df = pd.read_csv(metadata_path)
    
    # Extract image names from the file paths
    annot_df['Image_name'] = annot_df['FileName']
    gps_df['Image_name'] = gps_df['FileName']

    # Merge the DataFrames based on the image names
    # Sometimes we don't have this information due to no bin
    keys = [key for key in ['Image_name', 'GPSDateTime', 'SubSecDateTimeOriginal', 'GPSLatitude', 'GPSLongitude', 'GPSTrack', 'GPSRoll', 'GPSPitch'] if key in gps_df] 
    try:
        merged_df = annot_df.merge(gps_df[keys], on='Image_name', how='left')
    except KeyError:
        print("[ERROR] No key to merge gps information in metadata.")
    
    # Drop the 'Image_name' column from merged_df
    merged_df.drop(columns='Image_name', inplace=True)
    
    merged_df.to_csv(merged_csv_path, index=False, header=True)

def evenly_select_images_on_interval(image_list):
    '''
    Function to select images evenly throughout a list based on their indexes.
    '''
    total_images = len(image_list)
    index_list = np.linspace(0, total_images, 100, dtype=int, endpoint=False)
    selected_images = [image_list[i] for i in index_list]
    return selected_images

def create_trajectory_map(metadata_path, global_trajectories):
    '''
    Function to create the trajectory maps.
    - metadata_path is the path to the metadata.csv file or the metadata_image.csv file.
    - global_trajectories is a boolean to indicate if you are doing the global trajectory map or not.
    Return True if image was generated else False
    '''
    df = pd.read_csv(metadata_path)
    if "GPSLatitude" not in df or "GPSLongitude" not in df:
        print("[ERROR] Not enough gps information to draw trajectory map")
        return False

    imagery = GoogleTiles(url='https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}')

    if not global_trajectories:
        fig = plt.figure(figsize=(2,2), dpi=300)
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        map_path = "map.png"
        adb_lat_range = [-13, -7] # latitude range of aldabra and mayotte
        if df['GPSLatitude'].between(adb_lat_range[0], adb_lat_range[1]).any():
            ax.set_extent([df.GPSLongitude.min()-0.005, df.GPSLongitude.max()+0.005, df.GPSLatitude.min()-0.005,df.GPSLatitude.max()+0.005])
            ax.add_image(imagery, 17) # aldabra/mayotte position so we adjust the zoom level
            ax.plot(df.GPSLongitude, df.GPSLatitude, color='tab:grey', linewidth=0.1)
        else: # other positions
            ax.set_extent([df.GPSLongitude.min()-0.001, df.GPSLongitude.max()+0.001, df.GPSLatitude.min()-0.001,df.GPSLatitude.max()+0.001])
            ax.add_image(imagery, 19)
            ax.plot(df.GPSLongitude, df.GPSLatitude, color='tab:grey', linewidth=0.3)
    else: # global trajectories map
        df['GPSDateTime'] = pd.to_datetime(df['GPSDateTime'])
        df['SubSecDateTimeOriginal'] = pd.to_datetime(df['SubSecDateTimeOriginal'], format="%Y:%m:%d %H:%M:%S.%f")
        df['Date'] = df['GPSDateTime'].fillna(df['SubSecDateTimeOriginal'])
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        fig = plt.figure(figsize=(10,10), dpi=600)
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        ax.set_extent([df.GPSLongitude.min()-1, df.GPSLongitude.max()+1, df.GPSLatitude.min()-1,df.GPSLatitude.max()+1])
        ax.add_image(imagery, 19)

        # drawing trajectories for each session based on their associated dates
        for date in df["Date"].unique():
            subset = df[df['Date'] == date]
            ax.plot(subset.GPSLongitude, subset.GPSLatitude, color='yellow', linewidth=5)
        # saving the map in the same folder as metadata_image.csv
        map_path = os.path.join(os.path.dirname(metadata_path), "000_global_map.png")

    fig.savefig(map_path, bbox_inches='tight',pad_inches=0, dpi=300)
    print("Trajectory map created!")
    return True

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def create_predictions_map(predictions_path, classes):
    """
        Create a folder of map for each predictions.
        - predictions_path is the path to predictions file
        
        return the folder path to the images 
    """
    predictions_path = Path(predictions_path)
    if not Path.exists(predictions_path):
        print(f"File {predictions_path} doesn't exist")
        return

    df = pd.read_csv(predictions_path)
    if len(df) == 0: return None # No predictions
    if "GPSLongitude" not in df or "GPSLatitude" not in df: return None # No GPS coordinate

    imagery = GoogleTiles(url='https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}')

    # Create temp directory
    tmp_path = Path("/tmp/pred_jpg")
    tmp_path.mkdir(parents=True, exist_ok=True)
    if len(list(tmp_path.iterdir())) > 0:
        for i in tmp_path.iterdir():
            i.unlink()
    
    cmap = get_cmap(len(classes))
    for i, category in tqdm.tqdm(enumerate(classes)):
        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        ax.set_extent([df.GPSLongitude.min()-0.0003, df.GPSLongitude.max()+0.0003, df.GPSLatitude.min()-0.0003, df.GPSLatitude.max()+0.0003])
        ax.add_image(imagery, 17)
        ax.plot(df[df[category] == 1].GPSLongitude, df[df[category] == 1].GPSLatitude, '.', color=cmap(i), markersize=2.5, markeredgewidth=0)
        ax.plot(df[df[category] == 0].GPSLongitude, df[df[category] == 0].GPSLatitude, '.', color='tab:gray', markersize=2.0, markeredgewidth=0)
        ax.set_title(category)
        path_to_save_img = Path(tmp_path, f"multiple_page_{category.replace('/', '')}_subplots.jpg")
        plt.savefig(str(path_to_save_img), dpi=300)
        plt.close()

    return tmp_path

def get_uselful_images(frame_path, jacques_predictions):

    list_frames_useful = []
    df_jacques = pd.read_csv(jacques_predictions)
    
    for frame in sorted(list(frame_path.iterdir())):
        result = df_jacques[df_jacques["FileName"] == frame.name]
        if len(result) == 0: continue
        
        if result.iloc[0]["Useless"] == 0:
            list_frames_useful.append(frame)

    return list_frames_useful

def create_pdf_preview(pdf_preview_path, session_name, list_of_images, metadata_path, prediction_gps_path, classes):
    '''
    Function to create a pdf preview of the session. It will contains:
    - a trajectory map
    - 100 thumbnails of images selected evenly throughout the session
    - a sneakpeek to the metadata file of the session
    '''

    # PDF creation
    pdf_file = os.path.join(pdf_preview_path, f"000_{session_name}_preview.pdf")
    c = canvas.Canvas(pdf_file, pagesize=letter)
    page_width, page_height = letter
    max_height = page_height - 100

    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, 730, "Session Summary")
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.blue)
    c.drawString(30, 705, session_name)

    # Trajectory map
    if create_trajectory_map(metadata_path, False):
        print("Adding map to the PDF...")
        image_map = Image.open("map.png")
        image_map_width, image_map_height = image_map.size
        x = (page_width - image_map_width) / 2
        y = (page_height - image_map_height) / 2
        c.drawImage("map.png", x, y)
        os.remove("map.png") # deleting map.png

        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(colors.black)
        c.drawString(30, 650, "Trajectory map")
        print("Map added!")

    # Thumbnails
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, 730, "Images previews")

    selected_images = evenly_select_images_on_interval(list_of_images)
    print("Images previews selected!\n")
    x_coord = 30
    y_coord = max_height

    for i, image in enumerate(selected_images):
        if i % 5 == 0 and i != 0:
            # Start a new row of images
            x_coord = 30
            y_coord -= 110

        img = Image.open(image)
        img.thumbnail((100, 100))

        img_width, img_height = img.size

        temp_image_path = os.path.join(pdf_preview_path, f'temp_{i}.jpg')
        img.save(temp_image_path)

        if y_coord - img_height < 50:
            c.showPage()
            y_coord = max_height

        c.drawImage(temp_image_path, x_coord, y_coord - img_height)

        os.remove(temp_image_path)

        x_coord += 110

    c.showPage()
    c.setPageSize(landscape(letter))

    # Metadata preview
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, 530, "Metadata preview")
    print("Loading data for metadata preview...")
  
    df = pd.read_csv(metadata_path)
    print("Data loaded!")
    preview_df = df.head(20)
    print("Preview dataframe created!")

    keys = [key for key in ["FileName", "photo_identifier", "GPSDateTime", "SubSecDateTimeOriginal", "GPSLatitude", "GPSLongitude", "FileSize", "ImageHeight", "ImageWidth"] if key in preview_df]
    try:
        preview_df = preview_df[keys]
    except KeyError:
        print("[ERROR] No key to merge gps information in metadata.")

    print("Creation of the PDF table...")
    all_cols = list(df.columns)
    table_data = [list(preview_df.columns)] + preview_df.values.tolist()
    table = Table(table_data)

    table.setStyle([
        ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 1)),  # Header row text color (blue)
        ('FONTSIZE', (0, 1), (-1, -1), 8), # Font size of all cells
        ('BACKGROUND', (0, 0), (-1, 0), (0.7, 0.7, 0.7)),  # Header row background color (gray)
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center align all cells
        ('INNERGRID', (0, 0), (-1, -1), 0.25, (0, 0, 0)),  # Inner gridlines
        ('BOX', (0, 0), (-1, -1), 0.25, (0, 0, 0)),  # Cell borders
    ])

    table.wrapOn(c, 10, 20)
    table.drawOn(c, 30, 100)
    print("PDF table sucessfully created!")
    
    text = c.beginText(30, 70)
    text.setFont('Courier', 8)
    line = f"All metadata columns names: {all_cols}"
    wraped_text = "\n".join(wrap(line, 160))
    text.textLines(wraped_text)
    c.drawText(text)
    
    c.setFont("Courier", 8)
    c.drawString(30, 80, f"Total images: {len(df)}")

    c.save()

    # Create predictions images and pdf
    img_folder_predictions_path = create_predictions_map(prediction_gps_path, classes)
    # Can be None if no predictions in csv file (all images useless)
    if img_folder_predictions_path:
        pdf_predictions_path = Path(img_folder_predictions_path, "temp.pdf")
        pred_images_path = [img_name for img_name in sorted(list(img_folder_predictions_path.iterdir())) if img_name.suffix.lower() == ".jpg"]

        images = [ Image.open(f) for f in pred_images_path ]
        images[0].save(
            pdf_predictions_path, "PDF" ,resolution=200.0, save_all=True, append_images=images[1:]
        )

        # Concat with trajectory pdf
        merger = PdfMerger()
        for pdf in [pdf_file, pdf_predictions_path]:
            merger.append(pdf)
        merger.write(pdf_file)
        merger.close()

    print("PDF created!")