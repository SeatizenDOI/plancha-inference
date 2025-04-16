import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from geocube.api.core import make_geocube
from math import radians, cos, sin, sqrt, atan2

from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def haversine(point1, point2):
    """Calculate the Haversine distance between two geographic points."""
    R = 6371000  # Earth radius in meters
    lat1, lon1 = point1
    lat2, lon2 = point2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def compute_grid_value(predictions_csv):
    """Prepare gridded data by processing a dataframe from CSV and calculating grid_value."""
    
    distances = []
    for i in range(len(predictions_csv) - 1):
        p1 = (predictions_csv.iloc[i]['GPSLatitude'], predictions_csv.iloc[i]['GPSLongitude'])
        p2 = (predictions_csv.iloc[i+1]['GPSLatitude'], predictions_csv.iloc[i+1]['GPSLongitude'])
        distances.append(haversine(p1, p2))

    median_within = np.median(distances)

    return median_within

def prepare_gridded_data(predictions_csv, target_class, grid_value, interpolation_method):
    def calculate_degree_spacing(meters, avg_latitude):
        # Conversion factor from meters to degrees (approximate)
        meters_per_degree = 111319

        # Convert meters to degrees for latitude
        degrees_latitude = meters / meters_per_degree

        # Convert latitude from degrees to radians
        avg_latitude_rad = np.radians(avg_latitude)

        # Convert meters to degrees for longitude
        degrees_longitude = meters / (meters_per_degree * np.cos(avg_latitude_rad))

        return degrees_latitude, degrees_longitude
    # Convert grid_value, which is in meters, to degrees of latitude and longitude
    latitude_spacing, longitude_spacing = calculate_degree_spacing(grid_value, predictions_csv['GPSLatitude'].mean())
    # Extract relevant columns
    df = predictions_csv[['GPSLongitude', 'GPSLatitude', target_class]]
    # Assuming 'GPSLongitude' and 'GPSLatitude' are already in decimal degrees.
    # Creating gridded data using np.meshgrid
    grid_x, grid_y = np.meshgrid(
        np.arange(df['GPSLongitude'].min(), df['GPSLongitude'].max(), longitude_spacing),
        np.arange(df['GPSLatitude'].min(), df['GPSLatitude'].max(), latitude_spacing)
    )
    
    # Perform linear interpolation using griddata
    points = df[['GPSLongitude', 'GPSLatitude']].values
    values = df[target_class].values
    grid_z = griddata(points, values, (grid_x, grid_y), method=interpolation_method)
    
    # Flatten the grid for DataFrame creation
    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()
    grid_z_flat = grid_z.ravel()
    
    # Create a DataFrame from the gridded data
    df_gridded = pd.DataFrame({
        'GPSLongitude': grid_x_flat,
        'GPSLatitude': grid_y_flat,
        target_class: grid_z_flat
    })

    # Compute the convex hull for the original points
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])
    
    # Mask the gridded data based on convex hull
    mask = np.array([hull_path.contains_point(i) for i in zip(df_gridded['GPSLongitude'], df_gridded['GPSLatitude'])])
    df_gridded.loc[~mask, target_class] = np.nan  # Apply mask

    # Remove NaN values that result from the interpolation step
    df_gridded = df_gridded.dropna(subset=[target_class])

    return df_gridded, latitude_spacing, longitude_spacing

def create_rasters_for_classes(predictions_csv_path, classes, output_path, sessiontag, interpolation_method):

    predictions_csv = pd.read_csv(predictions_csv_path)
    if len(predictions_csv) == 0:
        print("[ERROR] No predictions.")
        return None
    
    if "GPSLongitude" not in predictions_csv or "GPSLatitude" not in predictions_csv: 
        print("[ERROR] No GPS coordinate.")
        return None
    
    if round(predictions_csv["GPSLatitude"].std(), 10) == 0.0 or round(predictions_csv["GPSLongitude"].std(), 10) == 0.0: 
        print("[ERROR] All frames have the same gps coordinate.")
        return None

    # Assuming compute_grid_value and prepare_gridded_data are already defined and correct
    grid_value = compute_grid_value(predictions_csv)

    if grid_value == 0.0:
        print("[ERROR] Something occurs during computing grid value. Mission is not a polygon.")
        return None

    for target_class in tqdm(classes):
        df_gridded, lat_spacing, lon_spacing = prepare_gridded_data(predictions_csv, target_class, grid_value, interpolation_method)
        gdf = gpd.GeoDataFrame(df_gridded, geometry=gpd.points_from_xy(df_gridded['GPSLongitude'], df_gridded['GPSLatitude']), crs='EPSG:4326')
        # Calculate initial resolution based on median distances
        resol =  np.max([lat_spacing, lon_spacing])
        cube = make_geocube(vector_data=gdf, resolution=(-resol, resol))
        raster_path = os.path.join(output_path, f"{sessiontag}_{target_class.replace('/', '_')}_raster.tif")
        cube[target_class].rio.to_raster(raster_path)