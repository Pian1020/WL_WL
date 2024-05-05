import zipfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from pylab import *
from datetime import datetime, timedelta

def createDataset(path):
    dfs_wind = []

    # Columns to be selected
    columns_to_select = [
        "Time and Date",
        "Wind Direction (deg) at 199m",
        "Horizontal Wind Speed (m/s) at 199m",
        "Vertical Wind Speed (m/s) at 199m",
        "Wind Direction (deg) at 179m",
        "Horizontal Wind Speed (m/s) at 179m",
        "Vertical Wind Speed (m/s) at 179m",
        "Wind Direction (deg) at 159m",
        "Horizontal Wind Speed (m/s) at 159m",
        "Vertical Wind Speed (m/s) at 159m",
        "Wind Direction (deg) at 139m",
        "Horizontal Wind Speed (m/s) at 139m",
        "Vertical Wind Speed (m/s) at 139m",
        "Wind Direction (deg) at 119m",
        "Horizontal Wind Speed (m/s) at 119m",
        "Vertical Wind Speed (m/s) at 119m",
        "Wind Direction (deg) at 99m",
        "Horizontal Wind Speed (m/s) at 99m",
        "Vertical Wind Speed (m/s) at 99m",
        "Wind Direction (deg) at 79m",
        "Horizontal Wind Speed (m/s) at 79m",
        "Vertical Wind Speed (m/s) at 79m",
        "Wind Direction (deg) at 49m",
        "Horizontal Wind Speed (m/s) at 49m",
        "Vertical Wind Speed (m/s) at 49m",
        "Wind Direction (deg) at 38m",
        "Horizontal Wind Speed (m/s) at 38m",
        "Vertical Wind Speed (m/s) at 38m",
        "Wind Direction (deg) at 29m",
        "Horizontal Wind Speed (m/s) at 29m",
        "Vertical Wind Speed (m/s) at 29m",
        "Wind Direction (deg) at 10m",
        "Horizontal Wind Speed (m/s) at 10m",
        "Vertical Wind Speed (m/s) at 10m"
    ]

    # loop that iterates over all the files in the directory
    for file in os.listdir(path):
        # checks if the current file ends with the ".zip"
        if file.endswith(".zip"):
            # Extract the file
            with zipfile.ZipFile(os.path.join(path, file), "r") as zip_ref:
                csv_filename = [f for f in zip_ref.namelist() if f.endswith('.CSV')][0]
                with zip_ref.open(csv_filename) as f:
                    # Read CSV file, skip the first row and select specified columns
                    df = pd.read_csv(f, skiprows=1, usecols=columns_to_select)

                    # Process each row to create separate DataFrames
                    for index, row in df.iterrows():
                        heights = [10, 29, 38, 49, 79, 99, 119, 139, 159, 179, 199]
                        wind_directions = [row[col] for col in df.columns if 'Wind Direction' in col]
                        horizontal_speeds = [row[col] for col in df.columns if 'Horizontal Wind Speed' in col]
                        vertical_speeds = [row[col] for col in df.columns if 'Vertical Wind Speed' in col]
                        datetime_str = row["Time and Date"]
                        datetime_obj = [datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")] * len(heights)
                        
                        # Create a DataFrame for each row
                        height_df = pd.DataFrame({
                            'Height': heights,
                            'Wind Direction': wind_directions,
                            'Horizontal Wind Speed': horizontal_speeds,
                            'Vertical Wind Speed': vertical_speeds,
                            'datetime': datetime_obj
                        })

                        # Append the DataFrame to the list
                        dfs_wind.append(height_df)
    # Convert the list of DataFrames to a single DataFrame
    df_multi = pd.concat(dfs_wind)
    
    return df_multi

if __name__ == "__main__":
    # Determine the file path
    root = os.getcwd()
    file_path = root + "/ProcessedData"

    # Read files from the disk and process the data
    dataset = createDataset(file_path)
    print(dataset)
