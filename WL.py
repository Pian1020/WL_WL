import pandas as pd
import zipfile
import os

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
                    dfs_wind.append(df)
    
    return dfs_wind

if __name__ == "__main__":
    # Determine the file path
    root = os.getcwd()
    file_path = root + "/ProcessedData"

    # Read files from the disk
    dfs_wind = createDataset(file_path)
    print(dfs_wind)
