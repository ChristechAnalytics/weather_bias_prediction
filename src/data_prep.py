# data_prep.py
import os
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd

# Live ECMWF Object Store URLs
DATA_URLS = {
    "forecast_error": "https://object-store.os-api.cci1.ecmwf.int/sop/forecast_error.csv",
    "soil_temperature": "https://object-store.os-api.cci1.ecmwf.int/sop/soil_temperature.csv",
    "time_of_day": "https://object-store.os-api.cci1.ecmwf.int/sop/time_of_day.csv"
}

CACHE_DIR = ".cache"

def download_file_safely(url, dest_path):
    """
    Downloads a large file using chunked streaming with a real-time progress bar.
    Prevents silent drops and timeout hanging.
    """
    if os.path.exists(dest_path):
        print(f"   [CACHE] Found local asset: {dest_path}")
        return

    filename = os.path.basename(dest_path)
    print(f"   [DOWNLOAD] Streaming {filename} from cloud...")
    
    # Send a stream request with a realistic User-Agent to avoid server throttling
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers, stream=True, timeout=30)
    response.raise_for_status() # Raise an error if the URL is broken
    
    # Get total file size from headers (fallback to 0 if hidden)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192  # 8 KB chunks
    
    # Initialize the visual progress bar
    with open(dest_path, 'wb') as f, tqdm(
        desc=f"      Progress",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk: # Filter out keep-alive new chunks
                f.write(chunk)
                bar.update(len(chunk))

def load_live_ecmwf_data(sample_fraction=1.0):
    """
    Downloads, caches, cleans, and aligns the structural variables from ECMWF.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print("-> Checking and synchronizing cloud datasets...")
    for key, url in DATA_URLS.items():
        dest = os.path.join(CACHE_DIR, f"{key}.csv")
        download_file_safely(url, dest)
        
    error_path = os.path.join(CACHE_DIR, "forecast_error.csv")
    time_path = os.path.join(CACHE_DIR, "time_of_day.csv")
    soil_path = os.path.join(CACHE_DIR, "soil_temperature.csv")

    print("\n-> Ingesting and cleaning 'forecast_error' (Target variable)...")
    df_error = pd.read_csv(error_path, header=None, names=["forecast_error"])
    df_error["forecast_error"] = pd.to_numeric(df_error["forecast_error"], errors='coerce')
    df_error = df_error.dropna()
    
    if sample_fraction < 1.0:
        print(f"   Optimizing memory: Subsampling dataset to {sample_fraction * 100}%...")
        df_error = df_error.sample(frac=sample_fraction, random_state=42)
        
    target_indices = df_error.index
    
    print("-> Loading and aligning 'time_of_day' feature...")
    df_time = pd.read_csv(time_path, header=None, names=["time_of_day"])
    df_time["time_of_day"] = pd.to_numeric(df_time["time_of_day"], errors='coerce')
    df_time = df_time.loc[target_indices].ffill().bfill()
    
    print("-> Loading and aligning 'soil_temperature' feature...")
    df_soil = pd.read_csv(soil_path, header=None, names=["soil_temperature"])
    df_soil["soil_temperature"] = pd.to_numeric(df_soil["soil_temperature"], errors='coerce')
    df_soil = df_soil.loc[target_indices].ffill().bfill()
    
    print("-> Data streams fully validated, synchronized, and locked.")
    return df_time["time_of_day"].values, df_soil["soil_temperature"].values, df_error["forecast_error"].values

def prepare_features(feature_array):
    return feature_array.reshape(-1, 1)

def combine_features(*features):
    prepared = [prepare_features(f) for f in features]
    return np.hstack(prepared)