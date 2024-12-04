import stumpy
import time
import sys
import pandas as pd
import numpy as np
import os
import tracemalloc
import wfdb

sys.path.append("external_dependecies")
from data_loader import convert_tsf_to_dataframe
from scipy.signal import savgol_filter

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    paths = [
        # os.path.join(current_dir, "..", "Datasets", "FOETAL_ECG.dat"),
        # os.path.join(current_dir, "..", "Datasets", "evaporator.dat"),
        # os.path.join(current_dir, "..", "Datasets", "RUTH.csv"),
        # os.path.join(current_dir, "..", "Datasets", "oikolab_weather_dataset.tsf"),
        # os.path.join(current_dir, '..', 'Datasets', 'CLEAN_House1.csv'),
        # os.path.join(current_dir, "..", "Datasets", "whales.parquet"),
        # os.path.join(current_dir, "..", "Datasets", "quake.parquet"),
         os.path.join(current_dir, "..", "Datasets", "FL010")
    ]

    windows = [50, 75, 500, 5000, 1000, 300, 100]

    # Base test for time elapsed
    for number, path in enumerate(paths):
        number_r = number + 7
        # Load the dataset
        if number_r == 3:
            data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(path, 0)
            d = np.array(
                [data.loc[i, "series_value"].to_numpy() for i in range(data.shape[0])],
                order="C",
                dtype=np.float64,
            ).T
            # Apply a savgol filter to the data
            d = savgol_filter(d, 300, 1, axis=0)
        elif number_r == 4:
            data = pd.read_csv(path)
            data = data.drop(["Time", "Unix", "Issues"], axis=1)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float64)
        elif number_r == 2:
            data = pd.read_csv(path)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float64)
        elif number_r == 5 or number_r == 6:
            data = pd.read_parquet(path)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float64)
        elif number_r == 8:
            d, data = wfdb.rdsamp(paths[number_r])
        else:
            data = pd.read_csv(path, sep=r"\s+")
            data = data.drop(data.columns[[0]], axis=1)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float64)
            
        tracemalloc.start()
        for i in [5000, 10000,50000]:
            d_temp = d[:i,:]
            print(d_temp.shape)
            start = time.perf_counter()
            m = stumpy.mstump(d_temp.T, windows[number])
            #
            end = time.perf_counter() - start
            size, peak = tracemalloc.get_traced_memory()
            print("Dataset", number, "time elapsed:", end, "seconds")
            print(
                f"Current memory usage is {size / 10**6}MB; Peak was {peak / 10**6}MB"
            )
