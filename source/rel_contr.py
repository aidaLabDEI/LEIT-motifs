import sys
import pandas as pd
import numpy as np
import os
import wfdb
from itertools import product

sys.path.append("external_dependecies")
from data_loader import convert_tsf_to_dataframe
from scipy.signal import savgol_filter
from base import z_normalized_euclidean_distanceg

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    paths = [
        # os.path.join(current_dir, "..", "Datasets", "FOETAL_ECG.dat"),
        # os.path.join(current_dir, "..", "Datasets", "evaporator.dat"),
        # os.path.join(current_dir, "..", "Datasets", "RUTH.csv"),
        # os.path.join(current_dir, "..", "Datasets", "oikolab_weather_dataset.tsf"),
         os.path.join(current_dir, '..', 'Datasets', 'CLEAN_House1.csv'),
         os.path.join(current_dir, "..", "Datasets", "whales.parquet"),
         os.path.join(current_dir, "..", "Datasets", "quake.parquet"),
         os.path.join(current_dir, "..", "Datasets", "FL010")
    ]

    windows = [50, 75, 500, 5000, 1000, 300, 100]
    dimensionality = [8,2,4,2,6,4,5,3]

    # Base test for time elapsed
    for number, path in enumerate(paths):
        number_r = number + 4
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
        elif number_r == 7:
            d, data = wfdb.rdsamp(path)
        else:
            data = pd.read_csv(path, sep=r"\s+")
            data = data.drop(data.columns[[0]], axis=1)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32).T
        d = d.astype(np.float32)
        rnd = np.random.default_rng()
        indices_1 = rnd.integers(0, d.shape[0]-windows[number_r]+1, (10000))
        indices_2 = rnd.integers(0, d.shape[0]-windows[number_r]+1, (10000))
        distances = []
        for i,j in product(indices_1, indices_2):
            distances.append(
                z_normalized_euclidean_distanceg(
                    d[i:i+windows[number_r]], d[j:j+windows[number_r]], np.arange(d.shape[1], dtype=np.int32), np.mean(d[i:i+windows[number_r]], axis=1), np.std(d[i:i+windows[number_r]], axis=1),
                    np.mean(d[j:j+windows[number_r]], axis=1), np.std(d[j:j+windows[number_r]], axis=1), dimensionality[number_r]
                )
            )
        print(f'Dataset {number_r}:', np.mean(distances))
        
