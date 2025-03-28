import sys
import pandas as pd
import numpy as np
import os
import wfdb
import stumpy

sys.path.append("external_dependecies")
from data_loader import convert_tsf_to_dataframe
from scipy.signal import savgol_filter
from base import z_normalized_euclidean_distancegmulti

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    paths = [
        os.path.join(current_dir, "..", "Datasets", "FOETAL_ECG.dat"),
        os.path.join(current_dir, "..", "Datasets", "evaporator.dat"),
        os.path.join(current_dir, "..", "Datasets", "RUTH.csv"),
        os.path.join(current_dir, "..", "Datasets", "oikolab_weather_dataset.tsf"),
        os.path.join(current_dir, "..", "Datasets", "CLEAN_House1.csv"),
        os.path.join(current_dir, "..", "Datasets", "whales.parquet"),
        os.path.join(current_dir, "..", "Datasets", "quake.parquet"),
        os.path.join(current_dir, "..", "Datasets", "FL010"),
    ]

    windows = [50, 75, 500, 5000, 1000, 300, 100, 200]
    dimensionality = [8, 2, 4, 2, 3, 6, 3, 3]

    # Base test for time elapsed
    for number, path in enumerate(paths):
        number_r = number
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
            d = np.nan_to_num(d, nan=np.nanmean(d))
        elif number_r == 7:
            d, data = wfdb.rdsamp(path)
        else:
            data = pd.read_csv(path, sep=r"\s+")
            data = data.drop(data.columns[[0]], axis=1)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float64).T
        d = d.astype(np.float64)

        if d.shape[0] > d.shape[1]:
            d = d.T
        print(d.shape)
        # Cut to the first 10000 samples
        if d.shape[1] > 20000:
            num = np.random.randint(0, d.shape[1] - 20000)
            d = d[:, num : num + 20000]
        d += np.random.normal(0, 0.01, d.shape)
        # COmpute the matrix profile
        MP, I_prof = stumpy.mstump(d, m=windows[number_r])
        # Find the indices of the min at the dimensionality expressed in the variable and the dimensions of the min
        MP_dim = MP[dimensionality[number_r] - 1]
        I_dim = I_prof[dimensionality[number_r] - 1]
        # Find the indices of the min
        min_index = np.argmin(MP_dim)
        nn_index = I_dim[min_index]
        d = d.astype(np.float32)
        # Compute the distances for each dimension between the min and the nearest neighbor
        k_maxdist = z_normalized_euclidean_distancegmulti(
            d[:, min_index : min_index + windows[number_r]].T,
            d[:, nn_index : nn_index + windows[number_r]].T,
            np.mean(d[:, min_index : min_index + windows[number_r]], axis=1),
            np.std(d[:, min_index : min_index + windows[number_r]], axis=1),
            np.mean(d[:, nn_index : nn_index + windows[number_r]], axis=1),
            np.std(d[:, nn_index : nn_index + windows[number_r]], axis=1),
        )
        ordered_distances = k_maxdist[1]
        k_maxdist = ordered_distances[0]  # [dimensionality[number_r] - 1]

        # Find the n-th min
        n_min_index = np.argsort(MP_dim)[-1]
        nn_index = I_dim[n_min_index]

        # Compute the distances for each dimension between the min and the nearest neighbor
        n_maxdist = z_normalized_euclidean_distancegmulti(
            d[:, n_min_index : n_min_index + windows[number_r]].T,
            d[:, nn_index : nn_index + windows[number_r]].T,
            np.mean(d[:, n_min_index : n_min_index + windows[number_r]], axis=1),
            np.std(d[:, n_min_index : n_min_index + windows[number_r]], axis=1),
            np.mean(d[:, nn_index : nn_index + windows[number_r]], axis=1),
            np.std(d[:, nn_index : nn_index + windows[number_r]], axis=1),
        )
        ordered_distances = n_maxdist[1]
        n_maxdist = ordered_distances[0]  # [dimensionality[number_r] - 1]

        print("Dataset: ", number_r, "contrast:", (n_maxdist) / k_maxdist)
