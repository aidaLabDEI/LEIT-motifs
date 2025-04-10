import os
import sys

sys.path.append("source")
from RP_GRAPH import pmotif_findg
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "external_dependencies"))
from data_loader import convert_tsf_to_dataframe
from base import create_shared_array
from scipy.signal import savgol_filter


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    paths = [
        os.path.join(current_dir, "..", "Datasets", "FOETAL_ECG.dat"),
        os.path.join(current_dir, "..", "Datasets", "evaporator.dat"),
        os.path.join(current_dir, "..", "Datasets", "RUTH.csv"),
        os.path.join(current_dir, "..", "Datasets", "oikolab_weather_dataset.tsf"),
        # os.path.join(current_dir, '..', 'Datasets', 'CLEAN_House1.csv'),
        # os.path.join(current_dir, "..", "Datasets", "whales.parquet"),
        # os.path.join(current_dir, "..", "Datasets", "quake.parquet"),
    ]
    dataset_names = [
        "potentials",
        "evaporator",
        "RUTH",
        "weather",
        "el_load",
        "whales",
        "quake",
    ]
    r_vals_computed = [8, 8, 16, 32, 8, 16, 8]
    windows = [50, 75, 500, 5000, 1000, 200, 100]
    dimensionality = [8, 2, 4, 2, 6, 4, 2]
    deltas = [0.5, 0.8]  # [0.01, 0.1, 0.2]
    recall_results = pd.DataFrame(columns=["Dataset", "delta", "recall", "mare"])

    for number, path in enumerate(paths):
        number_r = number
        # Load the dataset
        if number_r == 3:
            data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(path, 0)
            d = np.array(
                [data.loc[i, "series_value"].to_numpy() for i in range(data.shape[0])],
                order="C",
                dtype=np.float32,
            ).T
            # Apply a savgol filter to the data
            d = savgol_filter(d, 300, 1, axis=0)
        elif number_r == 4:
            data = pd.read_csv(path)
            data = data.drop(["Time", "Unix", "Issues"], axis=1)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
        elif number_r == 2:
            data = pd.read_csv(path)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
            # d += np.random.normal(0, 0.1, d.shape)
        elif number_r == 5 or number_r == 6:
            data = pd.read_parquet(path)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
            if number_r == 6:
                # fill nan values with the mean
                d = np.nan_to_num(d, nan=np.nanmean(d))
            else:
                d = d.T
            d += np.random.normal(0, 0.01, d.shape)
        else:
            data = pd.read_csv(path, sep=r"\s+")
            data = data.drop(data.columns[[0]], axis=1)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
        dimensions = d.shape[1]
        n = d.shape[0]
        shm_ts, ts = create_shared_array((n, dimensions), np.float32)
        ts[:] = d[:]
        del d

        for delta in deltas:
            results = []
            for _ in range(3):
                motifs, num_dist, _ = pmotif_findg(
                    shm_ts.name,
                    n,
                    dimensions,
                    windows[number_r],
                    1,
                    dimensionality[number_r],
                    r_vals_computed[number_r],
                    0.5,
                    100,
                    8,
                    delta,
                )
                results.append(motifs[0][0])
            # print(results)
            min_true = np.min(results)
            # Count the recall
            mare = np.mean((results-min_true)/min_true)
            recall_val = np.sum((np.array(results) - min_true) < 0.1) / len(results)
            recall_results = recall_results._append(
                {
                    "Dataset": dataset_names[number_r],
                    "delta": delta,
                    "recall": recall_val,
                    "mare": mare,  
                },
                ignore_index=True,
            )

    recall_results.to_csv("recall_results.csv", index=False, header=False, mode="a")
