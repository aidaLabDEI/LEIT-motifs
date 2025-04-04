import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "external_dependecies"))
sys.path.append("source")
from random_selection import RP

import time
import pandas as pd
import numpy as np
import wfdb
from data_loader import convert_tsf_to_dataframe
from base import create_shared_array

# from extra import relative_contrast
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter

if __name__ == "__main__":
    matplotlib.use("WebAgg")

    # Get from command line arguments the number of the dataset to be used, the window size, dimensionality, K and L
    # 0: FOETAL_ECG.dat
    # 1: evaporator.dat
    # 2: oikolab_weather_dataset.tsf
    # 3: RUTH.csv
    if len(sys.argv) < 7:
        print(
            "Usage: python rs_test.py <dataset> <window_size> <motif_dimensionality> <alpha> <c> <device>"
        )
        sys.exit(1)
    dataset = int(sys.argv[1])
    window_size = int(sys.argv[2])
    dimensionality = int(sys.argv[3])
    alpha = int(sys.argv[4])
    c = int(sys.argv[5])
    device = int(sys.argv[6])

    paths = [
        "Datasets/FOETAL_ECG.dat",
        "Datasets/evaporator.dat",
        "Datasets/oikolab_weather_dataset.tsf",
        "Datasets/RUTH.csv",
        "Datasets/CLEAN_House1.csv",
        "Datasets/whales.parquet",
        "Datasets/quake.parquet",
        "Datasets/steamgen.csv",
        "Datasets/FL010",
    ]
    d = None

    # Load the dataset
    if dataset == 2:
        data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(paths[2], 0)
        d = np.array(
            [data.loc[i, "series_value"].to_numpy() for i in range(data.shape[0])],
            order="C",
            dtype=np.float32,
        ).T
        # Apply a savgol filter to the data
        d = savgol_filter(d, 300, 1, axis=0)
    elif dataset == 4:
        data = pd.read_csv(paths[dataset])
        data = data.drop(["Time", "Unix", "Issues"], axis=1)
        d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
        # Add some noise to remove step-like patterns
        d += np.random.normal(0, 0.1, d.shape)
    elif dataset == 3 or dataset == 7:
        data = pd.read_csv(paths[dataset], dtype=np.float32)
        d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
        # if dataset != 3:
        # Add some noise to remove step-like patterns
        d += np.random.normal(0, 0.01, d.shape)
    elif dataset == 5 or dataset == 6:
        data = pd.read_parquet(paths[dataset])
        d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
        if dataset == 6:
            # fill nan values with the mean
            d = np.nan_to_num(d, nan=np.nanmean(d))
        else:
            d = d.T
        d += np.random.normal(0, 0.01, d.shape)
    elif dataset == 8:
        d, data = wfdb.rdsamp(paths[dataset])
        d += np.random.normal(0, 0.01, d.shape)
    else:
        data = pd.read_csv(paths[dataset], sep=r"\s+")
        data = data.drop(data.columns[[0]], axis=1)
        d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
    del data
    r = 8  # find_width_discr(d, window_size, K)
    # d = np.concatenate((d, np.random.normal(0,0.01, (d.shape[0], 4))), axis=1)
    print(d.shape)
    thresh = 0
    dimensions = d.shape[1]
    n = d.shape[0]
    shm_ts, ts = create_shared_array((n, dimensions), np.float32)
    ts[:] = d[:]
    # del d
    # Start the timer
    # tracemalloc.start()
    start = time.perf_counter()
    motifs, num_dist = RP(
        shm_ts.name, n, dimensions, window_size, dimensionality, 1, alpha, c, 100, 0.9
    )
    end = time.perf_counter() - start
    print("Time elapsed: ", end)
    print("Distance computations:", num_dist)
    # Plot
    # motifs = queue.PriorityQueue()
    print(motifs)

    copy = motifs
    motifs = copy
    # motifs = find_all_occur(extract, motifs, window_size)
    colors = [
        "crimson",
        "forestgreen",
        "deepskyblue",
        "pink",
        "cyan",
        "yellow",
        "orange",
        "gray",
        "purple",
    ]
    rng = np.random.default_rng(seed=42)
    fig, axs = plt.subplots(dimensions, 1, sharex=True, layout="constrained")
    X = pd.DataFrame(ts)
    for i, dimension in enumerate(X.columns):
        axs[i].plot(
            X[dimension], label=dimension, linewidth=1.2, color="lightgray"
        )  # "#6263e0")
        axs[i].set_axis_off()

        for idx, motif in enumerate(motifs):
            # Highlight the motifs in all dimensions it appears
            for m in motif[1][1]:
                if i in motif[1][2]:
                    axs[i].plot(
                        X[dimension].iloc[m : m + window_size],
                        color=colors[idx],
                        linewidth=1.8,
                        alpha=0.7,
                    )
    if device == 1:
        plt.savefig("motifs.svg", format="svg")
    else:
        plt.show()
        # Compute relative contrast
        # rc1= relative_contrast(d, window_size, dimensionality)
        # print("RC1:", rc1)

    # Free the shared memory
    shm_ts.close()
    shm_ts.unlink()
