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
import matplotlib.pyplot as plt
import matplotlib


def generate_random_walk(d, n):
    """
    Generates a multidimensional timeseries of dimensions d and length n of random walks.

    Args:
        d: The number of dimensions.
        n: The length of the timeseries.

    Returns:
        A numpy array of shape (n, d) representing the random walk timeseries.
    """
    # Initialize the timeseries with zeros
    timeseries = np.zeros((n, d))

    # Simulate the random walk for each dimension
    for i in range(d):
        # Generate random steps between -1 and 1
        steps = np.random.choice([-1, 1], size=n)
        weight = np.random.rand(n) * 5
        steps = steps * weight

        # Perform the random walk accumulation
        timeseries[:, i] = np.cumsum(steps)

    return timeseries


if __name__ == "__main__":
    matplotlib.use("WebAgg")
    number_r = 1

    current_dir = os.path.dirname(__file__)
    paths = [
        os.path.join(current_dir, "..", "Datasets", "FOETAL_ECG.dat"),
        os.path.join(current_dir, "..", "Datasets", "evaporator.dat"),
        os.path.join(current_dir, "..", "Datasets", "RUTH.csv"),
        os.path.join(current_dir, "..", "Datasets", "oikolab_weather_dataset.tsf"),
        os.path.join(current_dir, "..", "Datasets", "CLEAN_House1.csv"),
        os.path.join(current_dir, "..", "Datasets", "whales.parquet"),
        os.path.join(current_dir, "..", "Datasets", "quake.parquet"),
    ]

    r_vals_computed = [4, 8, 16, 32, 8, 16, 8]
    windows = [50, 75, 500, 5000, 1000, 200, 150]
    dimensionality = [8, 2, 4, 2, 6, 6, 3]

    path = paths[number_r]
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
        d = d[:100000, :]
    elif number_r == 2:
        data = pd.read_csv(path)
        d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
        d += np.random.normal(0, 0.1, d.shape)
    elif number_r == 5 or number_r == 6:
        data = pd.read_parquet(path)
        d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
        d += np.random.normal(0, 0.01, d.shape)
    else:
        data = pd.read_csv(path, sep=r"\s+")
        data = data.drop(data.columns[[0]], axis=1)
        d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
    noise_dim = 5
    d = np.concatenate((d, generate_random_walk(noise_dim, d.shape[0])), axis=1)
    dimensions = d.shape[1]
    n = d.shape[0]
    shm_ts, ts = create_shared_array((n, dimensions), np.float32)
    ts[:] = d[:]
    del d

    motifs, num_dist, _ = pmotif_findg(
        shm_ts.name,
        n,
        dimensions,
        windows[number_r],
        1,
        dimensionality[number_r],
        r_vals_computed[number_r],
        0.5,
        200,
        8,
    )
    colors = [
        "red",
        "green",
        "pink",
        "pink",
        "cyan",
        "yellow",
        "orange",
        "gray",
        "purple",
    ]

    fig, axs = plt.subplots(dimensions, 1, sharex=True)
    X = pd.DataFrame(ts)
    for i, dimension in enumerate(X.columns):
        axs[i].plot(X[dimension], label=dimension, linewidth=1.2, color="#6263e0")
        axs[i].set_axis_off()
        # axs[i].set_xlabel("Time")
        # axs[i].set_ylabel("Dimension " + str(dimension))
        # axs[i].legend()
        for idx, motif in enumerate(motifs):
            # Highlight the motifs in all dimensions
            for m in motif[1][1]:
                if i in motif[1][2][0]:
                    axs[i].plot(
                        X[dimension].iloc[m : m + windows[number_r]],
                        color=colors[idx],
                        linewidth=1.8,
                        alpha=0.7,
                    )
    plt.show()
