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
    
    dataframe = pd.DataFrame(columns=["Dataset", "Noise", "Motif"])

    for number_r in range(4):
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

        noise_dims = [4, 16, 32, 128, 256]
        collected = []
        for noise_dim in noise_dims:
            d_temp = np.concatenate((d, np.random.normal(0,0.01, (d.shape[0], noise_dim))), axis=1)
            print(d_temp.shape)
            dimensions = d_temp.shape[1]
            n = d_temp.shape[0]
            shm_ts, ts = create_shared_array((n, dimensions), np.float32)
            ts[:] = d_temp[:]
            del d_temp

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

            collected.append(motifs[0][1][1])

        for noise_dim, motifs in zip(noise_dims, collected):
            # If the motif overlaps with the other collected ones, we put a 1
            res = 0
            for other in collected:
                if (motifs[0] < other[0] + windows[number_r] or motifs[0] > other[0] - windows[number_r] or
                    motifs[1] < other[1] + windows[number_r] or motifs[1] > other[1] - windows[number_r]):
                    res = 1
                    break

            dataframe = dataframe._append(
                    {
                        "Dataset": number_r,
                        "Noise": noise_dim,
                        "Motif": res,
                    },
                    ignore_index=True,
                )


    dataframe.to_csv("Results/noise.csv", index=False)