import os
import sys

sys.path.append("source")
from RP_GRAPH import pmotif_findg
import time
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "external_dependencies"))
from data_loader import convert_tsf_to_dataframe
from base import create_shared_array
from scipy.signal import savgol_filter
import gc


def main():
    """
    Main function that performs an extended test on multiple datasets.
    It measures the time elapsed, relative contrast, and other parameters for each dataset.
    It also performs tests for different values of K, L, r, failure probability, and noise dimensions.
    """

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

    r_vals_computed = [4, 8, 16, 32, 8, 16, 8]
    windows = [50, 75, 500, 5000, 1000, 200, 100]
    dimensionality = [8, 2, 4, 2, 6, 4, 2]

    K_results = pd.DataFrame(
        columns=[
            "Dataset",
            "K",
            "Time elapsed",
            "dist_computed",
        ]
    )
    L_results = pd.DataFrame(
        columns=[
            "Dataset",
            "L",
            "Time elapsed",
            "Time int",
            "dist_computed",
        ]
    )
    R_results = pd.DataFrame(
        columns=[
            "Dataset",
            "r",
            "Time elapsed",
            "dist_computed",
        ]
    )

    # Base test for time elapsed
    for number, path in enumerate(paths):
        number_r = number
        results = pd.DataFrame(
            columns=[
                "Dataset",
                "Time elapsed",
                "RC1",
                "K",
                "L",
                "w",
                "r",
                "dist_computed",
                "Time int",
            ]
        )

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
            d += np.random.normal(0, 0.1, d.shape)
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

        if number_r == 0:
            # lauch a computation just to compile numba
            pmotif_findg(shm_ts.name, n, dimensions, 50, 1, 8, 8, 0, 10, 8)
        print("Starting")
        start = time.perf_counter()
        for i in range(1):
            if number_r == 6:
                motifs, num_dist, _ = pmotif_findg(
                    shm_ts.name,
                    n,
                    dimensions,
                    windows[number_r],
                    1,
                    dimensionality[number_r],
                    r_vals_computed[number_r],
                    0.5,
                    50,
                    8,
                )
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
        end = (time.perf_counter() - start) / 1
        motifs = motifs

        rel_cont = 0  # relative_contrast(d, motifs[0][1][1], windows[number])
        temp_df = pd.DataFrame(
            [
                {
                    "Dataset": number_r,
                    "Time elapsed": end,
                    "RC1": rel_cont,
                    "K": 8,
                    "L": 200,
                    "w": windows[number_r],
                    "r": r_vals_computed[number_r],
                    "dist_computed": num_dist,
                }
            ]
        )
        results = results._append(temp_df, ignore_index=True)
        results.to_csv("p1" + str(number_r) + ".csv", index=False)
        gc.collect()

        Ks = [4, 8, 12, 16]
        Ls = [10, 50, 100, 150, 200, 400]
        rs = [4, 8, 16, 32]

        # Testing on hashing

        for K in Ks:
            start = time.perf_counter()
            for i in range(1):
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
                    K,
                )
            end = time.perf_counter() - start
            temp_df = pd.DataFrame(
                [
                    {
                        "Dataset": number_r,
                        "Time elapsed": end,
                        "RC1": 0,
                        "K": K,
                        "L": 200,
                        "w": windows[number_r],
                        "r": r_vals_computed[number_r],
                        "dist_computed": num_dist,
                    }
                ]
            )
            results = results._append(temp_df, ignore_index=True)

            K_results = K_results._append(
                {
                    "Dataset": number_r,
                    "K": K,
                    "Time elapsed": end,
                    "dist_computed": num_dist,
                },
                ignore_index=True,
            )
            gc.collect()

        print("K fin")

        for L in Ls:
            start = time.perf_counter()
            for i in range(1):
                motifs, num_dist, ht = pmotif_findg(
                    shm_ts.name,
                    n,
                    dimensions,
                    windows[number_r],
                    1,
                    dimensionality[number_r],
                    r_vals_computed[number_r],
                    0.5,
                    L,
                    8,
                )
            end = time.perf_counter() - start
            temp_df = pd.DataFrame(
                [
                    {
                        "Dataset": number_r,
                        "Time elapsed": end,
                        "Time int": ht,
                        "RC1": 0,
                        "K": 8,
                        "L": L,
                        "w": windows[number_r],
                        "r": r_vals_computed[number_r],
                        "dist_computed": num_dist,
                    }
                ]
            )
            results = results._append(temp_df, ignore_index=True)

            L_results = L_results._append(
                {
                    "Dataset": number_r,
                    "L": L,
                    "Time elapsed": end,
                    "Time int": ht,
                    "dist_computed": num_dist,
                },
                ignore_index=True,
            )
            gc.collect()
        print("L fin")
        results.to_csv("r_partial_dataset" + str(number_r) + ".csv", index=False)

        for _, r in rs:
            start = time.perf_counter()
            for i in range(1):
                motifs, num_dist, _ = pmotif_findg(
                    shm_ts.name,
                    n,
                    dimensions,
                    windows[number_r],
                    1,
                    dimensionality[number_r],
                    r,
                    0.5,
                    200,
                    8,
                )
            end = time.perf_counter() - start
            temp_df = pd.DataFrame(
                [
                    {
                        "Dataset": number_r,
                        "Time elapsed": end,
                        "RC1": 0,
                        "K": 8,
                        "L": 200,
                        "w": windows[number_r],
                        "r": r,
                        "dist_computed": num_dist,
                    }
                ]
            )
            results = results._append(temp_df, ignore_index=True)

            R_results = R_results._append(
                {
                    "Dataset": number_r,
                    "r": r,
                    "Time elapsed": end,
                    "dist_computed": num_dist,
                },
                ignore_index=True,
            )
            gc.collect()

        results.to_csv("r_dataset" + str(number_r) + ".csv", index=False)
        print("Dataset", number_r, "finished")
        shm_ts.unlink()

    K_results.to_csv("Results/K_results.csv", index=False)
    L_results.to_csv("Results/L_results.csv", index=False)
    R_results.to_csv("Results/R_results.csv", index=False)

    r"""
    # Mem test
    results = pd.DataFrame(columns=["Dataset", "Mem"])
    for number, path in enumerate(paths):
        number_r = number + 5

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
            d += np.random.normal(0, 0.1, d.shape)
        elif number_r == 5 or number_r == 6:
            data = pd.read_parquet(path)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
            if number_r == 6:
                # fill nan values with the mean
                d = np.nan_to_num(d, nan=np.nanmean(d))
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

        tracemalloc.start()
        _, _, _ = pmotif_findg(
            shm_ts.name,
            n,
            dimensions,
            windows[number_r],
            1,
            dimensionality[number_r],
            r_vals_computed[number_r],
            0.5,
            50,  # 200,
            8,
        )
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        temp_df = pd.DataFrame([{"Dataset": number_r, "Mem": peak}])
        results = results._append(temp_df, ignore_index=True)

    results.to_csv("Mem_results.csv", index=False)
    """


if __name__ == "__main__":
    # from multiprocessing import freeze_support
    # freeze_support()
    main()
