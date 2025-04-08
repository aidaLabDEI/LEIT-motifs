import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "external_dependecies"))
sys.path.append("source")
# from RP_MH import pmotif_find2
# from RP_DC import pmotif_find3
from RP_GRAPH import pmotif_findg
from RP_GRAPH_MULTI import pmotif_findg_multi

# from RPG_CF import pmotif_findauto
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
    if len(sys.argv) < 6:
        print(
            "Usage: python test.py <dataset> <window_size> <ranged search 0/1> <range_low_motif_dimensionality> <optional><range_high_motif_dimensionality> <K> <L> <device>"
        )
        sys.exit(1)
    dataset = int(sys.argv[1])
    window_size = int(sys.argv[2])
    ranged = int(sys.argv[3])
    if ranged:
        dimensionality = (int(sys.argv[4]), int(sys.argv[5]))
        K = int(sys.argv[6])
        L = int(sys.argv[7])
        if len(sys.argv) == 9:
            device = int(sys.argv[8])
        else:
            device = 0
    else:
        dimensionality = int(sys.argv[4])
        K = int(sys.argv[5])
        L = int(sys.argv[6])
        if len(sys.argv) == 8:
            device = int(sys.argv[7])
        else:
            device = 0

    paths = [
        "Datasets/FOETAL_ECG.dat",              #0
        "Datasets/evaporator.dat",              #1
        "Datasets/oikolab_weather_dataset.tsf", #2
        "Datasets/RUTH.csv",                    #3
        "Datasets/CLEAN_House1.csv",            #4
        "Datasets/whales.parquet",              #5
        "Datasets/quake.parquet",               #6
        "Datasets/steamgen.csv",                #7
        "Datasets/FL010",                       #8
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
    r = 4  # find_width_discr(d, window_size, K)
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
    # Find the motifs
    # for i in range(5):
    if ranged:
        motifs, num_dist, hash_t = pmotif_findg_multi(
            shm_ts.name, n, dimensions, window_size, 1, dimensionality, r, thresh, L, K
        )
    else:
        motifs, num_dist, hash_t = pmotif_findg(
            shm_ts.name,
            n,
            dimensions,
            window_size,
            1,
            dimensionality,
            r,
            thresh,
            L,
            K,
            0.2
        )

    end = time.perf_counter() - start
    print("Time elapsed: ", end, "of which", hash_t, "for hashing")
    print("Distance computations:", num_dist)
    # size, peak = tracemalloc.get_traced_memory()
    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics("lineno")

    # print(f"Current memory usage is {size / 10**6}MB; Peak was {peak / 10**6}MB")
    with open("results.txt", "a") as f:
        f.write(
            f"Time elapsed: {end} of which {hash_t} for hashing\nDistance computations: {num_dist}\n"
        )
    # for stat in top_stats[:10]:
    #   print(stat)

    # Plot
    # motifs = queue.PriorityQueue()
    # print(motifs)
    for motif in motifs:
        print(motif[0])
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
        if ranged:
            for j, dim_mot in enumerate(motifs):
                for idx, motif in enumerate(dim_mot):
                    # Highlight the motifs in all dimensions
                    for m in motif[1][1]:
                        if i in motif[1][2]:
                            axs[i].plot(
                                X[dimension].iloc[m : m + window_size],
                                color=colors[(idx + j) % len(colors)],
                                linewidth=1.8,
                                alpha=0.7,
                            )
        else:
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
