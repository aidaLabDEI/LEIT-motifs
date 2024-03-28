from functions import pmotif_find2, find_all_occur, relative_contrast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Starting...")

    df = pd.read_csv("https://zenodo.org/record/4328047/files/toy.csv?download=1")
    ts_test = df.to_numpy()

    # Set parameters
    window_size = 30
    lsh_threshold = 0.30
    projection_iter = 3
    k = 3
    proj= 2
    colors = ["red", "green", "blue", "pink", "cyan", "yellow", "orange", "gray", "purple", "gray", "hotpink", "lime"]


    motifs, distance_comp = pmotif_find2(ts_test, window_size, projection_iter, k, proj, 40,lsh_threshold, 40, 16)
    print(distance_comp)
    motifs = motifs.queue
    #motifs = find_all_occur(ts_test, motifs, window_size)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    for i, dimension in enumerate(df.columns):
        axs[i].plot(df[dimension], label=dimension)
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel(f"Value - {dimension}")
        axs[i].legend()

        for idx, motif in enumerate(motifs):
            # Highlight the motifs in all dimensions
            for m in motif[1][1]:
                if i in motif[1][2][0]:
                    axs[i].axvspan(m, m + window_size, color=colors[idx], alpha=0.3)

    plt.suptitle("MultiDimensional Timeseries with Motifs Highlighted")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()