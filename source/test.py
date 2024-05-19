from RP_MH import pmotif_find2
import time, sys, pandas as pd, numpy as np, queue
sys.path.append('external_dependecies')
from data_loader import convert_tsf_to_dataframe
from base import z_normalized_euclidean_distance, find_all_occur, relative_contrast
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # Get from command line arguments the number of the dataset to be used, the window size, dimensionality, K and L
    # 0: FOETAL_ECG.dat
    # 1: evaporator.dat
    # 2: oikolab_weather_dataset.tsf
    if len(sys.argv) < 6:
        print("Usage: python test.py <dataset> <window_size> <dimensionality_motif> <K> <L>")
        sys.exit(1)
    dataset = int(sys.argv[1])
    window_size = int(sys.argv[2])
    dimensionality = int(sys.argv[3])
    K = int(sys.argv[4])
    L = int(sys.argv[5])

    paths = ["Datasets\FOETAL_ECG.dat", "Datasets\evaporator.dat", "Datasets\oikolab_weather_dataset.tsf"]
    d = None

    # Load the dataset
    if dataset == 2:
        data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(paths[2], 0)
        d = np.array([data.loc[i,"series_value"].to_numpy() for i in range(data.shape[0])]).T
    else:
        data = pd.read_csv(paths[dataset], delim_whitespace= True)
        data = data.drop(data.columns[[0]],axis=1)
        d = data.to_numpy()
    

    
    thresh = 0.5#dimensionality/d.shape[1]
    # Start the timer
    start = time.process_time()
    
    # Find the motifs
    motifs, num_dist = pmotif_find2(d, window_size, 0, 1, dimensionality, 8, thresh, L, K)


    end = (time.process_time() - start)
    print("Time elapsed: ", end)
    print("Distance computations:", num_dist)
    
    # Plot
    #motifs = queue.PriorityQueue()
    copy = motifs.queue
    motifs = copy
    #motifs = find_all_occur(extract, motifs, window_size)
    colors = ["red", "green", "blue", "pink", "cyan", "yellow", "orange", "gray", "purple"]
    fig, axs = plt.subplots(8, 1, figsize=(12, 8))
    X = pd.DataFrame(d)
    for i, dimension in enumerate(X.columns):
        axs[i].plot(X[dimension], label=dimension)
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


    # Compute relative contrast 
    # rc1= relative_contrast(d, motifs[0][1][1], window_size)
