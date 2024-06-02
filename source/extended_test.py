from RP_MH import pmotif_find2
from RP_DC import pmotif_find3
import time, sys, os, pandas as pd, numpy as np, queue
sys.path.append(os.path.join(os.path.dirname(__file__), 'external_dependencies'))
from data_loader import convert_tsf_to_dataframe
from base import z_normalized_euclidean_distance
from find_bin_width import find_width_discr
from extra import relative_contrast
import matplotlib.pyplot as plt


paths = [
    os.path.join(current_dir, '..', 'Datasets', 'FOETAL_ECG.dat'),
    os.path.join(current_dir, '..', 'Datasets', 'evaporator.dat'),
    os.path.join(current_dir, '..', 'Datasets', 'oikolab_weather_dataset.tsf')
]

results = pd.DataFrame(columns=['Dataset', 'Time elapsed', 'RC1', 'K', 'L', 'w', 'r', 'dist_computed'])

r_vals_computed = [2, 8, 16]
windows = [45, 70, 1000]
dimensionality = [4, 2, 2]

# Base test for time elapsed
for number, path in enumerate(paths):

    # Load the dataset
    if number == 2:
        data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(paths[2], 0)
        d = np.array([data.loc[i,"series_value"].to_numpy() for i in range(data.shape[0])], order='C').T
    elif number == 3:
        data = pd.read_csv(paths[number])
        data = data.drop(['Time','Unix', 'Issues'],axis=1)
        d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
    else:
        data = pd.read_csv(paths[number], delim_whitespace= True)
        data = data.drop(data.columns[[0]], axis=1)
        d = np.ascontiguousarray(data.to_numpy())

    if number == 0:
            # lauch a computation just to compile numba
        pmotif_find2(d, 20, 1, 1, 2, 0.5, 1, 4)

    start = time.process_time()
    for i in range(3):
        motifs, num_dist = pmotif_find2(d, windows[number], 1, dimensionality[number], r_vals_computed[number], 0.5, 100, 8)
    end = (time.process_time() - start)/3

    results = results.append({'Dataset': number, 'Time elapsed': end, 'RC1': relative_contrast(motifs, d, windows[number]), 'K': 8, 'L': 100, 'w': windows[number], 'r': r_vals_computed[number], 'dist_computed': num_dist}, ignore_index=True)
    print("Dataset", number, "finished")
# Run the garbage collector
import gc
gc.collect()

# Test for different K, L and r values independently
Ks = [4, 8, 16]
Ls = [10, 50, 100, 150, 200]
rs = [2, 8, 16, 32]

for number, path in enumerate(paths):
# Load the dataset
    if number == 2:
        data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(paths[2], 0)
        d = np.array([data.loc[i,"series_value"].to_numpy() for i in range(data.shape[0])], order='C').T
    elif number == 3:
        data = pd.read_csv(paths[number])
        data = data.drop(['Time','Unix', 'Issues'],axis=1)
        d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
    else:
        data = pd.read_csv(paths[number], delim_whitespace= True)
        data = data.drop(data.columns[[0]], axis=1)
        d = np.ascontiguousarray(data.to_numpy())

    for K in Ks:
                start = time.process_time()
                for i in range(3):
                    motifs, num_dist = pmotif_find2(d, windows[number], 1, dimensionality[number], r_vals_computed[number], 0.5, 100, K)
                end = (time.process_time() - start)/3
                results = results.append({'Dataset': number, 'Time elapsed': end, 'RC1': np.nan, 'K': K, 'L': 100, 'w': windows[number], 'r': r_vals_computed[number], 'dist_computed': num_dist}, ignore_index=True)
    gc.collect()
    for L in Ls:
                start = time.process_time()
                for i in range(3):
                    motifs, num_dist = pmotif_find2(d, windows[number], 1, dimensionality[number], r_vals_computed[number], 0.5, L, 8)
                end = (time.process_time() - start)/3
                results = results.append({'Dataset': number, 'Time elapsed': end, 'RC1': np.nan, 'K': 8, 'L': L, 'w': windows[number], 'r': r_vals_computed[number], 'dist_computed': num_dist}, ignore_index=True)
    gc.collect()
    for r in rs:
                start = time.process_time()
                for i in range(3):
                    motifs, num_dist = pmotif_find2(d, windows[number], 1, dimensionality[number], r, 0.5, 100, 8)
                end = (time.process_time() - start)/3
                results = results.append({'Dataset': number, 'Time elapsed': end, 'RC1': np.nan, 'r': r, 'K':8, 'L':100, 'dist_computed': num_dist}, ignore_index=True)

    print("Extended test for dataset", number, "finished")

results.to_csv('results.csv', index=False)
