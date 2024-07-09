import os, sys
sys.path.append('source')
from RP_MH import pmotif_find2
from RP_DC import pmotif_find3
import time, pandas as pd, numpy as np, queue
sys.path.append(os.path.join(os.path.dirname(__file__), 'external_dependencies'))
from data_loader import convert_tsf_to_dataframe
from base import z_normalized_euclidean_distance
from find_bin_width import find_width_discr
from extra import relative_contrast
import matplotlib.pyplot as plt
import gc

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

    # Perform the random walk accumulation
    timeseries[:, i] = np.cumsum(steps)

  return timeseries

def main():
    """
    Main function that performs an extended test on multiple datasets.
    It measures the time elapsed, relative contrast, and other parameters for each dataset.
    It also performs tests for different values of K, L, r, failure probability, and noise dimensions.
    """

    current_dir = os.path.dirname(__file__)
    paths = [
        os.path.join(current_dir, '..', 'Datasets', 'FOETAL_ECG.dat'),
        os.path.join(current_dir, '..', 'Datasets', 'evaporator.dat'),
        os.path.join(current_dir, '..', 'Datasets', 'RUTH.csv'),
        os.path.join(current_dir, '..', 'Datasets', 'oikolab_weather_dataset.tsf'),
    ]

    results = pd.DataFrame(columns=['Dataset', 'Time elapsed', 'RC1', 'K', 'L', 'w', 'r', 'dist_computed'])

    r_vals_computed = [2, 8, 16, 16]
    windows = [45, 70, 500, 1000]
    dimensionality = [4, 2, 4, 2]

    # Base test for time elapsed
    for number, path in enumerate(paths):

        # Load the dataset
        if number == 3:
            data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(paths[3], 0)
            d = np.array([data.loc[i,"series_value"].to_numpy() for i in range(data.shape[0])], order='C').T
        elif number == 4:
            data = pd.read_csv(paths[number])
            data = data.drop(['Time','Unix', 'Issues'],axis=1)
            d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
        elif number == 2:
            data = pd.read_csv(paths[number])
            d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
        else:
            data = pd.read_csv(paths[number], delim_whitespace= True)
            data = data.drop(data.columns[[0]], axis=1)
            d = np.ascontiguousarray(data.to_numpy())

        if number == 0:
                # lauch a computation just to compile numba
            pmotif_find2(d, 20, 1, 1, 2, 0.1, 1, 4)
        print("Starting")
        start = time.process_time()
        for i in range(3):
            motifs, num_dist = pmotif_find2(d, windows[number], 1, dimensionality[number], r_vals_computed[number], dimensionality[number]/d.shape[1], 100, 8)
        end = (time.process_time() - start)/3
        motifs = motifs.queue

        rel_cont = 0#relative_contrast(d, motifs[0][1][1], windows[number])
        temp_df = pd.DataFrame([{ 'Dataset': number, 'Time elapsed': end, 'RC1': rel_cont, 'K': 8, 'L': 100, 'w': windows[number], 'r': r_vals_computed[number], 'dist_computed': num_dist}])
        results = results._append(temp_df, ignore_index=True)

        print("Dataset", number, "finished")
    # Run the garbage collector
    gc.collect()
    '''
    # Test for memory, run with mprof
    for number, path in enumerate(paths):
        gc.collect()
    # Load the dataset
        if number == 3:
            data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(paths[3], 0)
            d = np.array([data.loc[i,"series_value"].to_numpy() for i in range(data.shape[0])], order='C').T
        elif number == 4:
            data = pd.read_csv(paths[number])
            data = data.drop(['Time','Unix', 'Issues'],axis=1)
            d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
        elif number == 2:
            data = pd.read_csv(paths[number])
            d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
        else:
            data = pd.read_csv(paths[number], delim_whitespace= True)
            data = data.drop(data.columns[[0]], axis=1)
            d = np.ascontiguousarray(data.to_numpy())

        #tracemalloc.start()    
        motifs, num_dist = pmotif_find2(d, windows[number], 1, dimensionality[number], r_vals_computed[number], dimensionality[number]/d.shape[1], 100, 8)
        gc.collect()
        #current, peak = tracemalloc.get_traced_memory()  # Get the current and peak memory usage
       # tracemalloc.stop()  # Stop tracking memory allocations

        #print("Dataset", number)
        #print("Peak memory usage:", peak / 10**9, "GB")

    # Test for different K, L and r values independently
    Ks = [4, 8, 12, 16]
    Ls = []#[10, 50, 100, 150, 200, 400]
    rs = []#[2, 8, 16, 32]
    paths = [
        os.path.join(current_dir, '..', 'Datasets', 'FOETAL_ECG.dat'),
        os.path.join(current_dir, '..', 'Datasets', 'evaporator.dat'),
        os.path.join(current_dir, '..', 'Datasets', 'RUTH.csv'),
       # os.path.join(current_dir, '..', 'Datasets', 'oikolab_weather_dataset.tsf')
    ]


    for number, path in enumerate(paths):
    # Load the dataset
        if number == 3:
            data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(paths[3], 0)
            d = np.array([data.loc[i,"series_value"].to_numpy() for i in range(data.shape[0])], order='C').T
        elif number == 4:
            data = pd.read_csv(paths[number])
            data = data.drop(['Time','Unix', 'Issues'],axis=1)
            d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
        elif number == 2:
            data = pd.read_csv(paths[number])
            d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
        else:
            data = pd.read_csv(paths[number], delim_whitespace= True)
            data = data.drop(data.columns[[0]], axis=1)
            d = np.ascontiguousarray(data.to_numpy())



        for K in Ks:
                    start = time.process_time()
                    for i in range(2):
                        motifs, num_dist = pmotif_find2(d, windows[number], 1, dimensionality[number], r_vals_computed[number], dimensionality[number]/d.shape[1], 100, K)
                    end = (time.process_time() - start)/2
                    temp_df = pd.DataFrame([{ 'Dataset': number, 'Time elapsed': end, 'RC1': np.nan, 'K': K, 'L': 100, 'w': windows[number], 'r': r_vals_computed[number], 'dist_computed': num_dist}])
                    results = results._append(temp_df, ignore_index=True)   
              
        #gc.collect()

        for L in Ls:
                    start = time.process_time()
                    for i in range(2):
                        motifs, num_dist = pmotif_find2(d, windows[number], 1, dimensionality[number], r_vals_computed[number], dimensionality[number]/d.shape[1], L, 8)
                    end = (time.process_time() - start)/2
                    temp_df = pd.DataFrame([{ 'Dataset': number, 'Time elapsed': end, 'RC1': np.nan, 'K': 8, 'L': L, 'w': windows[number], 'r': r_vals_computed[number], 'dist_computed': num_dist}])
                    results = results._append(temp_df, ignore_index=True)
        gc.collect()
        for r in rs:
                    start = time.process_time()
                    for i in range(2):
                        motifs, num_dist = pmotif_find2(d, windows[number], 1, dimensionality[number], r, dimensionality[number]/d.shape[1], 100, 8)
                    end = (time.process_time() - start)/2
                    temp_df = pd.DataFrame([{ 'Dataset': number, 'Time elapsed': end, 'RC1': np.nan, 'K': 8, 'L': 100, 'w': windows[number], 'r': r, 'dist_computed': num_dist}])
                    results = results._append(temp_df, ignore_index=True)
        print("Extended test for dataset", number, "finished")

    # Failure test    

    Fail = pd.DataFrame(columns=['Dataset', 'Prob','Motif1', 'Motif2', 'Motif3', 'Motif4', 'Motif5', 'Motif6', 'Motif7', 'Motif8', 'Motif9'])
    failure_probs = [0.8, 0.5, 0.2]
    motif_found = []
    for number, path in enumerate(paths):
        motif_found.clear()
        # Load the dataset
        if number == 3:
            data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(paths[3], 0)
            d = np.array([data.loc[i,"series_value"].to_numpy() for i in range(data.shape[0])], order='C').T
        elif number == 4:
            data = pd.read_csv(paths[number])
            data = data.drop(['Time','Unix', 'Issues'],axis=1)
            d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
        elif number == 2:
            data = pd.read_csv(paths[number])
            d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
        else:
            data = pd.read_csv(paths[number], delim_whitespace= True)
            data = data.drop(data.columns[[0]], axis=1)
            d = np.ascontiguousarray(data.to_numpy())

        for failure_prob in failure_probs:
            start = time.process_time()
            for i in range(9):
                motifs, num_dist = pmotif_find2(d, windows[number], 1, dimensionality[number], r_vals_computed[number], dimensionality[number]/d.shape[1], 100, 8, failure_prob)
                motifs = motifs.queue
                motif_found.append(motifs[0][1][1])
            end = (time.process_time() - start)/3
            temp_df = pd.DataFrame([{ 'Dataset': number, 'Prob': failure_prob, 'Motif1': motif_found[0], 'Motif2': motif_found[1], 'Motif3': motif_found[2], 'Motif4': motif_found[3], 'Motif5': motif_found[4], 'Motif6': motif_found[5], 'Motif7': motif_found[6], 'Motif8': motif_found[7], 'Motif9': motif_found[8]}])
            Fail = Fail._append(temp_df, ignore_index=True)
            print("Dataset", number, "finished")
            motif_found.clear()  
    
    paths = [
        os.path.join(current_dir, '..', 'Datasets', 'FOETAL_ECG.dat'),
        os.path.join(current_dir, '..', 'Datasets', 'evaporator.dat')
    ]

    # Noise dimensions test
    Noise = pd.DataFrame(columns=['Dataset', 'Noise','Motif1', 'Motif2', 'Motif3'])
    noise_dim = [10, 50, 100]
    motif_found = []
    for number, path in enumerate(paths):
        motif_found.clear()
        # Load the dataset
        if number == 3:
            data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(paths[3], 0)
            d = np.array([data.loc[i,"series_value"].to_numpy() for i in range(data.shape[0])], order='C').T
        elif number == 4:
            data = pd.read_csv(paths[number])
            data = data.drop(['Time','Unix', 'Issues'],axis=1)
            d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
        elif number == 2:
            data = pd.read_csv(paths[number])
            d = np.ascontiguousarray(data.to_numpy(dtype=np.float64))
        else:
            data = pd.read_csv(paths[number], delim_whitespace= True)
            data = data.drop(data.columns[[0]], axis=1)
            d = np.ascontiguousarray(data.to_numpy())

        for nd in noise_dim:
            # Add nd random walks to the dataset
            noise = generate_random_walk(nd, d.shape[0])
            d_noise = np.concatenate((d, noise), axis=1)

            start = time.process_time()
            for i in range(3):
                motifs, num_dist = pmotif_find2(d_noise, windows[number], 1, dimensionality[number], r_vals_computed[number], dimensionality[number]/d_noise.shape[1], 100, 8)
                motifs = motifs.queue
                motif_found.append(motifs[0][1][1])
            end = (time.process_time() - start)/3
            temp_df = pd.DataFrame([{ 'Dataset': number, 'Noise': nd, 'Motif1': motif_found[0], 'Motif2': motif_found[1], 'Motif3': motif_found[2]}])
            Noise = Noise._append(temp_df, ignore_index=True)
            print("Dataset", number, "finished")
            motif_found.clear()  
    '''

    #Fail.to_csv('Failures2.csv', index=False)  
    results.to_csv('results_run7.csv', index=False)
    #Noise.to_csv('Noise.csv', index=False)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
