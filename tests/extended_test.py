import os
import sys
sys.path.append('source')
from RP_GRAPH import pmotif_findg
import time
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'external_dependencies'))
from data_loader import convert_tsf_to_dataframe
from base import create_shared_array
from scipy.signal import savgol_filter 
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
        #os.path.join(current_dir, '..', 'Datasets', 'FOETAL_ECG.dat'),
        #os.path.join(current_dir, '..', 'Datasets', 'evaporator.dat'),
        #os.path.join(current_dir, '..', 'Datasets', 'RUTH.csv'),
        os.path.join(current_dir, '..', 'Datasets', 'oikolab_weather_dataset.tsf'),
        #os.path.join(current_dir, '..', 'Datasets', 'CLEAN_House1.csv'),
        #os.path.join(current_dir, '..', 'Datasets', 'whales.parquet'),
        #os.path.join(current_dir, '..', 'Datasets', 'quake.parquet'),
    ]

    r_vals_computed = [4, 8, 16, 32]
    windows = [50, 75, 500, 5000]
    dimensionality = [8, 2, 4, 2]
    
    # Base test for time elapsed
    for number, path in enumerate(paths):
        number_r = number + 3
        results = pd.DataFrame(columns=['Dataset', 'Time elapsed', 'RC1', 'K', 'L', 'w', 'r', 'dist_computed'])


        # Load the dataset
        if number_r == 3:
            data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(path, 0)
            d = np.array([data.loc[i,"series_value"].to_numpy() for i in range(data.shape[0])], order='C', dtype=np.float32).T
            # Apply a savgol filter to the data
            d = savgol_filter(d, 300, 1, axis=0)
        elif number_r == 4:
            data = pd.read_csv(path)
            data = data.drop(['Time','Unix', 'Issues'],axis=1)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
            d = d[:100000,:]
        elif number_r == 2:
            data = pd.read_csv(path)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
            d += np.random.normal(0, 0.1, d.shape)
        elif number_r == 5 or number_r == 6:
            data = pd.read_parquet(paths[number_r])
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)
            d += np.random.normal(0, 0.01, d.shape)
        else:
            data = pd.read_csv(path, sep=r'\s+')
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
        '''
        start = time.process_time()
        for i in range(3):
            motifs, num_dist,_ = pmotif_findg(shm_ts.name, n, dimensions, windows[number_r], 1, dimensionality[number_r], r_vals_computed[number_r], 0.5, 200, 8)
        end = (time.process_time() - start)/3
        motifs = motifs
        
        rel_cont = 0#relative_contrast(d, motifs[0][1][1], windows[number])
        temp_df = pd.DataFrame([{ 'Dataset': number_r, 'Time elapsed': end, 'RC1': rel_cont, 'K': 8, 'L': 200, 'w': windows[number_r], 'r': r_vals_computed[number_r], 'dist_computed': num_dist}])
        results = results._append(temp_df, ignore_index=True)
        results.to_csv("p1"+str(number_r), index=False)
        gc.collect()
        '''
        Ks = [4, 8, 12, 16]
        Ls = [10, 50, 100, 150, 200, 400]
        rs = [4, 8, 16, 32]
        
        # Testing on hashing
        
        for K in Ks:
            start = time.process_time()
            for i in range(1):
                motifs, num_dist,_ = pmotif_findg(shm_ts.name, n, dimensions, windows[number_r], 1, dimensionality[number_r], r_vals_computed[number_r], 0.5, 200, K)
            end = (time.process_time() - start)
            temp_df = pd.DataFrame([{ 'Dataset': number_r, 'Time elapsed': end, 'RC1': 0, 'K': K, 'L': 200, 'w': windows[number_r], 'r': r_vals_computed[number_r], 'dist_computed': num_dist}])
            results = results._append(temp_df, ignore_index=True) 
            gc.collect()
        
        print("K fin")
        
        for L in Ls:
            start = time.process_time()
            for i in range(1):
                motifs, num_dist,ht = pmotif_findg(shm_ts.name, n, dimensions, windows[number_r], 1, dimensionality[number_r], r_vals_computed[number_r], 0.5, L, 8)
            end = (time.process_time() - start)
            temp_df = pd.DataFrame([{ 'Dataset': number_r, 'Time elapsed': end, 'Time int':ht, 'RC1': 0, 'K': 8, 'L': L, 'w': windows[number_r], 'r': r_vals_computed[number_r], 'dist_computed': num_dist}])
            results = results._append(temp_df, ignore_index=True) 
            gc.collect()
        print("L fin")
        results.to_csv("r_partial_dataset"+str(number_r), index=False)
        
        for r in rs:
            start = time.process_time()
            for i in range(1):
                motifs, num_dist,_ = pmotif_findg(shm_ts.name, n, dimensions, windows[number_r], 1, dimensionality[number_r], r, 0.5, 200, 8)
            end = (time.process_time() - start)
            temp_df = pd.DataFrame([{ 'Dataset': number_r, 'Time elapsed': end, 'RC1': 0, 'K': 8, 'L': 200, 'w': windows[number_r], 'r': r, 'dist_computed': num_dist}])
            results = results._append(temp_df, ignore_index=True) 
            gc.collect()

        results.to_csv("r_dataset"+str(number_r),index=False)
        
        print("Dataset", number_r, "finished")
        shm_ts.unlink()

    # Failure test    
    '''
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
                motifs, num_dist = pmotif_findg(d, windows[number], 1, dimensionality[number], r_vals_computed[number], dimensionality[number]/d.shape[1], 100, 8, failure_prob)
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
    noise_dim = [10, 20, 50]
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
                motifs, num_dist = pmotif_findg(d_noise, windows[number], 1, dimensionality[number], r_vals_computed[number], dimensionality[number]/d_noise.shape[1], 100, 8)
                motifs = motifs.queue
                motif_found.append(motifs[0][1][1])
            end = (time.process_time() - start)/3
            temp_df = pd.DataFrame([{ 'Dataset': number, 'Noise': nd, 'Motif1': motif_found[0], 'Motif2': motif_found[1], 'Motif3': motif_found[2]}])
            Noise = Noise._append(temp_df, ignore_index=True)
            print("Dataset", number, "finished")
            motif_found.clear()  
    '''

    #Fail.to_csv('Failures2.csv', index=False)  
    #Noise.to_csv('Noise.csv', index=False)

if __name__ == '__main__':
    #from multiprocessing import freeze_support
    #freeze_support()
    main()
