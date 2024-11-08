import stumpy
import time
import sys
import pandas as pd
import numpy as np
import os
sys.path.append('external_dependecies')
from data_loader import convert_tsf_to_dataframe
from scipy.signal import savgol_filter

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    paths = [
        os.path.join(current_dir, '..', 'Datasets', 'FOETAL_ECG.dat'),
        os.path.join(current_dir, '..', 'Datasets', 'evaporator.dat'),
        os.path.join(current_dir, '..', 'Datasets', 'RUTH.csv'),
        os.path.join(current_dir, '..', 'Datasets', 'oikolab_weather_dataset.tsf'),
        #os.path.join(current_dir, '..', 'Datasets', 'CLEAN_House1.csv'),
        #os.path.join(current_dir, '..', 'Datasets', 'whales.csv'),
    ]

    windows = [50, 75, 500, 5000, 1000, 300]
    
    # Base test for time elapsed
    for number, path in enumerate(paths):
        number_r = number
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
        elif number_r == 2 or number_r == 5 or number_r == 6:
            data = pd.read_csv(path)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32) if number_r == 2 else np.ascontiguousarray(data.to_numpy().T, dtype=np.float32)
        else:
            data = pd.read_csv(path, sep=r'\s+')
            data = data.drop(data.columns[[0]], axis=1)
            d = np.ascontiguousarray(data.to_numpy(), dtype=np.float32)

        tracemalloc.start()
        start = time.process_time()
        #for i in range(3):
        m = stumpy.mstump(d.T, windows[number])
       #
        end = (time.process_time() - start)
        size, peak = tracemalloc.get_traced_memory()
        print("Dataset", number, "time elapsed:", end, "seconds")
        print(f"Current memory usage is {size / 10**6}MB; Peak was {peak / 10**6}MB")
