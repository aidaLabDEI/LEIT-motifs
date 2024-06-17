import stumpy
import time, sys, pandas as pd, numpy as np, queue, os
sys.path.append('external_dependecies')
from data_loader import convert_tsf_to_dataframe

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    paths = [
        os.path.join(current_dir, '..', 'Datasets', 'FOETAL_ECG.dat'),
        os.path.join(current_dir, '..', 'Datasets', 'evaporator.dat'),
        os.path.join(current_dir, '..', 'Datasets', 'RUTH.csv'),
        os.path.join(current_dir, '..', 'Datasets', 'oikolab_weather_dataset.tsf')
    ]

    windows =   [45, 70, 500, 1000]

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

        print("Starting")

        #start = time.process_time()
        #for i in range(3):
        m = stumpy.mstump(d.T, windows[number])
       #end = (time.process_time() - start) / 3

        #print("Dataset", number, "time elapsed:", end, "seconds")