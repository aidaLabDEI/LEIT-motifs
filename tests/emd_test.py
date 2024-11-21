from extendedMD.emd import find_motifs_from_emd
import time
import sys
import pandas as pd
import os

sys.path.append("external_dependecies")
from data_loader import convert_tsf_to_dataframe

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    paths = [
        os.path.join(current_dir, "..", "Datasets", "FOETAL_ECG.dat"),
        os.path.join(current_dir, "..", "Datasets", "evaporator.dat"),
        # os.path.join(current_dir, '..', 'Datasets', 'RUTH.csv'),
        # os.path.join(current_dir, '..', 'Datasets', 'oikolab_weather_dataset.tsf')
    ]

    windows = [45, 70, 500, 1000]

    for number, path in enumerate(paths):
        # Load the dataset
        if number == 3:
            data, freq, fc_hor, mis_val, eq_len = convert_tsf_to_dataframe(paths[3], 0)
        elif number == 4:
            data = pd.read_csv(paths[number])
            data = data.drop(["Time", "Unix", "Issues"], axis=1)
        elif number == 2:
            data = pd.read_csv(paths[number])
        else:
            data = pd.read_csv(paths[number], delim_whitespace=True)
            data = data.drop(data.columns[[0]], axis=1)

        print("Starting")

        start = time.process_time()

        # for i in range(3):
        m = find_motifs_from_emd(data, 10, windows[number], 8, 16)

        end = time.process_time() - start

        print("Dataset", number, "time elapsed:", end, "seconds")
