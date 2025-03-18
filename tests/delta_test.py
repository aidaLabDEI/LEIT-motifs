import sys
import time

sys.path.append("source")
import stumpy
from LEITmotifs import LEITmotifs
import numpy as np
import pandas as pd

if __name__ == "__main__":
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
    deltas = [0.01, 0.1, 0.2]
    
    delta_results = pd.DataFrame(columns=["Dataset", "delta", "distance", "fail_prob"])
    
    
    
    
    
    delta_results.to_csv("delta_results.csv", index=False)