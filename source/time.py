from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import numpy as np, pandas as pd
import time
from numba import jit, prange

if __name__ == "__main__":
    data = pd.read_csv("Datasets/whales.csv")
    d = np.ascontiguousarray(data.to_numpy().T, dtype=np.float32)
    print(d.shape)
