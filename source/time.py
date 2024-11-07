from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import numpy as np, pandas as pd
import time
from numba import jit, prange
from base import create_shared_array
from  multiprocessing import shared_memory
import pyarrow.parquet as pq
import bisect
if __name__ == "__main__":
    a = np.zeros((10), dtype=np.uint8)
    a[0] += 2
    print(a)
    
    #Loop over the rows of the array




    '''
    # Create a (10,4,2) array of random ints
    data = pd.read_csv("Datasets/whales.csv")

    # Downcast the data in the dataframe to float32
    data = data.astype(np.float32)

    # Save aşparquet and compress
    data.to_parquet("Datasets/whales.parquet", compression="gzip")
    '''
