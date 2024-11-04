from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import numpy as np, pandas as pd
import time
from numba import jit, prange
from base import create_shared_array
from  multiprocessing import shared_memory
import pyarrow.parquet as pq
if __name__ == "__main__":
    data = pq.read_table("Datasets/whales.parquet").to_pandas()
    print(data.shape)
    '''
    # Create a (10,4,2) array of random ints
    data = pd.read_csv("Datasets/whales.csv")

    # Downcast the data in the dataframe to float32
    data = data.astype(np.float32)

    # Save aşparquet and compress
    data.to_parquet("Datasets/whales.parquet", compression="gzip")
    '''
