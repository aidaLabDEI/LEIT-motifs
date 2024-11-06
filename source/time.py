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
    l = [[-1,[0]],[-5,[4]],[-3,[8]],[-4,[7]]]
    l = sorted(l, key=lambda x: x[0], reverse=True)
    print(l)
    bisect.insort(l, [-2,[3]], key=lambda x: -x[0])
    print(l)
    bisect.insort(l, [-2,[1]], key=lambda x: -x[0])
    print(l)



    '''
    # Create a (10,4,2) array of random ints
    data = pd.read_csv("Datasets/whales.csv")

    # Downcast the data in the dataframe to float32
    data = data.astype(np.float32)

    # Save aşparquet and compress
    data.to_parquet("Datasets/whales.parquet", compression="gzip")
    '''
