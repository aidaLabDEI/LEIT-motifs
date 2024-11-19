import numpy as np, pandas as pd
from cachetools import LRUCache
from numba import njit, prange
import numba as nb, time

@njit(nb.bool(nb.int8[:], nb.int8[:]),
        fastmath=True, cache= True)
def eq(a,b):
    return np.all(a == b)

@njit(nb.int8(nb.int8[:,:], nb.int8[:,:]),
      fastmath=True, cache=True)
def multi_eq(a, b):
    sum = 0
    for i in prange(a.shape[0]):
      if eq(a[i], b[i]):
          sum += 1
    return sum
   
@njit(nb.float32[:](nb.float32[:,:]),fastmath=True, cache=True)
def comp_mean(a):
    res = np.empty(a.shape[0], dtype=np.float32)
    for i in range(a.shape[0]):
        res[i] = np.mean(a[i])
    return res

@njit(nb.float32[:](nb.float32[:,:]),fastmath=True, cache=True)
def comp_std(a):
    res = np.empty(a.shape[1], dtype=np.float32)
    for i in range(a.shape[0]):
        res[i] = np.std(a[i])
    return res

if __name__ == "__main__":
   # Create a (30,8) array of random ints
   f = np.random.randint(0, 100, (1024,5000))
   a = np.array(f, dtype=np.float32)
   f = np.random.randint(0, 100, (30,8))
   b = np.array(f, dtype=np.int8)

   _t = time.perf_counter()
   for _ in range(1000):
         comp_mean(a)
   print(time.perf_counter() - _t)
   print(comp_mean(a))
   _t = time.perf_counter()
   for _ in range(1000):
       np.mean(a, axis=1)
   print(time.perf_counter() - _t)
   print(np.mean(a, axis=1))




    
    # Create a (10,4,2) array of random ints
    #data = pd.read_csv("Datasets/whales.csv", dtype=np.float32)

    # Downcast the data in the dataframe to float32
    #data = data.T

    # Save aşparquet and compress
    #data.to_parquet("Datasets/whales.parquet", compression="snappy")
    
