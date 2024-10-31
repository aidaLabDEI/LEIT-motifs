from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import numpy as np, pandas as pd
import time
from numba import jit, prange
from base import create_shared_array
from  multiprocessing import shared_memory
if __name__ == "__main__":
    # Create a (10,4,2) array of random ints
    data = np.random.randint(0, 5, (10,4,4))

    # Sort each dimension on axis 1 considering the axis 2 as a single value to sort
    for i in range(data.shape[1]):
        lex = np.lexsort(data[:,i,:].T[::-1])
        data[:,i,:] = data[lex,i,:]
        print(data[:,i,:])
