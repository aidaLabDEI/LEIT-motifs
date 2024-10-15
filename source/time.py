from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import numpy as np
import time
from numba import jit, prange

if __name__ == "__main__":

    #create a 3d matrix of random integers
    matrix = np.random.randint(1, 10, (3, 3, 3))
    print("Matrix:")
    print(matrix)

    print(matrix[:,1,:-2])
