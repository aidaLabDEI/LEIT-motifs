import time
import numpy as np
from numba import jit, prange

@jit(nopython=True, fastmath=True, nogil=True)
def eq_sum(a):
    d = a == a
    f = np.ones(d.shape[0])
    for i in prange(d.shape[1]):
        f *= d[:, i] 
    return np.sum(f)

if __name__ == "__main__":

    a = np.ones((10, 10))



    start = time.time()
    for i in range(1000):
        d = a == a
        comp = np.sum(np.all(d, axis=1))
    end = time.time()
    print ("Time elapsed: ", (end - start)/1000)

    start = time.time()
    for i in range(1000):
        eq_sum(a)
    end = time.time()
    print("Time elapsed: ", (end - start)/1000)

    a = np.arange(10)
    print(a[:-0])