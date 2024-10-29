from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import numpy as np, pandas as pd
import time
from numba import jit, prange

if __name__ == "__main__":
    for i,j in itertools.product(range(8), range(10)):
        print(i,j)