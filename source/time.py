import numpy as np, pandas as pd
from cachetools import LRUCache
if __name__ == "__main__":

    cache = LRUCache(maxsize=2)
    print(cache)
    cache["a"] = 1
    cache["b"] = 2
    if "a" in cache:
        cache["a"] = 1
        print("a is in cache")
    cache["c"] = 3
    print(cache)


    
    # Create a (10,4,2) array of random ints
    #data = pd.read_csv("Datasets/whales.csv", dtype=np.float32)

    # Downcast the data in the dataframe to float32
    #data = data.T

    # Save aşparquet and compress
    #data.to_parquet("Datasets/whales.parquet", compression="snappy")
    
