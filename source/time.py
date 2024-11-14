import numpy as np, pandas as pd
if __name__ == "__main__":

    data = pd.read_parquet("Datasets/whales.parquet")
    print(data.shape)


    
    # Create a (10,4,2) array of random ints
    #data = pd.read_csv("Datasets/whales.csv", dtype=np.float32)

    # Downcast the data in the dataframe to float32
    #data = data.T

    # Save aşparquet and compress
    #data.to_parquet("Datasets/whales.parquet", compression="snappy")
    
