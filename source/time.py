import numpy as np
if __name__ == "__main__":
    a = np.array([[1,2,3],[1,1,1]])
    b = np.array([[3,5,6],[1,1,1]])
    print(np.sum((a==b).all(axis=1)))
    
    #Loop over the rows of the array




    '''
    # Create a (10,4,2) array of random ints
    data = pd.read_csv("Datasets/whales.csv")

    # Downcast the data in the dataframe to float32
    data = data.astype(np.float32)

    # Save aşparquet and compress
    data.to_parquet("Datasets/whales.parquet", compression="gzip")
    '''
