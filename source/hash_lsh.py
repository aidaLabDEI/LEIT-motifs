import numpy as np
from numba import jit, prange

class RandomProjection:
    def __init__(self, dim, r, K, random_state=None):
        self.dim = dim
        self.r = r
        self.K = K
        np.random.seed(random_state)
        self.a = np.random.randn(K, dim)  # K random vectors of dimension dim
        self.b = np.random.uniform(0, r, K)  # K random values
    
    def hash_vector(self, data):
        return compute_hash(data, self.a, self.b, self.r, self.K)

@jit(nopython=True, cache=True)
def compute_hash(data, a, b, r, K):
    hash_value = np.empty(K, dtype=np.int8)
    # print the size of a and data for debugging
    for i in range(K):
        projection = (np.dot(a[i], data) + b[i]) / r
        hash_value[i] = np.floor(projection)
    return hash_value


def euclidean_hash(data, rp):
    return compute_hash(data, rp.a, rp.b, rp.r, rp.K)


if __name__ == "__main__":
    dim = 100
    r = 2
    K = 8  # Length of the hash
    rp = RandomProjection(dim, r, K)
    data = np.random.rand(dim)
    print(data)

    hashed = euclidean_hash(data, rp)
    print(hashed)
