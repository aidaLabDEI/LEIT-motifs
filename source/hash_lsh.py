import numpy as np
from numba import jit, prange
import time

class RandomProjection:
    def __init__(self, dim, r, K, L, random_state=None):
        self.dim = dim
        self.r = r
        self.K = K
        self.L = L
        np.random.seed(random_state)
        sqrt_L = int(np.sqrt(L))
        K_half = K // 2
        
        # Generate sqrt(L) sets of K/2 random vectors and values for tensoring
        self.a_l = np.random.randn(sqrt_L, K_half, dim)  
        self.b_l = np.random.uniform(0, r, (sqrt_L, K_half))
        
        self.a_r = np.random.randn(sqrt_L, K_half, dim)  
        self.b_r = np.random.uniform(0, r, (sqrt_L, K_half))
    
    def hash_vector(self, data):
        return compute_hash(data, self.a_l, self.b_l, self.a_r, self.b_r, self.r, self.K, self.L)

@jit(nopython=True, cache=True)
def compute_hash(data, a_l, b_l, a_r, b_r, r, K, L):
    sqrt_L = int(np.sqrt(L))
    K_half = K // 2
    hash_left_all = np.empty((sqrt_L, K_half), dtype=np.int8)
    hash_right_all = np.empty((sqrt_L, K_half), dtype=np.int8)
    
    # Compute the K/2 hashes for both collections
    for l_idx in prange(sqrt_L):
        for i in range(K_half):
            projection_l = (np.dot(a_l[l_idx, i], data) + b_l[l_idx, i]) / r
            hash_left_all[l_idx, i] = np.floor(projection_l)

    for r_idx in prange(sqrt_L):
        for i in range(K_half):
            projection_r = (np.dot(a_r[r_idx, i], data) + b_r[r_idx, i]) / r
            hash_right_all[r_idx, i] = np.floor(projection_r)
    
    hash_values = np.empty((L, K), dtype=np.int8)
    
    # Interleave the results to get final L hashes of length K
    for j in prange(L):
        l_idx = j // sqrt_L
        r_idx = j % sqrt_L

        hash_left = hash_left_all[l_idx]
        hash_right = hash_right_all[r_idx]

        hash_values[j, 0::2] = hash_left
        hash_values[j, 1::2] = hash_right

    return hash_values

def euclidean_hash(data, rp):
    return compute_hash(data, rp.a_l, rp.b_l, rp.a_r, rp.b_r, rp.r, rp.K, rp.L)

if __name__ == "__main__":
    dim = 5000
    r = 8
    K = 8  # Length of the hash
    rp = RandomProjection(dim, r, K, 100)
    data = np.random.rand(dim)


    timei = time.process_time()
    for i in range(100000):
        hashed = euclidean_hash(data, rp)
    print("Time elapsed: ", time.process_time() - timei)
    #print(hashed)

