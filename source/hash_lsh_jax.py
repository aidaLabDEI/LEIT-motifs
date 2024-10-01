import numpy as np
import jax
import jax.numpy as jnp
import time

class RandomProjection:
    def __init__(self, dim, r, K, L, random_state=42):
        self.dim = dim
        self.r = r
        self.K = K
        self.L = L
        key = jax.random.key(random_state)
        self.sqrt_L = int(jnp.sqrt(L))
        self.K_half = K // 2
        
        # Generate sqrt(L) sets of K/2 random vectors and values for tensoring
        self.a_l = jax.random.normal(key, (self.sqrt_L, self.K_half, dim)) 
        self.b_l = jax.random.uniform(key, (self.sqrt_L, self.K_half), minval=0, maxval=r)
       
        
        self.a_r = jax.random.normal(key, (self.sqrt_L, self.K_half, dim)) 
        self.b_r = jax.random.uniform(key, (self.sqrt_L, self.K_half), minval=0, maxval=r)
    
def project(a, data, b, r):
    return (jnp.dot(a, data) + b) / r

def hash_vector(rp, data):
        hash_left_all = jnp.empty((rp.sqrt_L, rp.K_half), dtype=np.int8)
        hash_right_all = jnp.empty((rp.sqrt_L, rp.K_half), dtype=np.int8)
        hash_values = jnp.empty((rp.L, rp.K), dtype = np.int8)

        for i in range(rp.sqrt_L):
            for j in range (rp.K_half):
                projection_l = jax.jit(project)(rp.a_l[i, j], data, rp.b_l[i,j], rp.r)
                hash_left_all.at[i,j].set(jnp.floor(projection_l))

                projection_r = jax.jit(project)(rp.a_r[i, j], data, rp.b_r[i,j], rp.r)
                hash_right_all.at[i,j].set(jnp.floor(projection_r))

        for j in range (rp.L):
            l_idx = j // rp.sqrt_L
            r_idx = j % rp.sqrt_L

            hash_left = hash_left_all[l_idx]
            hash_right = hash_right_all[r_idx]

            hash_values.at[j, 0::2].set(hash_left)
            hash_values.at[j, 1::2].set(hash_right)
        return hash_values


def euclidean_hash(data, rp):
    result = hash_vector(rp, data)
    return result


if __name__ == "__main__":
    print("Starting")
    dim = 5000
    r = 8
    K = 8  # Length of the hash
    rp = RandomProjection(dim, r, K, 100)
    data = np.random.rand(dim)


    timei = time.process_time()
    for i in range(10):
        hashed = euclidean_hash(data, rp)
    print("Time elapsed: ", time.process_time() - timei)
    #print(hashed)

