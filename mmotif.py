import numpy as np
from sklearn.random_projection import SparseRandomProjection
from datasketch import MinHashLSH, MinHash
from scipy.spatial.distance import euclidean
import pandas as pd
import matplotlib.pyplot as plt


def z_normalized_euclidean_distance(ts1, ts2, random_indices):
    # Ensure both time series have the same dimensions
    if ts1.shape != ts2.shape:
        raise ValueError("Time series must have the same dimensions.")
    # Pick the dimensions used in this iteration
    ts1 = ts1[:,random_indices]
    ts2 = ts2[:,random_indices]

    # Calculate mean and standard deviation for each dimension
    mean_ts1 = np.mean(ts1, axis=0)
    std_ts1 = np.std(ts1, axis=0)

    mean_ts2 = np.mean(ts2, axis=0)
    std_ts2 = np.std(ts2, axis=0)

    sum_sqrd = 0.0

    # z-normalized dist
    for item1, item2 in zip(ts1,ts2):
      sum_sqrd += np.square((((item1-mean_ts1)/std_ts1) - ((item2-mean_ts2)/std_ts2)))
    return np.sum(np.sqrt(sum_sqrd))

def random_projection_cycle(data, n_components):
    # Perform random projection on the data
    transformer = SparseRandomProjection(n_components=n_components)
    projected_data = transformer.fit_transform(data)

    return projected_data

def minhash_signature(data, minhasher_seed):
    minhasher = MinHash(num_perm=128, seed=minhasher_seed)
    #z normalize before the hashing
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_hash = (data - mean)/std

    # Data hash is a z-normalized multidimensional subsequence, a signature must be created for the whole subsequence
    # Transposision takes the whole dimension
    for item in np.ascontiguousarray(data_hash.T):
        minhasher.update(item)

    return minhasher.copy()

def find_collisions(lsh, query_signature):
    # Query the LSH index for potential collisions
    result = lsh.query(query_signature)

    return result

def motif_find(time_series, window, projection_iter, k, project_comp, lsh_threshold):
    random_gen = np.random.default_rng()
  # Data
    dimension = time_series.shape[1]
    num_comp = project_comp
    top = []
  # Extract all subsequences
    subsequences = []

    for i in range(len(time_series) - window + 1):
        subsequence = time_series[i:i + window]
        subsequences.append(subsequence)

    for i in range(projection_iter):
      # Random project
        #pj_ts = random_projection_cycle(subsequences, num_comp)
        # Find num_comp random numbers between 0 and dimension of time series
        random_indices = np.random.choice(dimension, num_comp, replace=False)
        pj_ts = [subsequence[:,random_indices] for subsequence in subsequences]
      # Compute fingerprints
          # Create MinHash object
        minhash_seed = random_gen.integers(0, 2**32 - 1)
        minhash_signatures = []

        for projected_subsequence in pj_ts:
            minhash_sig = minhash_signature(projected_subsequence, minhash_seed)
            minhash_signatures.append(minhash_sig)

        lsh = MinHashLSH(threshold=lsh_threshold, num_perm=128)
        for ik, signature in enumerate(minhash_signatures):
          lsh.insert(ik, signature)

      # Find collisions
        for j, minhash_sig in enumerate(minhash_signatures):
                collisions = lsh.query(minhash_sig)
                if len(collisions) > 1:
                    # Remove trivial matches, same subsequence or overlapping subsequences
                    collisions = [(j, c) for c in collisions if c != j and abs(c - j) > window]
                    print(collisions)
                    for collision in collisions:
                      # If doesn't exists as the same or the reverse match or as an overlap
                      if collision not in top and (collision[1], collision[0]) not in top:
                        add = True
                        for stored in top:
                          if (not abs(collision[0] - stored[0]) > window or
                              not abs(collision[1] - stored[1]) > window or
                              not abs(collision[1] - stored[0]) > window or
                              not abs(collision[0] - stored[1]) > window):
                            add = False
                        # Add to top with the projection index
                          if add: top.append(tuple(collision, random_indices))


                        

                    # If top exceeds max length, keep only the top k based on distance
                    if len(top) > k:
                        top.sort(key=lambda x: z_normalized_euclidean_distance(subsequences[x[0][0]], subsequences[x[0][1]], x[1]))
                        top = top[:k]

    # Return top k collisions
    return top

