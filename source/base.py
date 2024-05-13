import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomDiscretizedProjections
from datasketch import MinHashLSH, MinHash
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import jit


class WindowedTS:
  def __init__(self, subsequences, window: int, rolling_avg, rolling_std, L: int, K: int, motif_dimensionality: int, bin_width: int):
    self.subsequences = subsequences
    self.w = window
    self.avgs = rolling_avg
    self.stds = rolling_std
    self.dimensionality = len(subsequences[0])
    self.num_sub = len(subsequences)
    self.L = L
    self.K = K
    self.d = motif_dimensionality
    self.r = bin_width

  def sub(self, i:int):
    return self.subsequences[i]

  def mean(self, i: int):
    return self.avgs[i]

  def std(self, i: int):
    return self.stds[i]

def euclidean_hash(data, rp):
  hash_str = rp.hash_vector(data)
  return list(map(np.int8, hash_str[0].split('_')))

@jit(nopython=True, cache=True)
def z_normalized_euclidean_distance(ts1, ts2, indices, mean_ts1, std_ts1, mean_ts2, std_ts2, dimensionality = None):
    # Ensure both time series have the same dimensions
    if ts1.shape != ts2.shape:
        raise ValueError("Time series must have the same dimensions.")

    # Pick the dimensions used in this iteration
    ts1 = ts1[indices]
    ts2 = ts2[indices]

    # Z-normalize each dimension separately
    ts1_normalized = (ts1 - mean_ts1[indices, np.newaxis]) / std_ts1[indices, np.newaxis]
    ts2_normalized = (ts2 - mean_ts2[indices, np.newaxis]) / std_ts2[indices, np.newaxis]

    # Compute squared differences and sum them
    squared_diff_sum = np.sqrt(np.sum(np.square(ts1_normalized - ts2_normalized),axis=1))

    if dimensionality and dimensionality != len(indices):
      min_indices = np.argsort(squared_diff_sum)
      return np.sum(squared_diff_sum[min_indices[:dimensionality]]), min_indices[:dimensionality], squared_diff_sum[min_indices[dimensionality]]

    return np.sum(squared_diff_sum), indices, np.max(squared_diff_sum)

def find_collisions(lsh, query_signature):
    # Query the LSH index for potential collisions
    result = lsh.query(query_signature)

    return result


def process_chunk(time_series, ranges, window, rp):
    mean_container = {}
    std_container = {}
    subsequences = []
    hash_mat = []

    for idx in ranges:
        hashed_sub = []
        subsequence = time_series[idx:idx+window].T

        subsequences.append(subsequence)

        mean_container[idx] = np.mean(subsequence, axis=1)

        std_held = np.std(subsequence, axis=1)

        std_container[idx] = np.where(std_held == 0, 0.00001, std_held)

        subsequence_n = (subsequence - mean_container[idx][:,np.newaxis]) / std_container[idx][:,np.newaxis]

        for rp_temp in rp:
          hashed_sub.append(np.apply_along_axis(euclidean_hash, 1, subsequence_n, rp_temp))
        hash_mat.append(hashed_sub)



    return hash_mat, std_container, mean_container, subsequences

@jit(parallel=True, nopython=True, cache=True)
def relative_contrast(ts, pair, window):
  dimensions = ts.shape[1]
  d, _ = z_normalized_euclidean_distance(ts[pair[0]:pair[0]+window],ts[pair[1]:pair[1]+window], np.arange(dimensions),
                                      np.mean(ts[pair[0]:pair[0]+window], axis=0).T, np.std(ts[pair[0]:pair[0]+window], axis=0).T,
                                      np.mean(ts[pair[1]:pair[1]+window], axis=0).T, np.std(ts[pair[1]:pair[1]+window], axis=0).T)

  num = 0
  sum = 0
  for i in range(ts.shape[0]-window+1):
    for j in range(ts.shape[0]-window+1):
      if abs(i-j) > window:
        num += 1
        mean_i = np.mean(ts[i:i+window], axis=0).T
        std_i = np.std(ts[i:i+window], axis=0).T
        mean_j = np.mean(ts[j:j+window], axis=0).T
        std_j = np.std(ts[j:j+window], axis=0).T
        d_ij, _, _ = z_normalized_euclidean_distance(ts[i:i+window], ts[j:j+window], np.arange(dimensions), mean_i, std_i, mean_j, std_j)
        sum += d_ij

  d_hat = sum/num

  return d_hat/d

@jit(parallel=True, nopython=True, cache=True)
def find_all_occur(ts, motifs, window):
  motif_copy = motifs
  for motif in motif_copy:
    occurrences = motif[1][1]

    base = motif[1][1][0]
    base = ts[base:base+window,:].T
    dim = motif[1][2]

    mean_i = np.mean(base, axis=1)
    std_i = np.std(base, axis=1)


    for i in range(ts.shape[0]-window+1):
      ins = True
      for occurr in occurrences:
        if abs(i-occurr) > window:
          ins = ins and True
        else: ins = False
      if ins:
        other = ts[i:i+window,:].T
        print(base, other, dim, mean_i)
        dist, _, _ = z_normalized_euclidean_distance(base, other, dim[0], mean_i, std_i,
                                              np.mean(other, axis=1), np.std(other, axis=1) )
        # If the distance is small enough consider it as a new occurrence of the motif
        if dist < 2:
          occurrences.append(i)
  return motif_copy