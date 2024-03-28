import pandas as pd
import stumpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
import datetime as dt
from numba import cuda
from tqdm import tqdm
import multiprocessing as mp

from nearpy import Engine
from nearpy.hashes import RandomDiscretizedProjections
from datasketch import MinHashLSH, MinHash


import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing
from itertools import combinations
import cProfile


class WindowedTS:
  def __init__(self, subsequences, window: int, rolling_avg, rolling_std):
    self.subsequences = subsequences
    self.w = window
    self.avgs = rolling_avg
    self.stds = rolling_std
    self.dimensionality = len(subsequences[0])

  def sub(self, i:int):
    return self.subsequences[i]

  def mean(self, i: int):
    return self.avgs[i]

  def std(self, i: int):
    return self.stds[i]

def euclidean_hash(data, rp):
  hash_str = rp.hash_vector(data)
  return list(map(int, hash_str[0].split('_')))

def z_normalized_euclidean_distance(ts1, ts2, indices, mean_ts1, std_ts1, mean_ts2, std_ts2):
    # Ensure both time series have the same dimensions
    if ts1.shape != ts2.shape:
        raise ValueError("Time series must have the same dimensions.")

    # Pick the dimensions used in this iteration
    ts1 = ts1[indices]
    ts2 = ts2[indices]

    '''
    # Calculate mean and standard deviation for each dimension
    mean_ts1 = np.mean(ts1, axis=0)
    std_ts1 = np.std(ts1, axis=0)

    mean_ts2 = np.mean(ts2, axis=0)
    std_ts2 = np.std(ts2, axis=0)
    '''

    # Z-normalize each dimension separately
    ts1_normalized = (ts1 - mean_ts1[indices].reshape(len(indices),1)) / std_ts1[indices].reshape(len(indices),1)
    ts2_normalized = (ts2 - mean_ts2[indices].reshape(len(indices),1)) / std_ts2[indices].reshape(len(indices),1)

    # Compute squared differences and sum them
    squared_diff_sum = np.sqrt(np.square(ts1_normalized - ts2_normalized))

    return np.sum(squared_diff_sum)

def minhash_signature(data, minhasher_seed, perm=64):
    minhasher = MinHash(num_perm=perm, seed=minhasher_seed)
    minhasher.update_batch(np.ascontiguousarray(data))

    return minhasher.copy()

def find_collisions(lsh, query_signature):
    # Query the LSH index for potential collisions
    result = lsh.query(query_signature)

    return result

def truncate(arr, cut):
  arr = arr[:len(arr)-1]
  return arr

def process_chunk(time_series, ranges, window, rp):
    mean_container = {}
    std_container = {}
    subsequences = []
    hash_mat = []

    for idx in ranges:
        hashed_sub = []
        subsequence = time_series[idx:idx+window]
        subsequences.append(subsequence.T)

        mean_container[idx] = np.mean(subsequence, axis=0)

        std_held = np.std(subsequence, axis=0)
        std_container[idx] = np.where(std_held == 0, 0.00001, std_held)

        subsequence_n = (subsequence - mean_container[idx]) / std_container[idx]
        for rp_temp in rp:
          hashed_sub.append(np.apply_along_axis(euclidean_hash, 0, subsequence_n, rp_temp))
        hash_mat.append(hashed_sub)



    return hash_mat, std_container, mean_container, subsequences

def relative_contrast(ts, pair, window):
  dimensions = ts.shape[1]
  d = z_normalized_euclidean_distance(ts[pair[0]:pair[0]+window],ts[pair[1]:pair[1]+window], np.arange(dimensions),
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
        d_ij = z_normalized_euclidean_distance(ts[i:i+window], ts[j:j+window], np.arange(dimensions), mean_i, std_i, mean_j, std_j)
        sum += d_ij

  d_hat = sum/num

  return d_hat/d

def find_all_occur(ts, motifs, window):
  for motif in motifs:
    occurrences = motif[1][1]

    base = motif[1][1][0]
    base = ts[base:base+window,:]
    dim = motif[1][2]

    mean_i = np.mean(base, axis=0).T
    std_i = np.std(base, axis=0).T


    for i in range(ts.shape[0]-window+1):
      ins = True
      for occurr in occurrences:
        if abs(i-occurr) > window:
          ins = ins and True
        else: ins = False
      if ins:
        other = ts[i:i+window,:]
        dist = z_normalized_euclidean_distance(base, other, dim[0], mean_i, std_i,
                                              np.mean(other, axis=0).T, np.std(other, axis=0).T )
        if dist < 1:
          motif[1][1].append(i)
          occurrences.append(i)
  return motifs

def minhash_cycle(i, subsequences, hash_mat, k, lsh_threshold, K):
        window = subsequences.w
        top = queue.PriorityQueue(k+1)
        random_gen = np.random.default_rng()
        #Save the couples that we already computed in this iteration
        already_comp = set()
        pj_ts = []
        for elem in hash_mat:
          pj_ts.append(elem[i].T)
        dist_comp= 0
        index_hash= 0
        collided = False
        while not collided and index_hash < K-1:
            if index_hash > 0:
                '''
                chunks = [(truncate, 2, piece, 1) for piece in np.array_split(pj_ts, mp.cpu_count())]
                pj_ts = []
                with mp.Pool(mp.cpu_count()) as pool:
                  results = pool.starmap(np.apply_along_axis, [chunk for chunk in chunks])
                for result in results:
                  pj_ts = np.concatenate([pj_ts, result]) if len(pj_ts)!=0 else result
                '''
                pj_ts= np.apply_along_axis(truncate, 0, pj_ts, 1)
                #print("Truncating")
            # Compute fingerprints
                # Create MinHash object
            minhash_seed = random_gen.integers(0, 2**32 - 1)
            minhash_signatures = []
            lsh = MinHashLSH(threshold=lsh_threshold, num_perm=int(K/2))
            for ik, signature in enumerate(MinHash.generator(np.ascontiguousarray(pj_ts), num_perm=int((K)/2), seed=minhash_seed)):
                minhash_signatures.append(signature)
                lsh.insert(ik, signature)
            # Find collisions
            for j, minhash_sig in enumerate(minhash_signatures):
                    collisions = lsh.query(minhash_sig)
                    #print(collisions)
                    if len(collisions) > 1:
                        # Remove trivial matches, same subsequence or overlapping subsequences
                        collisions = [sorted((j, c)) for c in collisions if c != j and abs(c - j) > window]
                        #print(collisions)
                        curr_dist = 0
                        for collision in collisions:
                            add = True

                        # If we already computed this couple skip
                            if tuple(collision) in already_comp:
                                add=False
                                break
                        # If already inserted skip
                            if( any(collision == stored_el1 for _, (_, stored_el1, _) in top.queue)):
                                add = False
                                break

                            # Check overlap with the already computed
                            for stored in top.queue:
                                #Access the collision
                                stored_dist = abs(stored[0])
                                stored_el = stored[1]
                                stored_el1 = stored_el[1]
                                #stored = stored[1][0]
                                # If it's an overlap of both indices, keep the one with the smallest distance

                                if (abs(collision[0] - stored_el1[0]) < window or
                                    abs(collision[1] - stored_el1[1]) < window or
                                    abs(collision[0] - stored_el1[1]) < window or
                                    abs(collision[1] - stored_el1[0]) < window):
                                  # Distance is computed only on distances that match
                                    dim = pj_ts[collision[0]] == pj_ts[collision[1]]
                                    dim = np.all(dim, axis=1)
                                    dim = [i for i, elem in enumerate(dim) if elem == True]

                                    #print(dim)
                                    if len(dim) < 2: break
                                    dist_comp += 1
                                    curr_dist = z_normalized_euclidean_distance(subsequences.sub(collision[0]), subsequences.sub(collision[1]),
                                                                                dim, subsequences.mean(collision[0]), subsequences.std(collision[0]),
                                                                           subsequences.mean(collision[1]), subsequences.std(collision[1]))
                                    if curr_dist/len(dim) < stored_dist:
                                        top.queue.remove(stored)
                                        top.put((-curr_dist/len(dim), [dist_comp, collision, [dim]]))
                                        already_comp.add(tuple(collision))
                                    collided = True
                                    add = False
                                    break

                            # Add to top with the projection index
                            if add:


                                # Pick just the equal dimensions to compute the distance
                                dim = pj_ts[collision[0]] == pj_ts[collision[1]]
                                dim = np.all(dim, axis=1)
                                dim = [i for i, elem in enumerate(dim) if elem == True]
                                if len(dim) < 2: break
                                dist_comp +=1
                                distance = z_normalized_euclidean_distance(subsequences.sub(collision[0]), subsequences.sub(collision[1]),
                                                                           dim, subsequences.mean(collision[0]), subsequences.std(collision[0]),
                                                                           subsequences.mean(collision[1]), subsequences.std(collision[1]))
                                top.put((-distance/len(dim), [dist_comp , collision, [dim]]))
                                already_comp.add(tuple(collision))
                                if top.full(): top.get(block=False)
                                collided = True
            # Repeat with K-1 hashes
            if not collided:
                index_hash +=1

    # Return top k collisions
        return top, dist_comp

def pmotif_find2(time_series, window, projection_iter, k, project_comp, bin_width, lsh_threshold, L, K):
    random_gen = np.random.default_rng()
  # Data
    dimension = time_series.shape[1]
    top = queue.PriorityQueue(maxsize=k+1)
    std_container = {}
    mean_container = {}

    index_hash = 0
    dist_comp = 0
  # Hasher
    engines= []
    rp = []
    # Create the repetitions for the LSH
    for i in range(L):
      rps= RandomDiscretizedProjections('rp', K, bin_width)
      engine = Engine(window, lshashes=[rps])
      rp.append(rps)
      engines.append(engine)

    chunks = [(np.array(time_series), ranges, window, rp) for ranges in np.array_split(np.arange(time_series.shape[0] - window + 1), multiprocessing.cpu_count())]

    hash_mat = np.array([]).reshape(0,L,K,dimension)
    subsequences = np.array([]).reshape(0,dimension,window)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
      results = pool.starmap(process_chunk, [chunk for chunk in chunks])

    for index, result in enumerate(results):

      hash_mat_temp, std_temp, mean_temp, sub_temp = result

      subsequences = np.concatenate([subsequences, sub_temp])
      hash_mat = np.concatenate([hash_mat, hash_mat_temp])
      std_container.update(std_temp)
      mean_container.update(mean_temp)

    '''
    for i in range(len(time_series) - window + 1):
        hashed_sub = []
        subsequence = time_series[i:i + window]
        subsequences.append(subsequence.T)
        mean_container[i] = np.mean(subsequence, axis=0)
        std_held = np.std(subsequence, axis=0)
        if np.any(std_held == 0):
        # Set standard deviation to a small value (epsilon) for zero entries
          std_container[i] = np.where(std_held == 0, 0.00001, std_held)
        else:
          std_container[i] = std_held
        subsequence = (subsequence - mean_container[i]) / std_container[i]
        for rp_temp in rp:
          hashed_sub.append(np.apply_along_axis(euclidean_hash, 0, subsequence, rp_temp))
        # Insert into the matrix at the corresponding row
        hash_mat.append(hashed_sub)
     '''
    windowed_ts = WindowedTS(subsequences, window, mean_container, std_container)

    print("Hashing finished")
    lock = threading.Lock()

   # cProfile.runctx("minhash_cycle(i, windowed_ts, hash_mat, k, lsh_threshold, K)",
    #                  {'minhash_cycle':minhash_cycle},
     #                  {'i':0, 'windowed_ts':windowed_ts, 'hash_mat':hash_mat, 'k':k, 'lsh_threshold':lsh_threshold, 'K':K})


    with ThreadPoolExecutor() as executor:

      result = {executor.submit(minhash_cycle, i, windowed_ts, hash_mat, k, lsh_threshold, K) for i in range(L)}
      with tqdm(total=L, desc="Iteration") as pbar:
        for future in as_completed(result):
          top_i, dist_comp_i = future.result()
          pbar.update(1)
         # print("Wait lock")
          with lock:
           # print("Acquired lock")
            top.queue.extend(top_i.queue)
            dist_comp += dist_comp_i

            # Order the queue
    top.queue.sort(reverse=True)
    #Remove overlapping tuples inside the queue


    for id, elem in enumerate(top.queue):
      for elem2 in top.queue[id+1:]:
        if (abs(elem[1][1][0] - elem2[1][1][0]) < window or
            abs(elem[1][1][1] - elem2[1][1][1]) < window or
            abs(elem[1][1][0] - elem2[1][1][1]) < window or
            abs(elem[1][1][1] - elem2[1][1][0]) < window):
          if abs(elem[0]) > abs(elem2[0]):
            top.queue.remove(elem)
          else:
            top.queue.remove(elem2)

    top.queue = top.queue[:k]

    return top, dist_comp