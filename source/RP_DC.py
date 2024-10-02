from base import *
from find_bin_width import *
from stop import stop3
import numpy as np
import pandas as pd
import queue, threading, multiprocessing, itertools
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from hash_lsh import RandomProjection, euclidean_hash
from numba import jit
import cProfile

@jit(nopython=True, cache=True, parallel=True)
def count(hash_line1, hash_line2):
    row = hash_line1 == hash_line2
    return np.sum(np.all(row, axis=1))

def find_matching_pairs(index, proj_hashes):
    dictionary = {}
    for num_sub, el in enumerate(proj_hashes):
      dictionary.setdefault(np.array2string(el), []).append(num_sub)
    matches = []
    for key, value in dictionary.items():
        if len(value) > 1:
            for pair in itertools.combinations(value, 2):
                matches.append((pair, index))
    return matches

def eq_cycle(i, j, subsequences, hash_mat, k, lsh_threshold):
        print("Cycle:", i, j)
        K = subsequences.K
        dimensionality = subsequences.dimensionality
        dimensionality_motifs = subsequences.d
        window = subsequences.w
        top = queue.PriorityQueue()
        random_gen = np.random.default_rng()

        pj_ts = hash_mat[:,j,:,:-i] if not i==0 else hash_mat[:,j,:,:]
        dist_comp= 0
        matching_pairs_with_index = []
        #Populate the dictionaries and find collisions
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit tasks for each dimension
            future_to_index = {executor.submit(find_matching_pairs, index, pj_ts[:,index,:]): index for index in range(dimensionality)}
            # Iterate over the completed futures to collect the results
            for future in as_completed(future_to_index):
                matching_pairs_with_index.extend(future.result())
        # Reduce emitted pairs to include all indices for each matching pair
        matching_pairs_with_indices = {}
        for pair, index in matching_pairs_with_index:
            if pair not in matching_pairs_with_indices:
                matching_pairs_with_indices[pair] = []
            matching_pairs_with_indices[pair].append(index)
        #print for debugging

        # Check between the collisions all the valid ones
        for collision_couple, coll_dim in matching_pairs_with_indices.items():
            coll_0 = collision_couple[0]
            coll_1 = collision_couple[1]
            if len(coll_dim) >= dimensionality_motifs and abs(coll_0-coll_1) > window:
                            add = True
                        # If we already computed this couple skip
                            if not i == 0:
                              #rows = hash_mat[coll_0,j,:,:-i+1] == hash_mat[coll_1,j,:,:-i+1]
                              #comp = np.sum(np.all(rows, axis=1))
                              comp = count(hash_mat[coll_0,j,:,:-i+1], hash_mat[coll_1,j,:,:-i+1])
                              if comp >= dimensionality:
                                #print("Skipped")
                                add = False
                                break

                            # Check overlap with the already computed
                            for stored in top.queue:
                                #Access the collision
                                stored_dist = abs(stored[0])
                                stored_el = stored[1]
                                stored_el1 = stored_el[1]

                                stor_0 = stored_el1[0]
                                stor_1 = stored_el1[1]
                                
                                # If it's an overlap of both indices, keep the one with the smallest distance
                                if (abs(coll_0 - stor_0) < window or
                                    abs(coll_1 - stor_1) < window or
                                    abs(coll_0 - stor_1) < window or
                                    abs(coll_1 - stor_0) < window):

                                    #if len(dim) < dimensionality: break
                                    dist_comp += 1
                                    curr_dist, dim, stop_dist = z_normalized_euclidean_distance(subsequences.sub(coll_0), subsequences.sub(coll_1),
                                                                                np.array(coll_dim), subsequences.mean(coll_0), subsequences.std(coll_0),
                                                                           subsequences.mean(coll_1), subsequences.std(coll_1), dimensionality_motifs)
                                    if curr_dist < stored_dist:
                                        top.queue.remove(stored)
                                        top.put((-curr_dist, [dist_comp, collision_couple, [dim], stop_dist]))

                                    add = False
                                    break

                            # Add to top with the projection index
                            if add:
                                dist_comp +=1
                                distance, dim, stop_dist = z_normalized_euclidean_distance(subsequences.sub(coll_0), subsequences.sub(coll_1),
                                                                           np.array(coll_dim), subsequences.mean(coll_0), subsequences.std(coll_0),
                                                                           subsequences.mean(coll_1), subsequences.std(coll_1), dimensionality_motifs)
                                top.put((-distance, [dist_comp , collision_couple, [dim], stop_dist]))


                                if top.full(): top.get(block=False)

    # Return top k collisions
        #print("Computed len:", dist_comp)
        return top, dist_comp

def pmotif_find3(time_series, window, k, motif_dimensionality, bin_width, lsh_threshold, L, K, fail_thresh=0.8):

    global dist_comp, dimension, b, s, top, failure_thresh, time_tot
    time_tot = 0
    random_gen = np.random.default_rng()
  # Data
    dimension = time_series.shape[1]
    n = time_series.shape[0]
    top = queue.PriorityQueue(maxsize=k+1)
    std_container = {}
    mean_container = {}

    
    failure_thresh = fail_thresh
    index_hash = 0

    dist_comp = 0
  # Hasher
    rp = RandomProjection(window, bin_width, K, L) #[]


    chunk_sz = int(np.sqrt(n))
    num_chunks = max(1, n // chunk_sz)
    
    chunks = [(time_series[ranges[0]:ranges[-1]+window], ranges, window, rp) for ranges in np.array_split(np.arange(n - window + 1), num_chunks)]

    shm_hash_mat, hash_mat = create_shared_array((n-window+1, L, dimension, K), dtype=np.int8)

    with Pool(processes=int(multiprocessing.cpu_count())) as pool:
        results = []

        for chunk in chunks:
            result = pool.apply_async(process_chunk, (*chunk, shm_hash_mat.name, hash_mat.shape, L, dimension, K))
            results.append(result)

        for result in results:
            std_temp, mean_temp = result.get()
            std_container.update(std_temp)
            mean_container.update(mean_temp)

    windowed_ts = WindowedTS(time_series, window, mean_container, std_container, L, K, motif_dimensionality, bin_width)

    print("Hashing finished")
    lock = threading.Lock()

    global stopped_event
    stopped_event = threading.Event()
    stopped_event.clear()

    def worker(i,j, K,L, r, motif_dimensionality, dimensions, k):
      global stopped_event, top, dist_comp
      if stopped_event.is_set(): return
      top_i, dist_comp_i = eq_cycle(i,j,windowed_ts, hash_mat, k, lsh_threshold)
      with lock:
        top.queue.extend(top_i.queue)
        top.queue.sort(reverse=True)
        top.queue = top.queue[:k]
        dist_comp += dist_comp_i
        if not top.empty():
            ss_val = stop3(top.queue[0], i, j, fail_thresh, K, L, bin_width, motif_dimensionality)
            if ss_val and len(top.queue) >= k:
                stopped_event.set()
                return

    with ThreadPoolExecutor(max_workers= int(multiprocessing.cpu_count()) ) as executor:
      futures = {executor.submit(worker, i,j, K,L, bin_width, motif_dimensionality, dimension, k): (i,j) for i in range(L) for j in range(K)}
      for future in as_completed(futures):
        if stopped_event.is_set():
            executor.shutdown(wait=False, cancel_futures=True)
            break
    '''
    condition = True
    for i in range(L):
      for j in range(K):
          if condition:
            top_i, dist_comp_i = eq_cycle(i, j, windowed_ts, hash_mat, k, lsh_threshold)
            dist_comp += dist_comp_i
            top.queue.extend(top_i.queue)
            top.queue.sort(reverse=True)
            top.queue = top.queue[:k]
            if not top.empty():
                condition = not stop(top.queue[0], i, j, fail_thresh, K, L, bin_width, motif_dimensionality)
    '''


    return top, dist_comp