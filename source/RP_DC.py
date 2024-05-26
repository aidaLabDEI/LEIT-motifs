from base import *
from find_bin_width import *
from stop import stop
import numpy as np
import pandas as pd
import queue, threading, multiprocessing, itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from hash_lsh import RandomProjection, euclidean_hash
import cProfile

def find_matching_pairs(index, proj_hashes):
    dictionary = {}
    for num_sub, el in enumerate(proj_hashes):
      dictionary.setdefault(np.array2string(el), []).append(num_sub)
    return [(pair, index) for key, value in dictionary.items() for pair in itertools.combinations(value, 2)]

def eq_cycle(i, j, subsequences, hash_mat, k, lsh_threshold):
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
                              rows = hash_mat[coll_0,j,:,:-i+1] == hash_mat[coll_1,j,:,:-i+1]
                              comp = np.sum(np.all(rows, axis=1))
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
                                    print("Vefore comp")
                                    curr_dist, dim, stop_dist = z_normalized_euclidean_distance(subsequences.sub(coll_0), subsequences.sub(coll_1),
                                                                                np.array(coll_dim), subsequences.mean(coll_0), subsequences.std(coll_0),
                                                                           subsequences.mean(coll_1), subsequences.std(coll_1), dimensionality_motifs)
                                    print("Overlap:", curr_dist, dim, collision_couple)
                                    if curr_dist < stored_dist:
                                        top.queue.remove(stored)
                                        top.put((-curr_dist, [dist_comp, collision, [dim], stop_dist]))

                                    add = False
                                    break

                            # Add to top with the projection index
                            if add:
                                print("Before comp")
                                dist_comp +=1
                                distance, dim, stop_dist = z_normalized_euclidean_distance(subsequences.sub(coll_0), subsequences.sub(coll_1),
                                                                           np.array(coll_dim), subsequences.mean(coll_0), subsequences.std(coll_0),
                                                                           subsequences.mean(coll_1), subsequences.std(coll_1), dimensionality_motifs)
                                print("Added:", distance, dim, collision)
                                top.put((-distance, [dist_comp , collision, [dim], stop_dist]))


                                if top.full(): top.get(block=False)

    # Return top k collisions
        print("Computed len:", len(already_comp))
        return top, dist_comp

def pmotif_find3(time_series, window, projection_iter, k, motif_dimensionality, bin_width, lsh_threshold, L, K, fail_thresh=0.8):

    global dist_comp, dimension, b, s, top, failure_thresh, windowed_ts

    random_gen = np.random.default_rng()
  # Data
    dimension = time_series.shape[1]
    n = time_series.shape[0]
    top = queue.PriorityQueue(maxsize=k+1)
    std_container = {}
    mean_container = {}
    b  = K/2
    s = 2
    failure_thresh = fail_thresh
    index_hash = 0

    dist_comp = 0
  # Hasher
    rp = RandomProjection(window, bin_width, K, L) #[]


    chunk_sz = int(np.sqrt(n))
    num_chunks = max(1, n // chunk_sz)
    chunks = [(time_series[ranges[0]:ranges[-1]+window], ranges, window, rp) for ranges in np.array_split(np.arange(n - window + 1), num_chunks)]

    hash_mat = np.array([], dtype=np.int8, order = 'C').reshape(0,L,dimension,K)
    subsequences = np.array([], order='C').reshape(0,dimension,window)

    with multiprocessing.Pool(processes=int(multiprocessing.cpu_count()/2)) as pool:
      results = pool.starmap(process_chunk, [chunk for chunk in chunks])

    for index, result in enumerate(results):

      hash_mat_temp, std_temp, mean_temp, sub_temp = result

      subsequences = np.concatenate([subsequences, sub_temp])
      # Print the shapes of hash_mat_temp and hash_mat
      #print("Hash mat temp shape:", len(hash_mat_temp), hash_mat_temp[0].shape)
      #print("Hash mat shape:", hash_mat.shape)
      hash_mat = np.concatenate([hash_mat, hash_mat_temp])
      std_container.update(std_temp)
      mean_container.update(mean_temp)
    #hash_mat = np.ascontiguousarray(hash_mat, dtype=np.int8)
    #print(hash_mat)
    windowed_ts = WindowedTS(subsequences, window, mean_container, std_container, L, K, motif_dimensionality, bin_width)

    print("Hashing finished")
    lock = threading.Lock()

    global stopped_event
    stopped_event = threading.Event()
    stopped_event.clear()

    #cProfile.runctx("minhash_cycle(i, windowed_ts, hash_mat, k, lsh_threshold, K)",
     #                 {'minhash_cycle':minhash_cycle},
      #                 {'i':0, 'windowed_ts':windowed_ts, 'hash_mat':hash_mat, 'k':k, 'lsh_threshold':lsh_threshold, 'K':K})

    def worker(i,j, K,L, r, motif_dimensionality, dimensions, k):
        global stopped_event, dist_comp, already_comp, b, s, top, failure_thresh, windowed_ts
        top_i, dist_comp_i, already_comp_i = eq_cycle(i, j, windowed_ts, hash_mat, k, lsh_threshold)
        element = None
        length = 0
        with lock:
            top.queue.extend(top_i.queue)
            top.queue.sort(reverse=True)
            length = len(top.queue)

            
            for id, elem in enumerate(top.queue):
                for elem2 in top.queue[id+1:]:

                  indices_1 = elem[1][1]
                  indices_2 = elem2[1][1]

                  if (abs(indices_1[0] - indices_2[0]) < window or
                      abs(indices_1[1] - indices_2[1]) < window or
                      abs(indices_1[0] - indices_2[1]) < window or
                      abs(indices_1[1] - indices_2[0]) < window):
                    if abs(elem[0]) > abs(elem2[0]):
                      top.queue.remove(elem)
                    else:
                      top.queue.remove(elem2)

            top.queue = top.queue[:k]
            dist_comp += dist_comp_i
            if length != 0:
              element = top.queue[0]

        if length == 0:
              pass
        else:
              ss_val = stop(element, motif_dimensionality/dimensions, b,s, i, j, failure_thresh, K, L, r, motif_dimensionality)
              #print("Stop:", ss_val, length)
              if length >= k and ss_val:
                  print("set exit")
                  stopped_event.set()

    with ThreadPoolExecutor(max_workers= int(multiprocessing.cpu_count()/2)) as executor:
        futures = [executor.submit(worker, i, j, K, L, bin_width, motif_dimensionality, dimension, k) for i in range(K) for j in range(L)]
        with tqdm(total=L*K, desc="Iteration") as pbar:
            for future in as_completed(futures):
                pbar.update()
                if stopped_event.is_set():  # Check if the stop event is set
                    executor.shutdown(wait= False, cancel_futures= True)
                    break

    return top, dist_comp