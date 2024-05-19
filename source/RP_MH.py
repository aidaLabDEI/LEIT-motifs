from base import *
from find_bin_width import *
from stop import stop
import numpy as np
import pandas as pd
import queue
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cProfile
from hash_lsh import RandomProjection, euclidean_hash
#from nearpy import Engine
#from nearpy.hashes import RandomDiscretizedProjections
from datasketch import MinHashLSH, MinHash



def minhash_cycle(i, j, subsequences, hash_mat, k, lsh_threshold):
        # Initialize variables
        K = subsequences.K
        dimensionality = subsequences.d
        window = subsequences.w
        top = queue.PriorityQueue(k+1)
        random_gen = np.random.default_rng()

        # Extract the repetition we are working with, cut at the index
        pj_ts = hash_mat[:,j,:,:-i] if not i==0 else hash_mat[:,j,:,:]

        # Count the number of distance computations
        dist_comp= 0
            # Compute fingerprints
                # Create MinHash object
        minhash_seed = random_gen.integers(0, 2**32 - 1)
        minhash_signatures = []
        lsh = MinHashLSH(threshold=lsh_threshold, num_perm=int(K/2))
        
        with lsh.insertion_session() as session:
              for ik, signature in enumerate(MinHash.generator(pj_ts, num_perm=int((K)/2), seed=minhash_seed)):
                minhash_signatures.append(signature)
                session.insert(ik, signature)

        # Find collisions
        for j_index, minhash_sig in enumerate(minhash_signatures):
                    collisions = lsh.query(minhash_sig)
                    #print(collisions)
                    if len(collisions) >= 1:
                        # Remove trivial matches, same subsequence or overlapping subsequences
                        collisions = [sorted((j_index, c)) for c in collisions if c != j_index and abs(c - j_index) > window]
                        curr_dist = 0

                        for collision in collisions:
                            coll_0 = collision[0]
                            coll_1 = collision[1]

                            add = True
                            # If they collide at the previous level, skip
                            if not i == 0:
                              rows = hash_mat[coll_0,j,:,:] == hash_mat[coll_1,j,:,:]
                              comp = np.sum(np.all(rows[:,:K-i], axis=1))
                              if comp >= dimensionality:
                                #print("Skipped")
                                add = False
                                break

                        # If already inserted skip
                            #if( any(collision == stored_el1 for _, (_, stored_el1, _ , _) in top.queue)):
                             #   add = False
                               # break

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

                                  # Distance is computed only on distances that match
                                    dim = pj_ts[coll_0] == pj_ts[coll_1]
                                    dim = np.all(dim, axis=1)
                                    dim = [i for i, elem in enumerate(dim) if elem == True]

                                    if len(dim) < dimensionality: break
                                    dist_comp += 1
                                    curr_dist, dim, stop_dist = z_normalized_euclidean_distance(subsequences.sub(coll_0), subsequences.sub(coll_1),
                                                                                np.array(dim), subsequences.mean(coll_0), subsequences.std(coll_0),
                                                                           subsequences.mean(coll_1), subsequences.std(coll_1), dimensionality)
                                    if curr_dist < stored_dist:
                                        top.queue.remove(stored)
                                        top.put((-curr_dist, [dist_comp, collision, [dim], stop_dist]))

                                    add = False
                                    break

                            # Add to top with the projection index
                            if add:

                                # Pick just the equal dimensions to compute the distance
                                dim = pj_ts[coll_0] == pj_ts[coll_1]
                                dim = np.all(dim, axis=1)
                                dim = [i for i, elem in enumerate(dim) if elem == True]
                                if len(dim) < dimensionality: break
                                dist_comp +=1
                                distance, dim, stop_dist = z_normalized_euclidean_distance(subsequences.sub(coll_0), subsequences.sub(coll_1),
                                                                           np.array(dim), subsequences.mean(coll_0), subsequences.std(coll_0),
                                                                           subsequences.mean(coll_1), subsequences.std(coll_1), dimensionality)
                                top.put((-distance, [dist_comp , collision, [dim], stop_dist]))

                                if top.full(): top.get(block=False)

    # Return top k collisions
        print("Computed len:", len(top.queue))
        return top, dist_comp

def pmotif_find2(time_series, window, projection_iter, k, motif_dimensionality, bin_width, lsh_threshold, L, K, fail_thresh=0.98):

    global dist_comp, dimension, b, s, top, failure_thresh

    random_gen = np.random.default_rng()
  # Data
    dimension = time_series.shape[1]
    top = queue.PriorityQueue(maxsize=k+1)
    std_container = {}
    mean_container = {}
    b  = K/2
    s = 2
    failure_thresh = fail_thresh
    index_hash = 0

    dist_comp = 0
  # Hasher
    engines= []
    rp = []
    # Create the repetitions for the LSH
    for i in range(L):
      rps= RandomProjection(window, bin_width, K) 
      #RandomDiscretizedProjections('rp', K, bin_width)
      #engine = Engine(window, lshashes=[rps])
      rp.append(rps)
      #engines.append(engine)

    chunks = [(np.array(time_series), ranges, window, rp) for ranges in np.array_split(np.arange(time_series.shape[0] - window + 1), multiprocessing.cpu_count())]

    hash_mat = np.array([], dtype=np.int8).reshape(0,L,dimension,K)
    subsequences = np.array([]).reshape(0,dimension,window)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
      results = pool.starmap(process_chunk, [chunk for chunk in chunks])

    for index, result in enumerate(results):

      hash_mat_temp, std_temp, mean_temp, sub_temp = result

      subsequences = np.concatenate([subsequences, sub_temp])
      hash_mat = np.concatenate([hash_mat, hash_mat_temp])
      std_container.update(std_temp)
      mean_container.update(mean_temp)
    hash_mat = np.ascontiguousarray(hash_mat, dtype=np.int8)
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
       # pr = cProfile.Profile()
       # pr.enable()
        global stopped_event, dist_comp, b, s, top, failure_thresh
        top_i, dist_comp_i = minhash_cycle(i, j, windowed_ts, hash_mat, k, lsh_threshold)
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
             # print("Stop:", ss_val, length)
              if length >= k and ss_val:
               # pr.disable()
                #pr.print_stats(sort="tottime")

                  #print("set exit")
                stopped_event.set()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(worker, i, j, K, L, bin_width, motif_dimensionality, dimension, k) for i in range(K) for j in range(L)]
        with tqdm(total=L*K, desc="Iteration") as pbar:
            for future in as_completed(futures):
                pbar.update()
                if stopped_event.is_set():  # Check if the stop event is set
                    executor.shutdown(wait= False, cancel_futures= True)
                    break

    return top, dist_comp