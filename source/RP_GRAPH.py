from base import *
from find_bin_width import *
import numpy as np
import queue, threading, multiprocessing, itertools
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from hash_lsh import RandomProjection, euclidean_hash
from numba import jit
from stop import stop3

def cycle(i, j, subsequences, hash_mat, ordering, k, fail_thresh):
        dist_comp = 0
        top = queue.PriorityQueue()
        window = subsequences.w
        n = subsequences.num_sub
        dimensionality = subsequences.dimensionality
        motif_dimensionality = subsequences.d
        L = subsequences.L
        K = subsequences.K 
        bin_width = subsequences.r
        counter = dict()
        hash_mat_curr = hash_mat[:,j,:,:-i] if i != 0 else hash_mat[:,j,:,:]
        # Let's assume that ordering has the lexigraphical order of the dimensions
        for curr_dim in range(dimensionality):
            ordering_dim = ordering[curr_dim,:,j]
            # Take the subsequent elements of the ordering and check if their hash is the same
            for elem_idx, elem1 in enumerate(ordering_dim):
                for elem2 in ordering_dim[elem_idx+1:]:
                    # No trivial match
                    if (abs(elem1 - elem2) <= window):
                        continue
                    # If same hash, increase the counter, see the next
                    if hash_mat_curr[elem1,curr_dim] == hash_mat_curr[elem2,curr_dim]:
                        counter.setdefault((elem1, elem2), 0)
                        counter[(elem1, elem2)] += 1
                    # Else skip because we know that the ordering ensures that the subsequences are different
                    else:
                        break

        '''
        # for each subsequence compare with the subsequences that follow
        # if they match increase the counter of their pair 
        for f in range(n - window + 1):
            if i == 0:
                current = hash_mat[f,j,:,:]
            else:
                current = hash_mat[f,j,:,-i]
            for l in range(f + window, n - window + 1):
                if i == 0:
                    eq = np.sum(np.all(current == hash_mat[l,j,:,:], axis=1))
                else:
                    eq = np.sum(np.all(current == hash_mat[l,j,:,-i], axis=1))
                if eq > 0:
                    counter.setdefault((f,l), 0)
                    counter[(f,l)] += eq
        '''
    # Get all entries whose counter is above or equal the motif dimensionality
        counter = {pair: v for pair, v in counter.items() if v >= motif_dimensionality}
    # Find the set of dimensions with the minimal distance
        for maximum_pair in counter.keys():
            dist_comp += 1
            curr_dist, dim, stop_dist= z_normalized_euclidean_distance(subsequences.sub(maximum_pair[0]), subsequences.sub(maximum_pair[1]),
                                                np.arange(dimensionality), subsequences.mean(maximum_pair[0]), subsequences.std(maximum_pair[0]),
                                                subsequences.mean(maximum_pair[1]), subsequences.std(maximum_pair[1]), motif_dimensionality)
            top.put((-curr_dist, [dist_comp, maximum_pair, [dim], stop_dist]))

        if len(top.queue) > k :
            top.queue = top.queue[:k]
        return top, dist_comp

def worker(i, j, windowed_ts, hash_mat, ordering, k, stop_i, failure_thresh):
        if stop_i:
            return
        top_temp, dist_comp_temp = cycle(i, j, windowed_ts, hash_mat, ordering, k, failure_thresh)
        
        return top_temp.queue, dist_comp_temp, i, j

def pmotif_findg(time_series, window, k, motif_dimensionality, bin_width, lsh_threshold, L, K, fail_thresh=0.8):
    global dist_comp, dimension, top, failure_thresh, time_tot
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

    # Create ordering
    ordering = np.zeros((dimension, n - window + 1, L), dtype=int)
    '''
    for rep in range(L):
        for curr_dim in range(dimension):
            print(hash_mat[:, rep, curr_dim, :])
            print(np.sort(hash_mat[:, rep, curr_dim, :]))
            ordering[curr_dim,:,rep] = np.argsort(hash_mat[:, rep, curr_dim, :])
    print(ordering)
    '''
    lock = threading.Lock()

    global stopped_event
    stopped_event = threading.Event()
    stopped_event.clear()


    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(worker, i, j, windowed_ts, hash_mat, ordering, k, stopped_event.is_set(), fail_thresh) for i, j in itertools.product(range(K), range(L))]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                top_temp, dist_comp_temp, i, j = result
                if dist_comp_temp == 0:
                    continue
                dist_comp += dist_comp_temp
                for elem in top_temp:
                    top.put(elem)
                    if len(top.queue) > k:
                        top.get()
                if stop3(top.queue[0], i, j, failure_thresh, K, L, bin_width, motif_dimensionality):
                    stopped_event.set()
                if stopped_event.is_set():  # Check if the stop event is set
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    shm_hash_mat.close()
    shm_hash_mat.unlink()
    return top, dist_comp