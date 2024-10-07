from base import *
from find_bin_width import *
import numpy as np
import queue, threading, multiprocessing, itertools
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from hash_lsh import RandomProjection, euclidean_hash
from numba import jit

def find_motifs_graphs(time_series, window, k, motif_dimensionality, bin_width, lsh_threshold=None, L, K, fail_thresh=0.8):
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

    lock = threading.Lock()

    global stopped_event
    stopped_event = threading.Event()
    stopped_event.clear()

    for i in range(K), j in range(L):
        counter = dict()
        # for each subsequence compare with the subsequences that follow
        # if they match increase the counter of their pair 
        for f in range(n - window + 1):
            if i == 0:
                current = hash_mat[f,j,:,:]
            else:
                current  = hash_mat[f,j,:,-i]
            for k in range(i + 1, n - window + 1):
                if i == 0:
                    eq = np.all(current == hash_mat[k,j,:,:])
                else:
                    eq = np.all(current == hash_mat[k,j,:,-i])
                if eq:
                    counter.setdefault((f,j), 0)
                    counter[(f,j)] += 1
    # Find the max entry in the counter
    maximum_pair = max(counter, key=counter.get())
    # Find the set of dimensions with the minimal distance
        


    return