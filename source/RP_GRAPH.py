from base import *
from find_bin_width import *
import numpy as np
import queue, threading, itertools
from multiprocessing import Pool, cpu_count
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from hash_lsh import RandomProjection, euclidean_hash
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
            ordered_view = hash_mat_curr[ordering_dim,curr_dim,:]
            # Take the subsequent elements of the ordering and check if their hash is the same
            for idx, elem1 in enumerate(ordered_view):
                for idx2, elem2 in enumerate(ordered_view[idx+1:]):
                    sub_idx1 = ordering_dim[idx]
                    sub_idx2 = ordering_dim[idx+idx2+1]
                    # No trivial match
                    if (abs(sub_idx1 - sub_idx2) < window):
                        continue
                    # If same hash, increase the counter, see the next
                    if (elem1 == elem2).all():
                        counter.setdefault((sub_idx1, sub_idx2), 0)
                        counter[sub_idx1, sub_idx2] += 1
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
                current = hash_mat[f,j,:,:-i]
            for l in range(f + window, n - window + 1):
                if i == 0:
                    eq = np.sum(np.all(current == hash_mat[l,j,:,:], axis=1))
                else:
                    eq = np.sum(np.all(current == hash_mat[l,j,:,:-i], axis=1))
                if eq > 0:
                    counter.setdefault((f,l), 0)
                    counter[(f,l)] += eq
        '''
    # Get all entries whose counter is above or equal the motif dimensionality
        counter = {pair: v for pair, v in counter.items() if v >= motif_dimensionality}
    # Find the set of dimensions with the minimal distance
        for maximum_pair in counter.keys():
            coll_0, coll_1 = maximum_pair
            # Iƒ we already seen it in a key of greater length, skip
            if not i == 0:
                rows = hash_mat[coll_0,j,:,:-i+1] == hash_mat[coll_1,j,:,:-i+1]
                comp = np.sum(np.all(rows, axis=1))
                if comp >= motif_dimensionality:
                    continue            
            dist_comp += 1
            curr_dist, dim, stop_dist= z_normalized_euclidean_distance(subsequences.sub(coll_0), subsequences.sub(coll_1),
                                                np.arange(dimensionality), subsequences.mean(coll_0), subsequences.std(coll_0),
                                                subsequences.mean(coll_1), subsequences.std(coll_1), motif_dimensionality)
            top.put((-curr_dist, [dist_comp, maximum_pair, [dim], stop_dist]))

        if len(top.queue) > k :
            top.queue = top.queue[:k]
        return top, dist_comp

def worker(i, j, windowed_ts, hash_mat, ordering, k, stop_i, failure_thresh):
        if stop_i:
            return
        top_temp, dist_comp_temp = cycle(i, j, windowed_ts, hash_mat, ordering, k, failure_thresh)

        return list(top_temp.queue), dist_comp_temp, i, j

def order_hash(hash_mat, ordering_name, shape, l, dimension):
    existing_order = shared_memory.SharedMemory(name=ordering_name)
    ordering = np.ndarray(shape, dtype=np.int32, buffer=existing_order.buf)

    for curr_dim in range(dimension):
        # Order hash[:,rep,dim,:] using as key the array of the last dimension
        hash_mat_curr = hash_mat[:,l,curr_dim,:]
        ordering[curr_dim,:,l] = np.lexsort(hash_mat_curr.T[::-1])
    return


def pmotif_findg(time_series, window, k, motif_dimensionality, bin_width, lsh_threshold, L, K, fail_thresh=0.8):
    global dimension, failure_thresh, time_tot
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
    shm_ordering, ordering = create_shared_array((dimension, n - window + 1, L), dtype=np.int32)

    with Pool(processes=int(cpu_count())) as pool:
        results = []

        for chunk in chunks:
            result = pool.apply_async(process_chunk, (*chunk, shm_hash_mat.name, hash_mat.shape, L, dimension, K))
            results.append(result)

        for result in results:
            std_temp, mean_temp = result.get()
            std_container.update(std_temp)
            mean_container.update(mean_temp)

        for rep in range(L):
           pool.apply_async(order_hash, (hash_mat, shm_ordering.name, ordering.shape, rep, dimension))



    windowed_ts = WindowedTS(time_series, window, mean_container, std_container, L, K, motif_dimensionality, bin_width)
    for l in range(L):   
        for curr_dim in range(dimension):
        # Order hash[:,rep,dim,:] using as key the array of the last dimension
            hash_mat_curr = hash_mat[:,l,curr_dim,:]
            ordering[curr_dim,:,l] = np.lexsort(hash_mat_curr.T[::-1])
    #print(ordering)
    global stopped_event
    stopped_event = threading.Event()
    stopped_event.clear()

    stop_val = False
    stop_count = 0
    stop_elem = None

    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(worker, i, j, windowed_ts, hash_mat, ordering, k, stopped_event.is_set(), fail_thresh) for i, j in itertools.product(range(K), range(L))]
        for future in as_completed(futures):
            top_temp, dist_comp_temp, i, j = future.result()
            if dist_comp_temp == 0: continue
            dist_comp += dist_comp_temp
            for element in top_temp:
                #Check is there's already an overlapping sequence, in that case keep the best match
                for stored in top.queue:
                    indices_1 = element[1][1]
                    indices_2 = stored[1][1]
                    if (abs(indices_1[0] - indices_2[0]) < window or
                        abs(indices_1[1] - indices_2[1]) < window or
                        abs(indices_1[0] - indices_2[1]) < window or
                        abs(indices_1[1] - indices_2[0]) < window):
                        if element[0] > stored[0]:
                            top.queue.remove(stored)
                            top.put(element)
                        else:
                            continue
                top.put(element)
                if len(top.queue) > k:
                    top.get()
                
            # If the top element of the queue is the same for 10 iterations return
            if stop_elem == top.queue[0]:
                stop_count += 1
            else:
                stop_count = 0
                stop_elem = top.queue[0]
            if stop_count == 20:
                stop_val = True
            if (stop_val and len(top.queue) >= k):
                    stopped_event.set()
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    shm_hash_mat.close()
    shm_ordering.close()
    shm_hash_mat.unlink()
    shm_ordering.unlink()
    return top, dist_comp