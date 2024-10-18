from base import *
import numpy as np, queue, itertools
from multiprocessing import Pool, cpu_count
from concurrent.futures import as_completed, ProcessPoolExecutor
from hash_lsh import RandomProjection, euclidean_hash
import cProfile
from stop import stopgraph

def conf_sampling(subsequences, i, num_conf, collisions, k):
    collisions = {v: key for v, key in collisions.items() if key >= subsequences.d}
    dimensions = np.arange(subsequences.dimensionality)
    motif_dimensionality = subsequences.d
    top = queue.PriorityQueue(k+1)
    dist_comp = 0
    for elem in collisions:
            dist_comp +=1
            coll_0, coll_1 = elem
            curr_dist, dim, stop_dist= z_normalized_euclidean_distanceg(subsequences.sub(coll_0), subsequences.sub(coll_1),
                                                dimensions, subsequences.mean(coll_0), subsequences.std(coll_0),
                                                subsequences.mean(coll_1), subsequences.std(coll_1), motif_dimensionality)
            top.put((-curr_dist, [dist_comp, elem, [dim], stop_dist]))
            if len(top.queue) > k: top.get()
    
    if top.empty(): return False, None, None
    else:           return True, top, dist_comp


def order_hash(hash_mat, l, dimension):
    for curr_dim in range(dimension):
        # Order hash[:,rep,dim,:] using as key the array of the last dimension
        hash_mat_curr = hash_mat[:,curr_dim,:]
        ordering = np.lexsort(hash_mat_curr.T[::-1])
    return ordering, l

def pmotif_findauto(time_series, window, k, motif_dimensionality, bin_width, lsh_threshold, L, K, fail_thresh=0.1):
    #pr = cProfile.Profile()
    #pr.enable()
  # Data
    dimension = time_series.shape[1]
    n = time_series.shape[0]
    top = queue.PriorityQueue(maxsize=k+1)
    std_container = {}
    mean_container = {}

    
    failure_thresh = fail_thresh
    dist_comp = 0
  # Hasher
    rp = RandomProjection(window, bin_width, K, L) #[]


    chunk_sz = int(np.sqrt(n))
    num_chunks = max(1, n // chunk_sz)
    
    chunks = [(time_series[ranges[0]:ranges[-1]+window], ranges, window, rp) for ranges in np.array_split(np.arange(n - window + 1), num_chunks)]

    shm_hash_mat, hash_mat = create_shared_array((n-window+1, L, dimension, K), dtype=np.int8)
    ordering = np.ndarray((dimension, n - window + 1, L), dtype=np.int32)

    # Hash the subsequences and order them lexigraphically
    with Pool(processes=int(cpu_count())) as pool:
        results = []
        ord_results = []

        for chunk in chunks:
            result = pool.apply_async(process_chunk, (*chunk, shm_hash_mat.name, hash_mat.shape, L, dimension, K))
            results.append(result)

        for result in results:
            std_temp, mean_temp = result.get()
            std_container.update(std_temp)
            mean_container.update(mean_temp)

        for rep in range(L):
          result = pool.apply_async(order_hash, (hash_mat[:,rep,:,:], rep, dimension))
          ord_results.append(result)
        
        for result in ord_results:
            ordering_temp, rep = result.get()
            ordering[:,:,rep] = ordering_temp

    windowed_ts = WindowedTS(time_series, window, mean_container, std_container, L, K, motif_dimensionality, bin_width)


    stop_val = False
    num_forests= np.ceil(np.log2(L)).astype(np.int32)
    trees_per_forest = L // num_forests

    for j in range(1,trees_per_forest):
        colls_dict = {}
        # FInd the smallest i: the first j trees have at most 10ij collisions
        for i in range(K-1):
            max_allowed_collisions = 10*(i+1)*(j+1)*dimension
            forests_ok = 0
            for forest in range(num_forests):
                trees_ok = 0
                for tree in range(j):
                    colls = 0
                    tree_idx = forest*trees_per_forest + tree
                    # !!!!!!! Cycle on the dimensions, check the max collisions !!!!!!!
                    for dim in range(dimension):
                        search = hash_mat[:,tree_idx,dim,:i+1]
                        ordering_dim = ordering[dim,:,tree_idx]
                        search = search[ordering_dim,:]
                        for idx, elem1 in enumerate(search):
                            for idx2, elem2 in enumerate(search[idx+1:]):
                                sub_idx1 = ordering_dim[idx]
                                sub_idx2 = ordering_dim[idx+idx2+1]
                                # No trivial match
                                if (abs(sub_idx1 - sub_idx2) <= window):
                                    continue
                                # If same hash, increase the counter, see the next
                                if np.all((elem1 == elem2)):
                                    # Save also the collision so if needed we have them
                                    colls_dict.setdefault((sub_idx1, sub_idx2),0)
                                    colls_dict[sub_idx1, sub_idx2] += 1
                                    colls += 1
                                else: break
                                # We know this tree can't contribute so stop
                            if colls > max_allowed_collisions: break
                        # If a tree misses, skip this iteration
                        if colls <= max_allowed_collisions: trees_ok += 1
                        else: break
                if trees_ok == j: 
                    forests_ok += 1 
            # If the forest can run confirmation sampling, we already have the collisions
            print(forests_ok, num_forests//4)
            if forests_ok >= num_forests // 4 : 
                fin, result, dist = conf_sampling(windowed_ts, i, num_forests//4, colls_dict, k)
                if fin: return result, dist
                else:
                    continue
                # Skip to next j
            


   # pr.disable()
    #pr.print_stats(sort='cumtime')
    shm_hash_mat.close()
    shm_hash_mat.unlink()
    return top, dist_comp
