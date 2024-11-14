from base import (
    create_shared_array,
    WindowedTS,
    z_normalized_euclidean_distanceg,
    process_chunk_graph,
)
from multiprocessing import shared_memory
import numpy as np
import queue
import itertools
import bisect
from multiprocessing import Pool, cpu_count
from concurrent.futures import as_completed, ProcessPoolExecutor
from hash_lsh import RandomProjection
import time
from stop import stopgraph
from cachetools import LRUCache


def worker(
    i, j, subsequences, hash_mat_name, ordering_name, ordered_name, k, failure_thresh
):
    # if i == 0 and j == 1:
    #    pr = cProfile.Profile()
    #   pr.enable()
    #print("Worker: ", i, j)
    dist_comp = 0
    top = queue.PriorityQueue(k + 1)
    window = subsequences.w
    n = subsequences.num_sub
    dimensionality = subsequences.dimensionality
    motif_dimensionality = subsequences.d
    K = subsequences.K

    # Time Series
    ex_time_series = shared_memory.SharedMemory(name=subsequences.subsequences)
    time_series = np.ndarray(
        (n, dimensionality), dtype=np.float32, buffer=ex_time_series.buf
    )
    # Utility data
    dimensions = np.arange(dimensionality, dtype=np.int32)
    means_ex = shared_memory.SharedMemory(name=subsequences.avgs.name)
    stds_ex = shared_memory.SharedMemory(name=subsequences.stds.name)
    means = np.ndarray((n-window+1, dimensionality), dtype=np.float32, buffer=means_ex.buf)
    stds = np.ndarray((n-window+1, dimensionality), dtype=np.float32, buffer=stds_ex.buf)
    # Ordered hashes, ordering indices and unordered hashes
    existing_arr = shared_memory.SharedMemory(name=ordered_name.name)
    existing_ord = shared_memory.SharedMemory(name=ordering_name.name)
    existing_hash = shared_memory.SharedMemory(name=hash_mat_name.name)
    hash_mat = np.ndarray(
        (n - window + 1, dimensionality, K), dtype=np.int8, buffer=existing_arr.buf
    )
    ordering = np.ndarray(
        (dimensionality, n - window + 1), dtype=np.int32, buffer=existing_ord.buf
    )
    original_mat = np.ndarray(
        (n - window + 1, dimensionality, K), dtype=np.int8, buffer=existing_hash.buf
    )
    seen = LRUCache(maxsize=n)
    # hash_mat_curr = hash_mat[:,:,:-i] if i != 0 else hash_mat
    # Let's assume that ordering has the lexigraphical order of the dimensions
    for curr_dim in range(dimensionality):
        ordering_dim = ordering[curr_dim, :]
        hash_mat_curr = (
            hash_mat[:, curr_dim, :-i] if i != 0 else hash_mat[:, curr_dim, :]
        )
        # Take the subsequent elements of the ordering and check if their hash is the same
        for idx, elem1 in enumerate(hash_mat_curr):
            for idx2, elem2 in enumerate(hash_mat_curr[idx + 1 :]):
                sub_idx1 = ordering_dim[idx]
                sub_idx2 = ordering_dim[idx + idx2 + 1]
                maximum_pair = (
                    [sub_idx1, sub_idx2]
                    if sub_idx1 < sub_idx2
                    else [sub_idx2, sub_idx1]
                )
                # No trivial match
                if maximum_pair[1] - maximum_pair[0] <= window:
                    continue
                # If same hash, check if there's a collision, see the next after
                if all(elem1 == elem2):
                    if tuple(maximum_pair) in seen:
                        seen[tuple(maximum_pair)] = True # Confirm so that the LRU can function
                        continue
                    tot_hash1 = (
                        original_mat[sub_idx1, :, :-i]
                        if i != 0
                        else original_mat[sub_idx1]
                    )
                    tot_hash2 = (
                        original_mat[sub_idx2, :, :-i]
                        if i != 0
                        else original_mat[sub_idx2]
                    )
                    if (
                        np.sum((tot_hash1 == tot_hash2).all(axis=1))
                        >= motif_dimensionality
                    ):
                        dist_comp += 1
                        # print("Comparing: ", sub_idx1, sub_idx2)
                        curr_dist, dim, stop_dist = z_normalized_euclidean_distanceg(
                            time_series[sub_idx1 : sub_idx1 + window].T,
                            time_series[sub_idx2 : sub_idx2 + window].T,
                            dimensions,
                            means[sub_idx1],
                            stds[sub_idx1],
                            means[sub_idx2],
                            stds[sub_idx2],
                            motif_dimensionality,
                        )
                        top.put(
                            (
                                -curr_dist,
                                [dist_comp, maximum_pair, [dim], stop_dist],
                            )
                        )
                        if top.qsize() > k:
                            top.get()
                    seen[tuple(maximum_pair)] = True
                # Otherwise we know that this subsequence and all the following ones will have different hashes
                else:
                    break

    ex_time_series.close()
    existing_arr.close()
    existing_ord.close()
    existing_hash.close()
    means_ex.close()
    stds_ex.close()
    # if i == 0 and j == 1:
    #    pr.disable()
    #   pr.print_stats(sort='cumtime')
    #print("Worker: ", i, j, "done")
    return top.queue, dist_comp, i, j  # , counter


def order_hash(
    hash_mat_name, indices_name, ordered_name, l_current, dimension, num_s, K
):
    for hash_name, indices_n, ordered_n in zip(
        hash_mat_name, indices_name, ordered_name
    ):
        hash_mat_data = shared_memory.SharedMemory(name=hash_name.name)
        hash_mat = np.ndarray(
            (num_s, dimension, K), dtype=np.int8, buffer=hash_mat_data.buf
        )
        indices_data = shared_memory.SharedMemory(name=indices_n.name)
        indices = np.ndarray(
            (dimension, num_s), dtype=np.int32, buffer=indices_data.buf
        )
        ordered_data = shared_memory.SharedMemory(name=ordered_n.name)
        ordered = np.ndarray(
            (num_s, dimension, K), dtype=np.int8, buffer=ordered_data.buf
        )
        for curr_dim in range(dimension):
            indices[curr_dim, :] = np.lexsort(hash_mat[:, curr_dim, :].T[::-1])
            ordered[:, curr_dim, :] = hash_mat[indices[curr_dim, :], curr_dim, :]

        # Assign the ordering to the shared memory in one go
        # hash_mat = hash_mat[indices,:]

        hash_mat_data.close()
        indices_data.close()
        ordered_data.close()
    return l_current


def pmotif_findg(
    time_series_name,
    n,
    dimension,
    window,
    k,
    motif_dimensionality,
    bin_width,
    lsh_threshold,
    L,
    K,
    fail_thresh=0.1,
):
    # pr = cProfile.Profile()
    # pr.enable()
    time_series_data = shared_memory.SharedMemory(name=time_series_name)
    time_series = np.ndarray(
        (n, dimension), dtype=np.float32, buffer=time_series_data.buf
    )
    # Data
    top = []  # queue.PriorityQueue(maxsize=k+1)
    std_container, _ = create_shared_array((n - window + 1, dimension), dtype=np.float32)
    mean_container, _ = create_shared_array((n - window + 1, dimension), dtype=np.float32)
    indices_container = []
    hash_container = []
    ordered_container = []

    # Create shared memory for everything
    for _ in range(L):
        arrn, _ = create_shared_array((n - window + 1, dimension, K), dtype=np.int8)
        hash_container.append(arrn)
        arro, _ = create_shared_array((n - window + 1, dimension, K), dtype=np.int8)
        ordered_container.append(arro)
        arri, _ = create_shared_array((dimension, n - window + 1), dtype=np.int32)
        indices_container.append(arri)

    dist_comp = 0
    # Hasher
    rp = RandomProjection(window, bin_width, K, L)  # []

    chunk_sz = int(np.sqrt(n))
    num_chunks = max(1, n // chunk_sz)

    chunks = [
        (time_series[ranges[0] : ranges[-1] + window], ranges, window, rp)
        for ranges in np.array_split(np.arange(n - window + 1), num_chunks)
    ]
    # ordering = np.ndarray((dimension, n - window + 1, L), dtype=np.int32)

    # Hash the subsequences and order them lexigraphically
    st = time.process_time()
    with Pool() as pool:
        results = []
        ord_results = []

        for chunk in chunks:
            result = pool.apply_async(
                process_chunk_graph, (*chunk, hash_container, L, dimension, n, K, mean_container, std_container)
            )
            results.append(result)

        for result in results:
            _ = result.get()

        sizeL = int(np.sqrt(L))
        splitted_hash = np.array_split(hash_container, sizeL)
        splitted_indices = np.array_split(indices_container, sizeL)
        splitted_ordered = np.array_split(ordered_container, sizeL)
        for split, indices, ordered in zip(
            splitted_hash, splitted_indices, splitted_ordered
        ):
            result = pool.apply_async(
                order_hash,
                (split, indices, ordered, sizeL, dimension, n - window + 1, K),
            )
            ord_results.append(result)

        for result in ord_results:
            _ = result.get()
    # Close the time series otherwise it will be copied in all children processes
    time_series_data.close()
    std_container.close()
    mean_container.close()
    del chunks

    hash_t = time.process_time() - st
    windowed_ts = WindowedTS(
        time_series_name,
        n,
        dimension,
        window,
        mean_container,
        std_container,
        L,
        K,
        motif_dimensionality,
        bin_width,
    )
    stop_val = False
    # counter_tot = dict()
    # Non parallelized version
    """
    for i, j in itertools.product(range(K), range(L)):
        top_temp, dist_comp_temp, _, _ = worker(
            i,
            j,
            windowed_ts,
            hash_container[j],
            indices_container[j],
            ordered_container[j],
            k,
            fail_thresh,
        )
        print(top_temp)
        dist_comp += dist_comp_temp
        for element in top_temp:
            add = True
            # Check is there's already an overlapping sequence, in that case keep the best match
            for stored in top:
                indices_1_0 = element[1][1][0]
                indices_1_1 = element[1][1][1]
                indices_2_0 = stored[1][1][0]
                indices_2_1 = stored[1][1][1]
                if (
                    (abs(indices_1_0 - indices_2_0) < window)
                    or (abs(indices_1_0 - indices_2_1) < window)
                    or (abs(indices_1_1 - indices_2_0) < window)
                    or (abs(indices_1_1 - indices_2_1) < window)
                ):
                    if element[0] > stored[0]:
                        top.remove(stored)
                    else:
                        add = False
                        continue
            if add:
                bisect.insort(top, element, key=lambda x: -x[0])
            if len(top) > k:
                top = top[:k]
        if len(top) == k:
            stop_val = stopgraph(
                top[0], i, j, fail_thresh, K, L, bin_width, motif_dimensionality
            )
            if stop_val and len(top) >= k:
                break


    # Cycle for the hash repetitions and concatenations
    """
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [
            executor.submit(
                worker,
                i,
                j,
                windowed_ts,
                hash_container[j],
                indices_container[j],
                ordered_container[j],
                k,
                fail_thresh,
            )
            for i, j in itertools.product(range(K), range(L))
        ]
        for future in as_completed(futures):
            try:
                top_temp, dist_comp_temp, i, j = future.result()
            except FileNotFoundError:
                continue
            print(top_temp)

            # counter_tot.update(counter)
            dist_comp += dist_comp_temp
            for element in top_temp:
                add = True
                # Check is there's already an overlapping sequence, in that case keep the best match
                for stored in top:
                    indices_1_0 = element[1][1][0]
                    indices_1_1 = element[1][1][1]
                    indices_2_0 = stored[1][1][0]
                    indices_2_1 = stored[1][1][1]
                    if (
                        (abs(indices_1_0 - indices_2_0) < window)
                        or (abs(indices_1_0 - indices_2_1) < window)
                        or (abs(indices_1_1 - indices_2_0) < window)
                        or (abs(indices_1_1 - indices_2_1) < window)
                    ):
                        if element[0] > stored[0]:
                            top.remove(stored)
                        else:
                            add = False
                            continue
                if add:
                    bisect.insort(top, element, key=lambda x: -x[0])
                if len(top) > k:
                    top = top[:k]
            if len(top) == k:
                stop_val = stopgraph(
                    top[0], i, j, fail_thresh, K, L, bin_width, motif_dimensionality
                )
                if stop_val and len(top) >= k:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    # pr.disable()
    # pr.print_stats(sort='cumtime')
    for arr in hash_container:
        arr.close()
        arr.unlink()
    for arr in indices_container:
        arr.close()
        arr.unlink()
    for arr in ordered_container:
        arr.close()
        arr.unlink()

    mean_container.unlink()
    std_container.unlink()
    return top, dist_comp, hash_t
