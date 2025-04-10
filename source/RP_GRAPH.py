from base import (
    create_shared_array,
    WindowedTS,
    process_chunk_graph,
    inner_cycle,
)
from typing import Tuple
from multiprocessing import shared_memory
import numpy as np
import itertools
import bisect
from multiprocessing import cpu_count
from concurrent.futures import as_completed, ProcessPoolExecutor
from hash_lsh import RandomProjection
import time
from stop import stopgraph


def worker(i, j, subsequences, hash_mat_name, ordering_name, ordered_name, k):
    # if i == 0 and j == 1:
    #    pr = cProfile.Profile()
    #   pr.enable()
    # print("Worker: ", i, j)
    dist_comp = 0
    top = []
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
    means_ex = shared_memory.SharedMemory(name=subsequences.avgs)
    stds_ex = shared_memory.SharedMemory(name=subsequences.stds)
    means = np.ndarray(
        (n - window + 1, dimensionality), dtype=np.float32, buffer=means_ex.buf
    )
    stds = np.ndarray(
        (n - window + 1, dimensionality), dtype=np.float32, buffer=stds_ex.buf
    )
    # Ordered hashes, ordering indices and unordered hashes
    existing_arr = shared_memory.SharedMemory(name=ordered_name)
    existing_ord = shared_memory.SharedMemory(name=ordering_name)
    existing_hash = shared_memory.SharedMemory(name=hash_mat_name)
    hash_mat = np.ndarray(
        (n - window + 1, dimensionality, K), dtype=np.int8, buffer=existing_arr.buf
    )
    ordering = np.ndarray(
        (dimensionality, n - window + 1), dtype=np.int32, buffer=existing_ord.buf
    )
    original_mat = np.ndarray(
        (n - window + 1, dimensionality, K), dtype=np.int8, buffer=existing_hash.buf
    )

    dist, pairs, dims, dists, dist_comp = inner_cycle(
        dimensionality,
        ordering,
        hash_mat,
        original_mat,
        time_series,
        window,
        motif_dimensionality,
        i,
        k,
        means,
        stds,
    )
    for d, p, dim, stop_dist in zip(dist, pairs, dims, dists):
        if d == np.inf:
            break
        top.append((-d, [dist_comp, p, dim, stop_dist]))

    ex_time_series.close()
    existing_arr.close()
    existing_ord.close()
    existing_hash.close()
    means_ex.close()
    stds_ex.close()
    # if i == 0 and j == 1:
    #    pr.disable()
    #   pr.print_stats(sort='cumtime')
    return top, dist_comp, i, j  # , counter


def order_hash(hash_mat_name, indices_name, ordered_name, dimension, num_s, K):
    hash_mat_data = shared_memory.SharedMemory(name=hash_mat_name)
    hash_mat = np.ndarray(
        (num_s, dimension, K), dtype=np.int8, buffer=hash_mat_data.buf
    )
    indices_data = shared_memory.SharedMemory(name=indices_name)
    indices = np.ndarray((dimension, num_s), dtype=np.int32, buffer=indices_data.buf)
    ordered_data = shared_memory.SharedMemory(name=ordered_name)
    ordered = np.ndarray((num_s, dimension, K), dtype=np.int8, buffer=ordered_data.buf)
    hash_mat_t = hash_mat.transpose(1, 0, 2)
    for curr_dim in range(dimension):
        curr_slice = hash_mat_t[curr_dim, :, :]
        indices[curr_dim, :] = np.lexsort(
            [curr_slice[:, k] for k in reversed(range(K))]
        )
        ordered[:, curr_dim, :] = curr_slice[indices[curr_dim, :], :]

    hash_mat_data.close()
    indices_data.close()
    ordered_data.close()
    return True


def pmotif_findg(
    time_series_name: str,
    n: int,
    dimension: int,
    window: int,
    k: int,
    motif_dimensionality: int,
    bin_width: int,
    lsh_threshold: float = 0,
    L: int = 200,
    K: int = 8,
    fail_thresh: float = 0.01,
) -> Tuple[list, int, float]:
    """
    Find the top-k motifs in a time series

    :param str time_series_name: The name of the shared memory block containing the time series
    :param int n: The length of the time series
    :param int dimension: The dimensionality of the time series
    :param int window: The length of the motif to find
    :param int k: The number of motifs to find
    :param int motif_dimensionality: The dimensionality of the motif
    :param int bin_width: The bin width for Discretized Random Projections
    :param float lsh_threshold: unused, for compatibility
    :param int L: The number of repetitions of the hashing
    :param int K: The number of concatenations of the hashing
    :param float fail_thresh: The allowed failure probability for the hashing
    :return: A tuple containing the top-k motifs, the number of distance computations and the hashing time
    """
    # pr = cProfile.Profile()
    # pr.enable()
    # Data
    try:
        top = []
        hash_t = 0
        std_container, _ = create_shared_array(
            (n - window + 1, dimension), dtype=np.float32
        )
        mean_container, _ = create_shared_array(
            (n - window + 1, dimension), dtype=np.float32
        )
        indices_container = []
        hash_container = []
        ordered_container = []
        close_container = []

        # Create shared memory for everything
        for _ in range(L):
            arrn, _ = create_shared_array((n - window + 1, dimension, K), dtype=np.int8)
            hash_container.append(arrn.name)
            arro, _ = create_shared_array((n - window + 1, dimension, K), dtype=np.int8)
            ordered_container.append(arro.name)
            arri, _ = create_shared_array((dimension, n - window + 1), dtype=np.int32)
            indices_container.append(arri.name)
            close_container.append(arrn)
            close_container.append(arro)
            close_container.append(arri)

        dist_comp = 0
        # Hasher
        rp = RandomProjection(window, bin_width, K, L)  # []

        chunk_sz = n // (cpu_count() * 2) if n <= 1000000 else np.sqrt(n)
        num_chunks = max(1, n // chunk_sz)
        chunks = [
            (
                time_series_name,
                ranges,
                window,
                rp,
                hash_container,
                L,
                dimension,
                n,
                K,
                mean_container.name,
                std_container.name,
            )
            for ranges in np.array_split(np.arange(n - window + 1), num_chunks)
        ]

        # Hash the subsequences and order them lexigraphically
        st = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=cpu_count(),
            # mp_context = multiprocessing.get_context("forkserver")
        ) as pool:
            # pool.map(process_chunk_graph, chunks)
            results = [pool.submit(process_chunk_graph, *chunk) for chunk in chunks]
            for future in as_completed(results):
                try:
                    future.result()
                except KeyboardInterrupt:
                    pool.shutdown(wait=False, cancel_futures=True)

            data = [
                (split, indices, ordered, dimension, n - window + 1, K)
                for split, indices, ordered in zip(
                    hash_container, indices_container, ordered_container
                )
            ]
            results = [pool.submit(order_hash, *da) for da in data]
            for future in as_completed(results):
                try:
                    _ = future.result()
                    # future.result()
                except KeyboardInterrupt:
                    pool.shutdown(wait=False, cancel_futures=True)
            # print("Ordered")
        # Close the time series otherwise it will be copied in all children processes
        std_container.close()
        mean_container.close()
        del chunks
        hash_t = time.perf_counter() - st
        print("Hashing time: ", hash_t)
        windowed_ts = WindowedTS(
            time_series_name,
            n,
            dimension,
            window,
            mean_container.name,
            std_container.name,
            L,
            K,
            motif_dimensionality,
            bin_width,
        )
        stop_val = False
        # confirmations = 0
        with ProcessPoolExecutor(
            max_workers=cpu_count(),
            # mp_context = multiprocessing.get_context("forkserver")
        ) as executor:
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
                )
                for i, j in itertools.product(range(K), range(L))
            ]
            for future in as_completed(futures):
                try:
                    top_temp, dist_comp_temp, i, j = future.result()
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
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
                            # print(element[0], stored[0])
                            if element[0] > stored[0]:
                                top.remove(stored)
                            # confirmations += 1
                            # elif element[0] == stored[0]:
                            #   confirmations += 1
                            else:
                                add = False
                                continue
                    if add:
                        bisect.insort(top, element, key=lambda x: -x[0])
                    if len(top) > k:
                        top = top[:k]
                if len(top) == k:
                    stop_val = stopgraph(
                        top[-1][1][3],
                        i,
                        j,
                        fail_thresh,
                        K,
                        L,
                        bin_width,
                        motif_dimensionality,
                    )
                    if (
                        stop_val  and (j+1 == L or j+1 == (L//2))
                    ):  # (stop_val or confirmations >= 4) and len(top) >= k:
                        print("i,j: ", i, j)
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
        return top, dist_comp, hash_t
    except (KeyboardInterrupt, FileNotFoundError, OSError):
        return top, dist_comp, hash_t
    finally:
        # pr.disable()
        # pr.print_stats(sort='cumtime')
        # Close all the shared memory
        for arr in close_container:
            arr.close()
            arr.unlink()
        mean_container.unlink()
        std_container.unlink()
