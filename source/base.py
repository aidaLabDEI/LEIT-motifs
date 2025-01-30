import numpy as np
from multiprocessing import shared_memory
from numba import jit, njit
import numba as nb
from hash_lsh import multi_compute_hash
from jitted_utilities import rolling_window, eq, multi_eq


class WindowedTS(object):
    def __init__(
        self,
        subsequences,
        n,
        d,
        window: int,
        rolling_avg,
        rolling_std,
        L: int,
        K: int,
        motif_dimensionality: int,
        bin_width: int,
    ):
        """
        Initialize the WindowedTS object.

        Parameters:
        - subsequences: The time series data.
        - n: The number of subsequences.
        - d: The dimensionality of the time series data.
        - window: The size of the window.
        - rolling_avg: The rolling average of the subsequences of the time series.
        - rolling_std: The rolling standard deviation of the subsequences of the time series.
        - L: The number of LSH repetitions.
        - K: The number of LSH concatenations.
        - motif_dimensionality: The dimensionality of the motif.
        - bin_width: The bin width for the LSH.

        Returns:
        - None
        """
        self.subsequences = subsequences
        self.w = window
        self.avgs = rolling_avg
        self.stds = rolling_std
        self.dimensionality = d
        self.num_sub = n
        self.L = L
        self.K = K
        self.d = motif_dimensionality
        self.r = bin_width

    # Helper functions
    def sub(self, i: int):
        return self.subsequences[i : i + self.w].T

    def mean(self, i: int):
        return self.avgs[i]

    def std(self, i: int):
        return self.stds[i]


@jit(
    nb.types.Tuple((nb.float64, nb.int8[:], nb.float64))(
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int32[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
        nb.int64,
    ),
    nopython=True,
    cache=True,
    fastmath=True,
)
def z_normalized_euclidean_distance(
    ts1, ts2, indices, mean_ts1, std_ts1, mean_ts2, std_ts2, dimensionality=None
):
    """
    Compute the z-normalized Euclidean distance between two subsequences, if a dimensionality is specified the algorithm
    will find the set of dimensions that minimize the distance.

    Parameters:
    ts1 (ndarray): The first subsequence.
    ts2 (ndarray): The second subsequence.
    indices (ndarray): The indices of the dimensions to consider.
    mean_ts1 (ndarray): The mean values of the first subsequence.
    std_ts1 (ndarray): The standard deviation values of the first subsequence.
    mean_ts2 (ndarray): The mean values of the second subsequence.
    std_ts2 (ndarray): The standard deviation values of the second subsequence.
    dimensionality (int, optional): The dimensionality of the result.

    Returns:
    tuple: A tuple containing the sum of the distances, the indices of the dimensions used, and the maximum distance.

    Raises:
    ValueError: If the subsequences have different dimensions.

    """
    # Ensure both time series have the same dimensions
    if ts1.shape != ts2.shape:
        raise ValueError("Time series must have the same dimensions.")

    # Pick the dimensions used in this iteration
    ts1 = ts1[indices]
    ts2 = ts2[indices]

    # Z-normalize each dimension separately
    ts1_normalized = (ts1 - mean_ts1[indices, np.newaxis]) / std_ts1[
        indices, np.newaxis
    ]
    ts2_normalized = (ts2 - mean_ts2[indices, np.newaxis]) / std_ts2[
        indices, np.newaxis
    ]

    # Compute squared differences and sum them
    squared_diff_sum = np.sqrt(
        np.sum(np.square(ts1_normalized - ts2_normalized), axis=1)
    )

    if dimensionality and dimensionality != len(indices):
        min_indices = np.argsort(squared_diff_sum)
        min_indices_corr = min_indices[:dimensionality]
        sum = np.sum(squared_diff_sum[min_indices_corr])

        return (
            sum,
            min_indices_corr.astype(np.int8),
            squared_diff_sum[min_indices_corr[-1]],
        )

    sum = np.sum(squared_diff_sum)
    return sum, indices.astype(np.int8), np.max(squared_diff_sum)


@jit(
    nb.types.Tuple((nb.float32, nb.int8[:], nb.float32[:]))(
        nb.float32[:, :],
        nb.float32[:, :],
        nb.int32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.int64,
    ),
    nopython=True,
    cache=True,
    fastmath=True,
)
def z_normalized_euclidean_distanceg(
    ts1, ts2, indices, mean_ts1, std_ts1, mean_ts2, std_ts2, dimensionality=None
):
    """
    Compute the z-normalized Euclidean distance between two subsequences, if a dimensionality is specified the algorithm
    will find the set of dimensions that minimize the distance.

    Parameters:
    ts1 (ndarray): The first subsequence.
    ts2 (ndarray): The second subsequence.
    indices (ndarray): The indices of the dimensions to consider.
    mean_ts1 (ndarray): The mean values of the first subsequence.
    std_ts1 (ndarray): The standard deviation values of the first subsequence.
    mean_ts2 (ndarray): The mean values of the second subsequence.
    std_ts2 (ndarray): The standard deviation values of the second subsequence.
    dimensionality (int, optional): The dimensionality of the result.

    Returns:
    tuple: A tuple containing the sum of the distances, the indices of the dimensions used, and the dimensional distances.

    Raises:
    ValueError: If the subsequences have different dimensions.

    """
    # Ensure both time series have the same dimensions
    if ts1.shape != ts2.shape:
        raise ValueError("Time series must have the same dimensions.")

    # Pick the dimensions used in this iteration
    ts1 = ts1[:, indices]
    ts2 = ts2[:, indices]

    # Z-normalize each dimension separately
    ts1_normalized = (ts1 - mean_ts1[indices]) / std_ts1[indices]
    ts2_normalized = (ts2 - mean_ts2[indices]) / std_ts2[indices]

    # Compute squared differences and sum them
    squared_diff_sum = np.sqrt(
        np.sum(np.square(ts1_normalized - ts2_normalized), axis=0)
    )

    if dimensionality and dimensionality != len(indices):
        min_indices = np.argsort(squared_diff_sum)
        min_indices_corr = min_indices[:dimensionality]
        sum = np.sum(squared_diff_sum[min_indices_corr])

        return sum, min_indices_corr.astype(np.int8), squared_diff_sum[min_indices_corr]

    sum = np.sum(squared_diff_sum)
    return sum, indices.astype(np.int8), squared_diff_sum


@jit(
    nb.types.Tuple((nb.int8[:], nb.float32[:]))(
        nb.float32[:, :],
        nb.float32[:, :],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
        nb.float32[:],
    ),
    nopython=True,
    cache=True,
    fastmath=True,
)
def z_normalized_euclidean_distancegmulti(
    ts1, ts2, mean_ts1, std_ts1, mean_ts2, std_ts2
):
    """
    Alternative method for the ranged search.
    Compute the z-normalized Euclidean distance between two subsequences, if a dimensionality is specified the algorithm
    will find the set of dimensions that minimize the distance.

    Parameters:
    ts1 (ndarray): The first subsequence.
    ts2 (ndarray): The second subsequence.
    indices (ndarray): The indices of the dimensions to consider.
    mean_ts1 (ndarray): The mean values of the first subsequence.
    std_ts1 (ndarray): The standard deviation values of the first subsequence.
    mean_ts2 (ndarray): The mean values of the second subsequence.
    std_ts2 (ndarray): The standard deviation values of the second subsequence.
    dimensionality (int, optional): The dimensionality of the result.

    Returns:
    tuple: A tuple containing the indices of the ordered dimensions and distances.

    Raises:
    ValueError: If the subsequences have different dimensions.

    """
    # Ensure both time series have the same dimensions
    if ts1.shape != ts2.shape:
        raise ValueError("Time series must have the same dimensions.")

    # Z-normalize each dimension separately
    ts1_normalized = (ts1 - mean_ts1) / std_ts1
    ts2_normalized = (ts2 - mean_ts2) / std_ts2

    # Compute squared differences and sum them
    squared_diff_sum = np.sqrt(
        np.sum(np.square(ts1_normalized - ts2_normalized), axis=0)
    )

    min_indices = np.argsort(squared_diff_sum)

    return min_indices.astype(np.int8), np.sort(squared_diff_sum)


def find_collisions(lsh, query_signature):
    """
    Finds potential collisions in the LSH index for a given query signature.

    Parameters:
    - lsh: The LSH index to query.
    - query_signature: The query signature to search for collisions.

    Returns:
    - result: The potential collisions found in the LSH index.
    """
    result = lsh.query(query_signature)

    return result


def create_shared_array(shape, dtype=np.float32):
    """
    Create a shared memory array with the given shape and data type.

    Parameters:
    shape (tuple): The shape of the array.
    dtype (data type, optional): The data type of the array. Defaults to np.float64.

    Returns:
    tuple: A tuple containing the shared memory object and the numpy array.

    """
    size = int(np.prod(shape) * np.dtype(dtype).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=size)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, array


def process_chunk(
    time_series,
    ranges,
    window,
    rp,
    shm_name_hash_mat,
    shm_shape_hash_mat,
    L,
    dimension,
    K,
):
    """
    Process a chunk of time series data.

    Args:
      time_series (numpy.ndarray): The time series chunk.
      ranges (list): The indices of the original time series data to process.
      window (int): The size of the window.
      rp (float): The random projection hasher.
      shm_name_hash_mat (str): The name of the shared memory for the hash matrix.
      shm_shape_hash_mat (tuple): The shape of the shared memory for the hash matrix.
      shm_name_subsequences (str): The name of the shared memory for the subsequences.
      shm_shape_subsequences (tuple): The shape of the shared memory for the subsequences.
      L (int): The length of the subsequences.
      dimension (int): The dimension of the time series data.
      K (int): The number of hash functions.

    Returns:
      tuple: A tuple containing the standard deviation container and the mean container.
    """
    existing_shm_hash_mat = shared_memory.SharedMemory(name=shm_name_hash_mat)

    hash_mat = np.ndarray(
        shm_shape_hash_mat, dtype=np.int8, buffer=existing_shm_hash_mat.buf
    )
    mean_container = {}
    std_container = {}

    for idx_ts, idx in enumerate(ranges):
        subsequence = np.ascontiguousarray(
            time_series[idx_ts : idx_ts + window].T, dtype=np.float32
        )

        mean_container[idx] = np.mean(subsequence, axis=1)
        std_held = np.std(subsequence, axis=1)
        std_container[idx] = np.where(std_held == 0, 0.00001, std_held)

        subsequence_n = (
            subsequence - mean_container[idx][:, np.newaxis]
        ) / std_container[idx][:, np.newaxis]

        hash_mat[idx] = multi_compute_hash(
            subsequence_n, rp.a_l, rp.b_l, rp.a_r, rp.b_r, rp.r, rp.K, rp.L
        )

    existing_shm_hash_mat.close()
    return std_container, mean_container


def hash_timeseries_cyclicconv(
    time_series_name, ranges, window, rp, hash_names, L, dimension, n, K
):
    exist_ts = shared_memory.SharedMemory(name=time_series_name)
    time_series = np.ndarray((n, dimension), dtype=np.float32, buffer=exist_ts.buf)

    # Compute the fourier transform of the time series and of the reversed vectors padded with zeros so to have the same length as the time series
    fft_ts = np.fft.fft(time_series, axis=0)
    vectors_l = rp.a_l  # (sqrt(L), K/2, window)
    vectors_r = rp.a_r  # (sqrt(L), K/2, window)
    products_l = np.zeros((n, K // 2, int(np.sqrt(L)), dimension), dtype=np.complex64)
    products_r = np.zeros((n, K // 2, int(np.sqrt(L)), dimension), dtype=np.complex64)

    for dim in range(dimension):
        for i in range(K // 2):
            for j in range(int(np.sqrt(L))):
                products_l[:, i, j, dim] = np.fft.ifft(
                    np.fft.fft(np.pad(np.flip(vectors_l[j, i])), (0, n))
                    * fft_ts[:, dim]
                )
                products_r[:, i, j, dim] = np.fft.ifft(
                    np.fft.fft(np.pad(np.flip(vectors_r[j, i])), (0, n))
                )

    # The element- wise product of the vector holds in position i the dot product of the vector with Ts[i:i+window]

    # Compute the K/2 hashes for both collections

    # Use the results to construct L hashes of length K for each subsequence of length window for each dimension of the time series

    pass


def process_chunk_graph(
    time_series_name, ranges, window, rp, hash_names, L, dimension, n, K, mean, std
):
    """
    Process a chunk of time series data.

    Params:
      time_series (numpy.ndarray): The time series chunk.
      ranges (list): The indices of the original time series data to process.
      window (int): The size of the window.
      rp (float): The random projection hasher.
      shm_name_hash_mat (str): The name of the shared memory for the hash matrix.
      shm_shape_hash_mat (tuple): The shape of the shared memory for the hash matrix.
      shm_name_subsequences (str): The name of the shared memory for the subsequences.
      shm_shape_subsequences (tuple): The shape of the shared memory for the subsequences.
      L (int): The length of the subsequences.
      dimension (int): The dimension of the time series data.
      K (int): The number of hash functions.

    Returns:
      tuple: A tuple containing the standard deviation container and the mean container.
    """

    # Open all the shared memory objects
    shm_hashes = [
        shared_memory.SharedMemory(name=hash_name) for hash_name in hash_names
    ]
    hash_arrs = [
        np.ndarray((n - window + 1, dimension, K), dtype=np.int8, buffer=shm.buf)
        for shm in shm_hashes
    ]
    mean_existing_shm = shared_memory.SharedMemory(name=mean)
    mean_container = np.ndarray(
        (n - window + 1, dimension), dtype=np.float32, buffer=mean_existing_shm.buf
    )
    std_existing_shm = shared_memory.SharedMemory(name=std)
    std_container = np.ndarray(
        (n - window + 1, dimension), dtype=np.float32, buffer=std_existing_shm.buf
    )
    exist_ts = shared_memory.SharedMemory(name=time_series_name)
    time_series = np.ndarray((n, dimension), dtype=np.float32, buffer=exist_ts.buf)
    try:
        for d in range(dimension):
            mean_container[ranges[0] : ranges[-1], d] = np.mean(
                rolling_window(
                    time_series[ranges[0] : ranges[-1] + window - 1, d], window
                ),
                axis=-1,
            )
            std_container[ranges[0] : ranges[-1], d] = np.nanstd(
                rolling_window(
                    time_series[ranges[0] : ranges[-1] + window - 1, d], window
                ),
                axis=-1,
            )
        for idx in ranges:
            subsequence = time_series[idx : idx + window]

            std_container[idx] = np.where(
                std_container[idx] == 0, 0.00001, std_container[idx]
            )

            subsequence_n = (subsequence - mean_container[idx]) / std_container[idx]
            subsequence_n = np.ascontiguousarray(subsequence_n.T, dtype=np.float32)
            hashed_sub = multi_compute_hash(
                subsequence_n, rp.a_l, rp.b_l, rp.a_r, rp.b_r, rp.r, rp.K, rp.L
            )
            for rep in range(L):
                hash_arrs[rep][idx] = hashed_sub[rep]
    # Close all the shared memory objects
    except Exception as e:
        print(e)
    finally:
        for shm in shm_hashes:
            shm.close()
        exist_ts.close()
        mean_existing_shm.close()
        std_existing_shm.close()
    # print("Finished processing chunk", ranges[0], ranges[-1])
    # return True


@njit(
    nb.types.Tuple(
        (nb.float32[:], nb.int32[:, :], nb.int8[:, :], nb.float32[:, :], nb.int32)
    )(
        nb.int32,
        nb.int32[:, :],
        nb.int8[:, :, :],
        nb.int8[:, :, :],
        nb.float32[:, :],
        nb.int32,
        nb.int32,
        nb.int32,
        nb.int32,
        nb.float32[:, :],
        nb.float32[:, :],
    ),
    fastmath=True,
    cache=True,
)
def inner_cycle(
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
):
    """
    Use the hash matrix to find the collisions

    :param int32 dimensionality: The dimensionality of the time series data.
    :param int32[:,:] ordering: The lexicographical ordering over the dimensions of the hashes.
    :param int8[:,:,:] hash_mat: The hash matrix, ordered for each dimension.
    :param int8[:,:,:] original_mat: The original matrix.
    :param float32[:,:] time_series: The time series data.
    :param int32 window: The size of the window.
    :param int32 motif_dimensionality: The dimensionality of the motif.
    :param int32 i: The current iteration.
    :param int32 k: The number of top motifs to return.
    :param float32[:,:] means: The means of the time series data.
    :param float32[:,:] stds: The standard deviations of the time series data.

    :return: The top distances, the pairs of indices, the dimensions used, the distances, and the total number of distance computations.
    """
    dist_comp = 0
    top_dist = np.full(k, np.inf, dtype=np.float32)
    top_pairs = np.full((k, 2), -1, dtype=np.int32)
    top_dims = np.full((k, motif_dimensionality), -1, dtype=np.int8)
    top_dists = np.full((k, motif_dimensionality), np.inf, dtype=np.float32)
    dimensions = np.arange(dimensionality, dtype=np.int32)
    for curr_dim in range(dimensionality):
        ordering_dim = ordering[curr_dim]
        hash_mat_curr = (
            hash_mat[:, curr_dim, :-i] if i != 0 else hash_mat[:, curr_dim, :]
        )
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
                if eq(elem1, elem2):
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
                    if multi_eq(tot_hash1, tot_hash2) >= motif_dimensionality:
                        dist_comp += 1
                        # print("Comparing: ", sub_idx1, sub_idx2)
                        curr_dist, dim, stop_dist = z_normalized_euclidean_distanceg(
                            time_series[sub_idx1 : sub_idx1 + window],
                            time_series[sub_idx2 : sub_idx2 + window],
                            dimensions,
                            means[sub_idx1],
                            stds[sub_idx1],
                            means[sub_idx2],
                            stds[sub_idx2],
                            motif_dimensionality,
                        )
                        # Insert the new distance into the sorted top distances
                        if (
                            curr_dist < top_dist[0]
                        ):  # Check against the largest value in top k
                            for insert_idx in range(k):
                                if curr_dist < top_dist[insert_idx]:
                                    # Shift elements to the right to make space for the new entry
                                    top_dist[1 : insert_idx + 1] = top_dist[:insert_idx]
                                    top_pairs[1 : insert_idx + 1] = top_pairs[
                                        :insert_idx
                                    ]
                                    top_dims[1 : insert_idx + 1] = top_dims[:insert_idx]
                                    top_dists[1 : insert_idx + 1] = top_dists[
                                        :insert_idx
                                    ]

                                    # Insert new values
                                    top_dist[insert_idx] = curr_dist
                                    top_pairs[insert_idx] = maximum_pair
                                    top_dims[insert_idx] = dim
                                    top_dists[insert_idx] = stop_dist
                                    break
                else:
                    break
    return top_dist, top_pairs, top_dims, top_dists, dist_comp


@njit(
    nb.types.Tuple(
        (
            nb.float32[:, :],
            nb.int32[:, :, :],
            nb.int8[:, :, :],
            nb.float32[:, :, :],
            nb.int32,
        )
    )(
        nb.int32,
        nb.int32[:, :],
        nb.int8[:, :, :],
        nb.int8[:, :, :],
        nb.float32[:, :],
        nb.int32,
        nb.int32,
        nb.int32,
        nb.int32,
        nb.int32,
        nb.float32[:, :],
        nb.float32[:, :],
    ),
    fastmath=True,
    cache=True,
)
def inner_cycle_multi(
    dimensionality,
    ordering,
    hash_mat,
    original_mat,
    time_series,
    window,
    motif_low,
    motif_high,
    i,
    k,
    means,
    stds,
):
    """
    Use the hash matrix to find the collisions

    :param int32 dimensionality: The dimensionality of the time series data.
    :param int32[:,:] ordering: The lexicographical ordering over the dimensions of the hashes.
    :param int8[:,:,:] hash_mat: The hash matrix, ordered for each dimension.
    :param int8[:,:,:] original_mat: The original matrix.
    :param float32[:,:] time_series: The time series data.
    :param int32 window: The size of the window.
    :param int32 motif_dimensionality: The dimensionality of the motif.
    :param int32 i: The current iteration.
    :param int32 k: The number of top motifs to return.
    :param float32[:,:] means: The means of the time series data.
    :param float32[:,:] stds: The standard deviations of the time series data.

    :return: The top distances, the pairs of indices, the dimensions used, the distances, and the total number of distance computations.
    """
    dist_comp = 0
    top_dist = np.full((k, motif_high - motif_low + 1), np.inf, dtype=np.float32)
    top_pairs = np.full((k, motif_high - motif_low + 1, 2), -1, dtype=np.int32)
    top_dims = np.full((k, motif_high - motif_low + 1, motif_high), -1, dtype=np.int8)
    top_dists = np.full(
        (k, motif_high - motif_low + 1, motif_high), -1, dtype=np.float32
    )
    range_dim = np.arange(motif_high - motif_low + 1)
    for curr_dim in range(dimensionality):
        ordering_dim = ordering[curr_dim]
        hash_mat_curr = (
            hash_mat[:, curr_dim, :-i] if i != 0 else hash_mat[:, curr_dim, :]
        )
        original_mat_curr = original_mat[:, :, :-i] if i != 0 else original_mat[:, :, :]
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
                if eq(elem1, elem2):
                    tot_hash1 = original_mat_curr[sub_idx1]
                    tot_hash2 = original_mat_curr[sub_idx2]
                    if multi_eq(tot_hash1, tot_hash2) >= motif_low:
                        dist_comp += 1
                        dim, stop_dist = z_normalized_euclidean_distancegmulti(
                            time_series[sub_idx1 : sub_idx1 + window],
                            time_series[sub_idx2 : sub_idx2 + window],
                            means[sub_idx1],
                            stds[sub_idx1],
                            means[sub_idx2],
                            stds[sub_idx2],
                        )

                        curr_dists = np.cumsum(stop_dist[:motif_high])

                        for subdim in range_dim:
                            curr_dist = curr_dists[subdim]
                            # Insert the new distance into the sorted top distances
                            if (
                                curr_dist < top_dist[0, subdim]
                            ):  # Check against the largest value in top k
                                for insert_idx in range(k):
                                    if curr_dist < top_dist[insert_idx, subdim]:
                                        # Shift elements to the right to make space for the new entry
                                        top_dist[1 : insert_idx + 1, subdim] = top_dist[
                                            :insert_idx, subdim
                                        ]
                                        top_pairs[1 : insert_idx + 1, subdim] = (
                                            top_pairs[:insert_idx, subdim]
                                        )
                                        top_dims[1 : insert_idx + 1, subdim] = top_dims[
                                            :insert_idx, subdim
                                        ]
                                        top_dists[1 : insert_idx + 1, subdim] = (
                                            top_dists[:insert_idx, subdim]
                                        )

                                        # Insert new values
                                        top_dist[insert_idx, subdim] = curr_dist
                                        top_pairs[insert_idx, subdim] = maximum_pair
                                        top_dims[
                                            insert_idx, subdim, : subdim + motif_low
                                        ] = dim[: subdim + motif_low]
                                        top_dists[
                                            insert_idx, subdim, : subdim + motif_low
                                        ] = stop_dist[: subdim + motif_low]
                                        break
                else:
                    break
    return top_dist, top_pairs, top_dims, top_dists, dist_comp


@njit(
    nb.types.Tuple(
        (
            nb.float32[:, :],
            nb.int32[:, :, :],
            nb.int8[:, :, :],
            nb.float32[:, :, :],
            nb.int32,
        )
    )(
        nb.int32,
        nb.int32[:, :],
        nb.int8[:, :, :],
        nb.int8[:, :, :],
        nb.float32[:, :],
        nb.int32,
        nb.int32,
        nb.int32,
        nb.int32,
        nb.int32,
        nb.float32[:, :],
        nb.float32[:, :],
    ),
    fastmath=True,
    cache=True,
)
def inner_cycle_multi_dict(
    dimensionality,
    ordering,
    hash_mat,
    original_mat,
    time_series,
    window,
    motif_low,
    motif_high,
    i,
    k,
    means,
    stds,
):
    """
    Use the hash matrix to find the collisions

    :param int32 dimensionality: The dimensionality of the time series data.
    :param int32[:,:] ordering: The lexicographical ordering over the dimensions of the hashes.
    :param int8[:,:,:] hash_mat: The hash matrix, ordered for each dimension.
    :param int8[:,:,:] original_mat: The original matrix.
    :param float32[:,:] time_series: The time series data.
    :param int32 window: The size of the window.
    :param int32 motif_dimensionality: The dimensionality of the motif.
    :param int32 i: The current iteration.
    :param int32 k: The number of top motifs to return.
    :param float32[:,:] means: The means of the time series data.
    :param float32[:,:] stds: The standard deviations of the time series data.

    :return: The top distances, the pairs of indices, the dimensions used, the distances, and the total number of distance computations.
    """
    dist_comp = 0
    couples ={}
    top_dist = np.full((k, motif_high - motif_low + 1), np.inf, dtype=np.float32)
    top_pairs = np.full((k, motif_high - motif_low + 1, 2), -1, dtype=np.int32)
    top_dims = np.full((k, motif_high - motif_low + 1, motif_high), -1, dtype=np.int8)
    top_dists = np.full(
        (k, motif_high - motif_low + 1, motif_high), -1, dtype=np.float32
    )
    range_dim = np.arange(motif_high - motif_low + 1)
    for curr_dim in range(dimensionality):
        ordering_dim = ordering[curr_dim]
        hash_mat_curr = (
            hash_mat[:, curr_dim, :-i] if i != 0 else hash_mat[:, curr_dim, :]
        )
        for idx, elem1 in enumerate(hash_mat_curr):
            for idx2, elem2 in enumerate(hash_mat_curr[idx + 1 :]):
                sub_idx1 = ordering_dim[idx]
                sub_idx2 = ordering_dim[idx + idx2 + 1]
                maximum_pair = (
                    (sub_idx1, sub_idx2)
                    if sub_idx1 < sub_idx2
                    else (sub_idx2, sub_idx1)
                )
                # No trivial match
                if maximum_pair[1] - maximum_pair[0] <= window:
                    continue
                if eq(elem1, elem2):
                    if maximum_pair not in couples:
                        couples[maximum_pair] = 1
                    else:
                        couples[maximum_pair] += 1
                else:
                    break

        for couple in couples:
                if couples[couple] >= motif_low:
                    dist_comp += 1
                    sub_idx1, sub_idx2 = couple
                    dim, stop_dist = z_normalized_euclidean_distancegmulti(
                        time_series[sub_idx1 : sub_idx1 + window],
                        time_series[sub_idx2 : sub_idx2 + window],
                        means[sub_idx1],
                        stds[sub_idx1],
                        means[sub_idx2],
                        stds[sub_idx2],
                    )
                    curr_dists = np.cumsum(stop_dist[:motif_high])

                    for subdim in range_dim:
                        curr_dist = curr_dists[subdim]
                        # Insert the new distance into the sorted top distances
                        if (
                            curr_dist < top_dist[0, subdim]
                        ):  # Check against the largest value in top k
                            for insert_idx in range(k):
                                if curr_dist < top_dist[insert_idx, subdim]:
                                    # Shift elements to the right to make space for the new entry
                                    top_dist[1 : insert_idx + 1, subdim] = top_dist[
                                        :insert_idx, subdim
                                    ]
                                    top_pairs[1 : insert_idx + 1, subdim] = (
                                        top_pairs[:insert_idx, subdim]
                                    )
                                    top_dims[1 : insert_idx + 1, subdim] = top_dims[
                                        :insert_idx, subdim
                                    ]
                                    top_dists[1 : insert_idx + 1, subdim] = (
                                        top_dists[:insert_idx, subdim]
                                    )

                                    # Insert new values
                                    top_dist[insert_idx, subdim] = curr_dist
                                    top_pairs[insert_idx, subdim] = maximum_pair
                                    top_dims[
                                        insert_idx, subdim, : subdim + motif_low
                                    ] = dim[: subdim + motif_low]
                                    top_dists[
                                        insert_idx, subdim, : subdim + motif_low
                                    ] = stop_dist[: subdim + motif_low]
                                    break

    return top_dist, top_pairs, top_dims, top_dists, dist_comp

