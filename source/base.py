import numpy as np
from multiprocessing import shared_memory
from numba import jit
import numba as nb
from hash_lsh import compute_hash


class WindowedTS:
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

        return sum, min_indices_corr.astype(np.int8), squared_diff_sum[min_indices_corr]

    sum = np.sum(squared_diff_sum)
    return sum, indices.astype(np.int8), squared_diff_sum


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
        hashed_sub = np.apply_along_axis(
            compute_hash,
            1,
            subsequence_n,
            rp.a_l,
            rp.b_l,
            rp.a_r,
            rp.b_r,
            rp.r,
            rp.K,
            rp.L,
        )
        hashed_sub = np.swapaxes(hashed_sub, 0, 1)
        hash_mat[idx] = hashed_sub

    existing_shm_hash_mat.close()
    return std_container, mean_container


def process_chunk_graph(
    time_series, ranges, window, rp, hash_names, L, dimension, n, K, mean, std
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

    # Open all the shared memory objects
    shm_hashes = [
        shared_memory.SharedMemory(name=hash_name.name) for hash_name in hash_names
    ]
    hash_arrs = [
        np.ndarray((n - window + 1, dimension, K), dtype=np.int8, buffer=shm.buf)
        for shm in shm_hashes
    ]
    mean_existing_shm = shared_memory.SharedMemory(name=mean.name)
    mean_container = np.ndarray((n - window + 1, dimension), dtype=np.float32, buffer=mean_existing_shm.buf)
    std_existing_shm = shared_memory.SharedMemory(name=std.name)
    std_container = np.ndarray((n - window + 1, dimension), dtype=np.float32, buffer=std_existing_shm.buf)


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
        hashed_sub = np.apply_along_axis(
            compute_hash,
            1,
            subsequence_n,
            rp.a_l,
            rp.b_l,
            rp.a_r,
            rp.b_r,
            rp.r,
            rp.K,
            rp.L,
        )
        hashed_sub = np.swapaxes(hashed_sub, 0, 1)

        for rep in range(L):
            hash_arrs[rep][idx] = hashed_sub[rep]
    # Close all the shared memory objects
    for shm in shm_hashes:
        shm.close()
    mean_existing_shm.close()
    std_existing_shm.close()
    #return True
