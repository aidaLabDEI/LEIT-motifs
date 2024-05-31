import numpy as np
import numpy.typing as npt
from numpy.fft import fft
from numba import njit

def z_normalize(ts: npt.ArrayLike) -> npt.ArrayLike:
    mean = np.mean(ts, axis=0)
    std = np.std(ts, axis=0)
    std[std == 0] = 1  # Avoid division by zero if standard deviation is zero
    return (ts - mean) / std

@njit
def compute_distances(ts_fft_real, ts_fft_imag, rand_indices):
    num_rand_indices = rand_indices.shape[0]
    num_subsequences = ts_fft_real.shape[0]
    num_dimensions = ts_fft_real.shape[2]
    distances = np.empty((num_rand_indices, num_subsequences))

    for i in range(num_rand_indices):
        idx = rand_indices[i]
        for j in range(num_subsequences):
            if j == idx:
                distances[i, j] = 0
            else:
                dist = 0
                for k in range(num_dimensions):
                    real_diff = ts_fft_real[idx, :, k] - ts_fft_real[j, :, k]
                    imag_diff = ts_fft_imag[idx, :, k] - ts_fft_imag[j, :, k]
                    dist += np.linalg.norm(real_diff + 1j * imag_diff) ** 2
                distances[i, j] = np.sqrt(dist)

    return distances

def find_width_discr(ts: npt.ArrayLike, window: int, K: int) -> int:
    # Z-normalize the entire time series for each dimension
    ts = z_normalize(ts)
    
    # Pick 40 random subsequences
    random_gen = np.random.default_rng()
    n = ts.shape[0]
    d = ts.shape[1]
    rand_indices = random_gen.choice(n - window, 40, replace=False)

    # Compute the FFT for all subsequences in the time series
    ts_fft_real = np.empty((n - window, window, d))
    ts_fft_imag = np.empty((n - window, window, d))
    for j in range(n - window):
        for k in range(d):
            ts_fft = fft(ts[j:j + window, k])
            ts_fft_real[j, :, k] = ts_fft.real
            ts_fft_imag[j, :, k] = ts_fft.imag

    # Compute distances
    distances = compute_distances(ts_fft_real, ts_fft_imag, rand_indices)
    
    # Flatten the distances array and filter out infinity values
    distances = distances[distances != np.inf]
    
    # Find r such that the max value of the distribution is < r * 2^K
    r = np.max(distances) / (2 ** K)
    return int(np.ceil(r))
