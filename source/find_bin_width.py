import numpy as np
import numpy.typing as npt
from numpy.fft import fft, ifft
from numba import njit

def z_normalize(ts: npt.ArrayLike) -> npt.ArrayLike:
    mean = np.mean(ts, axis=0)
    std = np.std(ts, axis=0)
    std[std == 0] = 1  # Avoid division by zero if standard deviation is zero
    return (ts - mean) / std


def compute_dot_products_fft(ts_fft, sub_fft_conj, num_subsequences):
    dot_products = np.zeros(num_subsequences)

    for j in range(num_subsequences):
        product_fft = ts_fft[j] * sub_fft_conj
        dot_product = np.sum(ifft(product_fft).real)
        dot_products[j] = dot_product

    return dot_products

def find_width_discr(ts: npt.ArrayLike, window: int, K: int) -> int:
    # Z-normalize the entire time series for each dimension
    ts = z_normalize(ts)
    
    # Pick 40 random subsequences
    random_gen = np.random.default_rng()
    n = ts.shape[0]
    d = ts.shape[1]
    rand_indices = random_gen.choice(n - window, 40, replace=False)

    num_subsequences = n - window + 1

    # Compute the FFT for all subsequences in the time series
    ts_fft = np.empty((num_subsequences, window, d), dtype=np.complex_)
    for j in range(num_subsequences):
        for k in range(d):
            ts_fft[j, :, k] = fft(ts[j:j + window, k])

    all_dot_products = []
    for idx in rand_indices:
        dot_products = np.zeros((num_subsequences, d))
        for k in range(d):
            sub_fft = ts_fft[idx, :, k]
            sub_fft_conj = np.conj(sub_fft)
            dot_products[:, k] = compute_dot_products_fft(ts_fft[:, :, k], sub_fft_conj, num_subsequences)
        all_dot_products.append(np.sum(dot_products, axis=1))

    all_dot_products = np.array(all_dot_products).flatten()

    # Find r such that the value of a certain percentile of the distribution is < r * 2^K
    percentile_value = np.percentile(all_dot_products, 6)
    r = abs(percentile_value / (2 ** K))
    return int(np.ceil(r))