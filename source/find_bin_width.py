import numpy as np
import numpy.typing as npt
from numpy.fft import fft, ifft


def find_width_discr(ts: npt.ArrayLike, window: int, K: int) -> int:
    # Pick 40 random subsequences
    random_gen = np.random.default_rng()
    n = ts.shape[0]
    d = ts.shape[1]
    num_rand_vec = 40
    a = np.random.randn(window,num_rand_vec, d)
    b = np.random.uniform(0, 1, (num_rand_vec, d))
    rand_indices = random_gen.choice(n - window, min(n, 2000), replace=False)

    num_subsequences = n - window + 1

    all_dot_products = np.zeros((min(n, 2000), num_rand_vec, d), dtype=np.int64)
    # Compute the FFT for all subsequences in the time series
    for j, idx in enumerate(rand_indices):
        subsequence = ts[idx:idx + window]
        subsequence = subsequence - np.mean(subsequence, axis=0)/np.std(subsequence, axis=0)
        for i in range(num_rand_vec):
            for k in range(d):
                all_dot_products[j, i, k] = np.floor(np.dot(subsequence[:, k], a[:,i, k]) + b[i, k])
  
    # Find r such that the value of a certain percentile of the distribution is < r * 2^K
    percentile_value =np.percentile(all_dot_products.flatten(), 95) - np.percentile(all_dot_products.flatten(), 5)
    r = abs(percentile_value / ((2 ** 8)))
    return int(np.ceil(r))