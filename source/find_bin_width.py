import numpy as np
import numpy.typing as npt


def find_width_discr(ts: npt.ArrayLike, window: int, K: int) -> int:
    # Pick 40 random subsequences
    random_gen = np.random.default_rng()
    n = ts.shape[0]
    d = ts.shape[1]
    num_rand_vec = 50
    a = np.random.randn(window, num_rand_vec, d) * 10
    # b = np.random.randint(0, 10000, (num_rand_vec, d))
    rand_indices = random_gen.choice(n - window, min(n, 2000), replace=False)

    # num_subsequences = n - window + 1

    all_dot_products = np.zeros((min(n, 2000), num_rand_vec, d), dtype=np.float32)
    # Compute the FFT for all subsequences in the time series
    for j, idx in enumerate(rand_indices):
        subsequence = ts[idx : idx + window]
        subsequence = (subsequence - np.mean(subsequence, axis=0)) / np.std(
            subsequence, axis=0
        )
        for i in range(num_rand_vec):
            for k in range(d):
                all_dot_products[j, i, k] = np.dot(subsequence[:, k], a[:, i, k])

    all_dot_products = all_dot_products.sum(axis=2)
    # Find r such that the value of a certain percentile of the distribution is < r * 2^K
    percentile_value = np.percentile(all_dot_products, 95) - np.percentile(
        all_dot_products, 5
    )
    print(percentile_value)
    r = abs(percentile_value / (2**8))
    r = max(r, 4)
    r = min(r, 32)
    return int(np.ceil(r))
