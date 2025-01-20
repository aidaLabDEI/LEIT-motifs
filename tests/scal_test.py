import sys
import time

sys.path.append("source")
import stumpy
from LEIT_motifs import LEITmotifs
import numpy as np

def generate_random_walk(d, n):
    """
    Generates a multidimensional timeseries of dimensions d and length n of random walks.

    Args:
        d: The number of dimensions.
        n: The length of the timeseries.

    Returns:
        A numpy array of shape (n, d) representing the random walk timeseries.
    """
    # Initialize the timeseries with zeros
    timeseries = np.zeros((n, d))

    # Simulate the random walk for each dimension
    for i in range(d):
        # Generate random steps between -1 and 1
        steps = np.random.choice([-1, 1], size=n)
        weight = np.random.rand(n) * 5
        steps = steps * weight

        # Perform the random walk accumulation
        timeseries[:, i] = np.cumsum(steps)

    return timeseries



if __name__ == "__main__":
    lengths = [10**i for i in range(4, 7)]
    window = 200
    dimensions = 5
    motif_dimensions = 2
    
    # LEIT-motifs or Stumpy
    engine = 0
    # Easy or hard
    f = 1
    random_indices = np.random.randint(0, lengths[0]- window, 2)
    # Check they don't overlap
    while random_indices[1] - random_indices[0] < window:
        random_indices = np.random.randint(0, lengths[0]- window, 2)

    for n in lengths:
        d = generate_random_walk(dimensions, n)
        # Plant a sinusoidal motif in the timeseries
        for index in random_indices:
            for dim in range(motif_dimensions):
                d[index:index + window, dim] += np.sin(np.linspace(0, 2 * np.pi, window))
            if f == 1:
                d[index:index + window, :motif_dimensions] += np.random.normal(0, 0.01, (window, motif_dimensions))
        
        if engine == 0:
            _,_, time_tot = LEITmotifs(d, window, 1, (motif_dimensions, motif_dimensions))
            print("Time taken for LEIT-motifs with n = ", n, " is ", time_tot)
        else:
            d = d.T
            time_tot = time.perf_counter()
            mp, _ = stumpy.mstump(d, m=window)
            time_tot = time.perf_counter() - time_tot
            print("Time taken for Stumpy with n = ", n, " is ", time_tot)
            

