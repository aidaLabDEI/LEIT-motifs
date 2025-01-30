import sys
import time

sys.path.append("source")
import stumpy
from LEITmotifs import LEITmotifs
import numpy as np
import pandas as pd

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
    path = "scalability.csv"
    dataframe = pd.DataFrame({"Algo": int(),
                              "Size": int(),
                              "Time (s)": float()}, index = [])
    # LEIT-motifs or Stumpy
    engines = [1]
    # Easy, medium or hard
    f = [1,2,3]
    random_indices = np.random.randint(0, lengths[0]- window, 2)
    # Check they don't overlap
    while random_indices[1] - random_indices[0] < window:
        random_indices = np.random.randint(0, lengths[0]- window, 2)

    for engine in engines:

        for n in lengths:
            d = np.random.normal(0,0.01, (n,dimensions))#generate_random_walk(dimensions, n)
            print(d.shape)
            # Plant a sinusoidal motif in the timeseries
            for index in random_indices:
                for dim in range(motif_dimensions):
                    d[index:index + window, dim] += np.sin(np.linspace(0, 2 * np.pi, window))
            
            if engine == 0:
                for difficulty in f:
                    # Add noise to the timeseries based on the difficulty that it should have
                    for index in random_indices:
                        if difficulty == 1:
                            d[index:index + window, :motif_dimensions] += np.random.normal(0, 0.01, (window, motif_dimensions))
                        elif difficulty == 2:
                            d[index:index + window, :motif_dimensions] += np.random.normal(0, 0.1, (window, motif_dimensions))

                    _,_, time_tot = LEITmotifs(d, window, 1, (motif_dimensions, motif_dimensions))
                    dataframe = dataframe._append({"Algo": int(difficulty), "Size": int(n), "Time (s)": time_tot}, ignore_index = True)
                    print("Time taken for LEIT-motifs with n = ", n, " is ", time_tot)
            else:
                d = d.T
                time_tot = time.perf_counter()
                mp, _ = stumpy.mstump(d, m=window)
                time_tot = time.perf_counter() - time_tot
                dataframe = dataframe._append({"Algo": 0, "Size": int(n), "Time (s)": time_tot}, ignore_index = True)
                print("Time taken for Stumpy with n = ", n, " is ", time_tot)
            print(dataframe)
            dataframe.to_csv(path, index = False)

