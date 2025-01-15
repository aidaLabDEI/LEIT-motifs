from typing import Tuple
import numpy as np

from RP_GRAPH import pmotif_findg
from RP_GRAPH_MULTI import pmotif_findg_multi
from base import create_shared_array
from find_bin_width import find_width_discr


def LEITmotifs(
    time_series: np.ndarray,
    window: int,
    k: int,
    motif_dimensionality_range: tuple[int, int],
    L: int = 200,
    K: int = 8,
    failure_probability: float = 0.01,
    r: int = 0,
) -> Tuple[list, int]:
    """
    Find the motifs in a time series using the LEIT-motifs algorithm

    Parameters
    ----------
    time_series : np.ndarray
        The time series to be analyzed
    window : int
        The window size
    k : int
        The number of motifs to find
    motif_dimensionality_range : tuple
        The range of the motif dimensionality, if the values are the same, the algorithm will use the base LEIT-motifs algorithm
    L : int, optional
        The number of LSH repetitions, by default 200
    K : int, optional
        The number of LSH concatenations, by default 8
    failure_probability : float, optional
        The failure probability, by default 0.01
    r : int, optional
        The LSH r value, by default it is estimated automatically using the developed heuristic

    Returns
    -------
    Tuple[list, int]
        The list of motifs containing elements of the type:
            ``[motif distance, [#id, [motif indices], [spanning dimensions], [dimensional distances]]] ``
        and the number of distances computed
    """

    # Ensure data is in float32 format
    time_series = np.ascontiguousarray(time_series.astype(np.float32))

    # Extract length and dimensionality
    length, dimensionality = time_series.shape

    # Check if the time series is in the correct format
    if dimensionality > length:
        time_series = time_series.T
        length, dimensionality = time_series.shape
    
    # Check the time series is multidimensional
    if dimensionality == 1:
        raise ValueError("The time series must be multidimensional, use ATTIMO for unidimensional time series")
    
    # Check that the request is multidimensional
    if (
        motif_dimensionality_range[0] == 1
    ):
        raise ValueError("The motifs to search must be multidimensional, use ATTIMO for unidimensional motifs")

    # Check the window size is smaller than the length of the time series and the motif range is valid
    if (
        window > length
        or motif_dimensionality_range[0] > motif_dimensionality_range[1]
        or motif_dimensionality_range[1] > dimensionality
        or motif_dimensionality_range[0] < 1
    ):
        raise ValueError("Invalid window size or motif dimensionality range, remember windows_size < length and 1 < motif_dimensionality_range[0] <= motif_dimensionality_range[1] <= dimensionality")

    # If the time series has NaN values, replace them with the mean
    if np.isnan(time_series).any():
        time_series = np.nan_to_num(time_series, nan=np.nanmean(time_series))
        raise Warning("The time series contains NaN values, they have been replaced with the mean, consider modeling the data to avoid this if the motif are not meaningful")

    # Add some noise to remove step-like patterns
    time_series += np.random.normal(0, 0.001, time_series.shape)

    # Estimate the r value
    if r == 0:
        r = find_width_discr(time_series, window, K)

    # Create a shared memory array to store the time series
    shm_ts, ts = create_shared_array((length, dimensionality), np.float32)
    ts[:] = time_series[:]

    # Find the motifs calling the correct algorithm based on range
    if motif_dimensionality_range[0] == motif_dimensionality_range[1]:
        motifs, num_dist, hash_t = pmotif_findg(
            shm_ts.name,
            length,
            dimensionality,
            window,
            k,
            motif_dimensionality_range[0],
            r,
            0,
            L,
            K,
            failure_probability,
        )
    else:
        motifs, num_dist, hash_t = pmotif_findg_multi(
            shm_ts.name,
            length,
            dimensionality,
            window,
            k,
            motif_dimensionality_range,
            r,
            0,
            L,
            K,
            failure_probability,
        )

    return motifs, num_dist

if __name__ == "__main__":
    import pandas as pd
    import os
    current_dir = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(current_dir, "..", "Datasets", "FOETAL_ECG.dat"), sep=r"\s+")
    data = data.drop(data.columns[[0]], axis=1)
    data = np.ascontiguousarray(data.to_numpy())
    try:
        motifs, num_dist = LEITmotifs(data, 50, 1, (8, 8))
        print(motifs)
        motifs, num_dist = LEITmotifs(data, 50, 1, (2, 8))
        print(motifs)
    except Exception as e:
        print(e)
    print("All tests passed")

