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

    # Check the window size is smaller than the length of the time series and the motif range is valid
    if (
        window > length
        or motif_dimensionality_range[0] > motif_dimensionality_range[1]
        or motif_dimensionality_range[1] > dimensionality
        or motif_dimensionality_range[0] < 1
    ):
        raise ValueError("Invalid window size or motif dimensionality range")

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
