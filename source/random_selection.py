from multiprocessing import cpu_count, shared_memory, Pool
import random
import numpy as np
from pyts.approximation import SymbolicAggregateApproximation, PiecewiseAggregateApproximation
from itertools import combinations
from base import z_normalized_euclidean_distanceg
from numba import njit, prange

@njit(fastmath=True, parallel=True, cache=True)
def fill_matrix(collision_matrix, dimensions_sets, sax_results, n, window):
    for set_idx in range(len(dimensions_sets)): # Loop over sets
        set_dims = dimensions_sets[set_idx] 
        
        words = sax_results[:, set_dims[0]].copy() # Get first dim
        for i in range(1, len(set_dims)):
             words += sax_results[:, set_dims[i]] 
             
        sorting = np.argsort(words)
        for idx1_enum in prange(len(sorting) - 1):
            elem1 = sorting[idx1_enum]
            for idx2_enum in range(idx1_enum + 1, len(sorting)):
                elem2 = sorting[idx2_enum]
                
                if words[elem1] == words[elem2]:
                    # Ensure lower index is first for upper triangle matrix
                    row, col = (elem1, elem2) if elem1 < elem2 else (elem2, elem1)
                    # Avoid trivial matches within window
                    if col - row >= window: # Check non-trivial match 
                         collision_matrix[row, col] += 1
                else:
                    # Since sorted, no further matches for elem1
                    break 
    return collision_matrix

def SAX(ts, chunk, transformers, window, c):
    """
    Applies each SAX transformer (one per dimension) on the given chunk.
    """
    n_samples = chunk.shape[0]
    n_dims = len(transformers)
    # Preallocate output array
    transformed_chunk = np.empty((n_samples-window+1, n_dims), dtype=np.int64())
    paa = PiecewiseAggregateApproximation(window_size=c)
    for i, transformer in enumerate(transformers):
        subsequences = [ts[s:s+window,i] for s in range(n_samples-window+1)]
        subsequences = paa.transform(subsequences)
        saxed = transformer.fit_transform(subsequences)
        # Join into one concateneted number
        for s, elem in enumerate(saxed):
            transformed_chunk[s, i] = int("".join(map(str, elem)))          
    return transformed_chunk

def RP(time_series_name, n, dimensionality, window, motif_dimensionality, k_motifs, dictionary_dim, c_dimensionality, max_iter, threshold):
    """
    The method is based on the work of :math:`Minnen, David, et al. "Detecting subdimensional motifs: An efficient algorithm for generalized multivariate pattern discovery." Seventh IEEE international conference on data mining (ICDM 2007). IEEE, 2007.`
    Iit works by creating SAX representations of the subsequences, selecting random subdimensions and filling a collision
    matrix when the SAX words are equal. 
    SAX is tuned with this heuristic: when the matrix is too dense the dictionary is augmented, when the matrix is too sparse the words are shortened.
    This heuristic may drastically increase the computation time, since the old collision matrix is trashed.
    We swapped the original method to find the subdimensional motifs that required computing a distance distribution for each dimension with
    the method used in :math:`LEIT-motifs`, this produces better results and removes the threshold input parameter that is difficult to tune manually on data.
    
    Parameters
    ----------
    time_series_name : str
        The name of the shared memory array containing the time series data.
    n : int
        The lenght if the time series.
    dimensionality : int
        The number of dimensions in the time series.
    window : int
        The size of the window for the motifs.
    motif_dimensionality : int
        The dimensionality of the motifs to find.
    k_motifs : int
        The number of motifs to find.
    dictionary_dim : int
        The number of symbols in the SAX dictionary.
    c_dimensionality : int
        The dimensionality of the SAX representation. If None, set to window.
    max_iter : int
        The maximum number of iterations for the Projection cycle.
    threshold : float
        The threshold for the distance to consider a match. Not used in this implementation.
        
    Returns
    -------
    top : list
        A list of the top k motifs found, each containing the distance, indices of the motifs, and dimensions.
    dist_comp : int
        The number of distance computations performed.
    """
    # If c_dimensionality is not specified, set it to window
    if c_dimensionality is None:
        c_dimensionality = window
    # Open the time series shared array
    ts_data = shared_memory.SharedMemory(name=time_series_name)
    ts = np.array(np.ndarray((n, dimensionality), dtype=np.float32, buffer=ts_data.buf))
    # Create a SAX transformer for each dimension
    transformers = []
    for f in range(dimensionality):
        transformer = SymbolicAggregateApproximation(
            n_bins = dictionary_dim,
            strategy="normal",
            alphabet="ordinal",
        )
        transformers.append(transformer)
    # Divide the time series into chunks and apply SAX
    chunk_sz = max(1000, n // cpu_count() * 2)
    num_chunks = max(1, n // chunk_sz)
    chunks = [(ts, ranges, transformers, window, c_dimensionality) for ranges in np.array_split(ts, num_chunks)]
    
    with Pool(cpu_count()) as pool:
        sax_results = pool.starmap(SAX, [(chunk) for chunk in chunks])
        sax_results = np.concatenate(sax_results, axis=0)
        sax_results = sax_results[:n-window+1, :] # Ensure correct size

    
    # Initialize the collision matrix and create the random projection sets 
    collision_matrix = np.zeros((n-window+1, n-window+1)) #dok_array(((n-window+1, n-window+1)), dtype=np.int16) #np.zeros((n-window+1, n-window+1))
    dimensions_sets = list(combinations(np.arange(dimensionality), motif_dimensionality))
    random.shuffle(dimensions_sets)
    
    # Fill the collision matrix 
    collision_matrix = fill_matrix(collision_matrix, dimensions_sets, sax_results, n, window)
    # Find the maximal entries in the collision matrix and compute the distances for them
    top = []
    dist_comp = 0
    max_elements = np.zeros((1, 2), dtype=np.int64)
    # Take all non-zero elements in the collision matrix
    exp_matrix_entries = motif_dimensionality
    while max_elements.shape[0] < 2 and exp_matrix_entries > 0:
        max_elements = np.argwhere(collision_matrix >= exp_matrix_entries)
        exp_matrix_entries -= 1
    
    for maximal in max_elements:
        maximal_sub1, maximal_sub2 = maximal
        dist_comp += 1
        curr_dist, dim, _ = (
        z_normalized_euclidean_distanceg(
            ts[maximal_sub1 : maximal_sub1 + window],
            ts[maximal_sub2 : maximal_sub2 + window],
            np.arange(dimensionality, dtype=np.int32),
            np.mean(ts[maximal_sub1 : maximal_sub1 + window], axis=-1),
            np.std(ts[maximal_sub1 : maximal_sub1 + window], axis=-1),
            np.mean(ts[maximal_sub2 : maximal_sub2 + window], axis=-1),
            np.std(ts[maximal_sub2 : maximal_sub2 + window], axis=-1),
            motif_dimensionality,
            )
        )
        
        # Insert only better elements
        if len(top)>=k_motifs and curr_dist < top[-1][0]:
            top.pop()
            top.insert(0,[curr_dist, [dist_comp, (maximal_sub1, maximal_sub2), dim]])
            sorted(top, reverse=False)
        elif len(top)>=k_motifs:
            continue
        else:  
            top.insert(0,[curr_dist, [dist_comp, (maximal_sub1, maximal_sub2), dim]])
    
    return top, dist_comp