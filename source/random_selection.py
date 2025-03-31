from multiprocessing import cpu_count, shared_memory, Pool
import random
import numpy as np
from pyts.approximation import SymbolicAggregateApproximation, PiecewiseAggregateApproximation
from itertools import combinations
from base import z_normalized_euclidean_distanceg
from numba import njit, prange

@njit(fastmath=True, parallel=True, cache=True)
def fill_matrix(collision_matrix, dimensions_sets, sax_results, n, window):
    for set_idx in prange(len(dimensions_sets)): # Loop over sets
        set_dims = dimensions_sets[set_idx] 
        
        words = sax_results[:, set_dims[0]].copy() # Get first dim
        for i in range(1, len(set_dims)):
             words += sax_results[:, set_dims[i]] 
             
        sorting = np.argsort(words)
        for idx1_enum in range(len(sorting) - 1):
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
    Expects chunk to be a 2D array of shape (n_samples, dimensionality).
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
    # Take all non-zero elements in the collision matrix
    max_elements = np.argwhere(collision_matrix > 0)
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