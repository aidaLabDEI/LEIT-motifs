from multiprocessing import cpu_count, shared_memory, Pool
import random
import numpy as np
from numpy.dtypes import StringDType
from pyts.approximation import SymbolicAggregateApproximation, PiecewiseAggregateApproximation
from itertools import combinations
from scipy.sparse import dok_array
from base import z_normalized_euclidean_distanceg

def SAX(chunk, transformers, window, c):
    """
    Applies each SAX transformer (one per dimension) on the given chunk.
    Expects chunk to be a 2D array of shape (n_samples, dimensionality).
    """
    n_samples = chunk.shape[0]
    n_dims = len(transformers)
    # Preallocate output array (assumed numeric output)
    transformed_chunk = np.empty((n_samples-window+1, n_dims), dtype=StringDType())
    paa = PiecewiseAggregateApproximation(window_size=c)
    for i, transformer in enumerate(transformers):
        subsequences = [chunk[s:s+window,i] for s in range(n_samples-window+1)]
        subsequences = paa.transform(subsequences)
        saxed = transformer.fit_transform(subsequences)
        # Join into one string for each dimension of a subsequence
        for s, elem in enumerate(saxed):
            transformed_chunk[s, i] = str(''.join(elem))            
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
        )
        transformers.append(transformer)
    # Divide the time series into chunks and apply SAX
    chunk_sz = max(1000, n // cpu_count() * 2)
    num_chunks = max(1, n // chunk_sz)
    chunks = [(ranges, transformers, window, c_dimensionality) for ranges in np.array_split(ts, num_chunks)]
    
    with Pool(cpu_count()) as pool:
        sax_results = pool.starmap(SAX, [(chunk) for chunk in chunks])
        sax_results = np.concatenate(sax_results, axis=0)
        pass
    
    # Initialize the collision matrix and create the random projection sets 
    collision_matrix =  dok_array(((n-window+1, n-window+1)), dtype=np.int8) #np.zeros((n-window+1, n-window+1))
    dimensions_sets = list(combinations(np.arange(dimensionality), motif_dimensionality))
    random.shuffle(dimensions_sets)
    
    # Fill the collision matrix 
    for set in dimensions_sets:
        words = sax_results[:,set[0]]
        # Concatenate the surviving dimensions into one word
        for dimension in set[1:]:
            np.add(words, sax_results[:,dimension], words)
            
        # Order lexicographycally the vector, keep an array with the original indices
        sorting = np.argsort(words)
        # Scan the vector, if the words match then increase the corresponding matrix entry
        for index1, elem1 in enumerate(sorting):
            for elem2 in sorting[index1+1:]:
                if np.equal(words[elem1], words[elem2]):
                    if elem1 > elem2:
                        elem1, elem2 = elem2, elem1
                    collision_matrix[elem1,elem2] +=1
                else:
                    break
    # Zero out the diagonal and the window around it
    for i in range(n-window+1):
        collision_matrix[i,i-window//2:i+window//2] = 0
    # Find the maximal entries in the collision matrix and compute the distances for them
    top = []
    max_elements = sorted(collision_matrix.items(), key=lambda x: x[1], reverse=True)[:k_motifs*100]
    dist_comp= 0
    for maximal in max_elements:
        maximal_sub1, maximal_sub2 = maximal[0]
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
        if len(top)>k_motifs and curr_dist > top[-1][0]:
            top.pop()
            top.insert(0,[-curr_dist, [dist_comp, (maximal_sub1, maximal_sub2), dim]])
            sorted(top, reverse=True)
        elif len(top)>k_motifs:
            continue
        else:  
            top.insert(0,[-curr_dist, [dist_comp, (maximal_sub1, maximal_sub2), dim]])
    
    return top, dist_comp