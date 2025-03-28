from multiprocessing import cpu_count, shared_memory, Pool
import numpy as np
from pyts.approximation import SymbolicAggregateApproximation

def SAX(chunk, transformers):
    pass

def RP(time_series_name, n, dimensionality, window, motif_dimensionality, word_length, max_iter, threshold):
    # Open the time series shared array
    ts_data = shared_memory.SharedMemory(name=time_series_name)
    ts = np.array(np.ndarray((n, dimensionality), dtype=np.float32, buffer=ts_data.buf))
    # Create a SAX transformer for each dimension
    transformers = []
    for _ in range(dimensionality):
        transformer = SymbolicAggregateApproximation(
            n_segments=window,
            alphabet_size_avg=word_length,
            alphabet_size_slope=word_length,
            strategy="quantile",
            verbose=False,
        )
        transformers.append(transformer)
    # Divide the time series into chunks and apply SAX
    chunk_sz = max(1000, n // cpu_count() * 2)
    num_chunks = max(1, n // chunk_sz)
    chunks = [(time_series_name, ranges, transformers) for ranges in np.array_split(ts, num_chunks)]
    
    with Pool(cpu_count()) as pool:
        sax_results = pool.map(SAX, chunks)
        sax_results = np.concatenate(sax_results, axis=0)
        pass
        
    return