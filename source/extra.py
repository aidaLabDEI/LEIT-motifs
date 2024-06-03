from base import *
from multiprocessing import Pool

def calculate_distance(args):
    ts_chunk_i, ts_chunk_j, means_chunk_i, stds_chunk_i, means_chunk_j, stds_chunk_j, dimensions = args
    d_ij, _, _ = z_normalized_euclidean_distance(ts_chunk_i, ts_chunk_j, np.arange(dimensions), means_chunk_i, stds_chunk_i, means_chunk_j, stds_chunk_j)
    return d_ij

def relative_contrast(ts, pair, window):
    dimensions = ts.shape[1]
    num_chunks = int(np.sqrt(ts.shape[0] - window + 1))

    # Precompute means and standard deviations for all windows
    means = np.array([np.mean(ts[i:i+window], axis=0) for i in range(ts.shape[0] - window + 1)])
    stds = np.array([np.std(ts[i:i+window], axis=0) for i in range(ts.shape[0] - window + 1)])

    # Calculate the distance for the given pair
    d, _, _ = z_normalized_euclidean_distance(ts[pair[0]:pair[0]+window], ts[pair[1]:pair[1]+window], np.arange(dimensions),
                                              means[pair[0]].T, stds[pair[0]].T, means[pair[1]].T, stds[pair[1]].T)

    num = 0
    total_sum = 0.0

    chunk_size = (ts.shape[0] - window + 1) // num_chunks
    args_list = []

    for i in range(num_chunks):
        for j in range(num_chunks):
            start_i = i * chunk_size
            end_i = min(start_i + chunk_size, ts.shape[0] - window + 1)
            start_j = j * chunk_size
            end_j = min(start_j + chunk_size, ts.shape[0] - window + 1)
            ts_chunk_i = ts[start_i:start_i+window]
            ts_chunk_j = ts[start_j:start_j+window]
            means_chunk_i = means[start_i].T
            stds_chunk_i = stds[start_i].T
            means_chunk_j = means[start_j].T
            stds_chunk_j = stds[start_j].T
            args_list.append((ts_chunk_i, ts_chunk_j, means_chunk_i, stds_chunk_i, means_chunk_j, stds_chunk_j, dimensions))

    with Pool() as pool:
        result = pool.map(calculate_distance, args_list)

    num = len(result)
    total_sum = sum(result)

    d_hat = total_sum / num
    return d_hat / d

def find_all_occur(ts, motifs, window):
  motif_copy = motifs
  for motif in motif_copy:
    occurrences = motif[1][1]

    base = motif[1][1][0]
    base = ts[base:base+window,:].T
    dim = motif[1][2]

    mean_i = np.mean(base, axis=1)
    std_i = np.std(base, axis=1)


    for i in range(ts.shape[0]-window+1):
      ins = True
      for occurr in occurrences:
        if abs(i-occurr) > window:
          ins = ins and True
        else: ins = False
      if ins:
        other = ts[i:i+window,:].T
        print(base, other, dim, mean_i)
        dist, _, _ = z_normalized_euclidean_distance(base, other, dim[0], mean_i, std_i,
                                              np.mean(other, axis=1), np.std(other, axis=1) )
        # If the distance is small enough consider it as a new occurrence of the motif
        if dist < 2:
          occurrences.append(i)
  return motif_copy