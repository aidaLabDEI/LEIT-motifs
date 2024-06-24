from base import *
from multiprocessing import Pool
import stumpy

def relative_contrast(ts, pair, window, dimensionality):

    pair_indices = pair[1][1]
    matching_dims = pair[1][2]


    dimensions = ts.shape[1]
    num_subsequences = ts.shape[0] - window + 1
    distances_matrix = np.zeros((num_subsequences, dimensions))
    dist_pair = np.zeros((num_subsequences, dimensions))
    sum_distances = 0.0

    # Compute distances for each dimension
    for i in range(num_subsequences):
      distances = np.zeros((num_subsequences, dimensions))
      for dim in range(dimensions):
        series = ts[:, dim]

        subseq = series[i:i+window]
        distances[:,dim] = stumpy.mass(subseq ,ts[:, dim])
      if i == pair_indices[0]:
        dist_pair = distances.copy()

      # Order the columns of the distance matrix
      distances = np.sort(distances, axis=1)

      # Remove the trivial matches, so the indices from i-window to i+window are removed on the first axis
      start = max(0, i - window)
      end = min(num_subsequences, i + window + 1)
      distances[start:end, :] = np.inf
      # Find the column that allows the smallest sum using dimensionality equal to the number of dimensions
      sum_d = np.sum(distances[:,:dimensionality], axis=1)

      sum_distances += np.min(sum_d)

    dist_pair = dist_pair[pair_indices[1],:]

    pair_distance = np.sum(dist_pair[matching_dims])

    # Calculate the average pairwise distance
    average_distance = sum_distances / num_subsequences

    return average_distance / abs(pair_distance)

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