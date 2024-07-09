from base import *
from multiprocessing import Pool
import stumpy

def relative_contrast(ts, window, dimensionality):
  # Compute the multidimensional matrix profile
  matrix_profile, mp_idx = stumpy.mstump(ts.T, m=window)

  total_sum = np.sum(matrix_profile[:,dimensionality])
  num = matrix_profile.shape[0]

  d = np.min(matrix_profile[:,dimensionality])

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