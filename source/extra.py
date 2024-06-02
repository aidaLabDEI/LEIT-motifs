from base import *

def relative_contrast(ts, pair, window):
    dimensions = ts.shape[1]

    # Precompute means and standard deviations for all windows
    means = np.array([np.mean(ts[i:i+window], axis=0) for i in range(ts.shape[0] - window + 1)])
    stds = np.array([np.std(ts[i:i+window], axis=0) for i in range(ts.shape[0] - window + 1)])

    # Calculate the distance for the given pair
    d, _, _ = z_normalized_euclidean_distance(ts[pair[0]:pair[0]+window], ts[pair[1]:pair[1]+window], np.arange(dimensions),
                                              means[pair[0]].T, stds[pair[0]].T, means[pair[1]].T, stds[pair[1]].T)

    num = 0
    total_sum = 0.0

    for i in range(ts.shape[0] - window + 1):
        for j in range(ts.shape[0] - window + 1):
            if abs(i - j) > window:
                num += 1
                mean_i = means[i].T
                std_i = stds[i].T
                mean_j = means[j].T
                std_j = stds[j].T
                d_ij, _, _ = z_normalized_euclidean_distance(ts[i:i+window], ts[j:j+window], np.arange(dimensions), mean_i, std_i, mean_j, std_j)
                total_sum += d_ij

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