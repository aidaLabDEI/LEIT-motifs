from base import *

def relative_contrast(ts, pair, window):
  dimensions = ts.shape[1]
  d, _, _ = z_normalized_euclidean_distance(ts[pair[0]:pair[0]+window],ts[pair[1]:pair[1]+window], np.arange(dimensions),
                                      np.mean(ts[pair[0]:pair[0]+window], axis=0).T, np.std(ts[pair[0]:pair[0]+window], axis=0).T,
                                      np.mean(ts[pair[1]:pair[1]+window], axis=0).T, np.std(ts[pair[1]:pair[1]+window], axis=0).T)

  num = 0
  sum = 0
  for i in range(ts.shape[0]-window+1):
    for j in range(ts.shape[0]-window+1):
      if abs(i-j) > window:
        num += 1
        mean_i = np.mean(ts[i:i+window], axis=0).T
        std_i = np.std(ts[i:i+window], axis=0).T
        mean_j = np.mean(ts[j:j+window], axis=0).T
        std_j = np.std(ts[j:j+window], axis=0).T
        d_ij, _, _ = z_normalized_euclidean_distance(ts[i:i+window], ts[j:j+window], np.arange(dimensions), mean_i, std_i, mean_j, std_j)
        sum += d_ij

  d_hat = sum/num

  return d_hat/d

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