from scipy.stats import norm
import numpy as np

def p(d, r):
    first_term = 1 - 2 * norm.cdf(-r / d)
    second_term = -2 / (np.sqrt(2 * np.pi) * r / d) * (1 - np.exp(-r**2 / (2 * d**2)))
    return first_term + second_term

def stop(collision, jacc, b, s, i, j, threshold, K, L, r, dim):
  '''
    Returns true if the probability of having missed a pair at distance d(collision)
    is less or equal than the threshold

    Parameters
    -----
     collision: list
      the element with max priority in the motif queue;
     jacc: list
      a vector that indicates which dimensions have a matching hash;
     b: int
       number of bands for the MinHash;
     s: int
       number of rows for the minhash;
     i: int
       number of actual concatenations for RP;
     j: int
       number of actual repetitions for RP;
     threshold: float
       failure probability

    Returns
    -----
    true, if the condition is verified

  '''
  # jacc is the vector with bool that indicate where teè dimensions match
  #jacc = sum(jacc)/len(jacc)
  # d is p(d) for the euclidean LSH
  d = p(abs(collision[1][3]), r)

  # Check the condition

  if i == K:
    return (np.power(1-np.power(d,(i*dim)),j))*(np.power(1-np.power(jacc,s),b)) <= threshold
  else:
    return (np.power(1-np.power(d,(i*dim)),j))*(np.power(1-np.power(d,(i+1*dim)),L-j))* np.power((1-np.power(np.power(jacc,s),b)),2) <= threshold
