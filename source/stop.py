from scipy.stats import norm
import numpy as np
from numba import jit


#@jit(nopython=True, fastmath=True)
def second_term(first, r, d):
    second_term = -2 / (np.sqrt(2 * np.pi) * r / d) * (1 - np.exp(-r**2 / (2 * d**2)))
    return first + second_term

def p(d, r):
    first_term = 1 - 2 * norm.cdf(-r / d)
    result = second_term(first_term, r, d)
    return result

@jit(nopython=True, fastmath=True)
def probability(d, i, j, b, s, jacc, K, L, dim):
  if i == K:
    return (np.power(1-np.power(d,(K*dim)),j))*(np.power(1-np.power(jacc,s),b))
  else:
    return (np.power(1-np.power(d,((K-i)*dim)),j))*(np.power(1-np.power(d,((K-i+1)*dim)),L-j))* np.power((1-np.power(np.power(jacc,s),b)),2)


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
  return probability(d, i, j, b, s, jacc, K, L, dim) <= threshold


def probability3(d, i, j, K, L, dim):
  i = K - i
  if i == K:
    print(np.power(1-np.power(d,(K*dim)),j))
    return (np.power(1-np.power(d,((K)*dim)),j))
  else:
    print(np.power(1-np.power(d,((i)*dim)),j))*(np.power(1-np.power(d,((i+1)*dim)),L-j))
    return (np.power(1-np.power(d,((i)*dim)),j))*(np.power(1-np.power(d,((i+1)*dim)),L-j))

def stop3(collision, i, j, threshold, K, L, r, dim):
  '''
    Returns true if the probability of having missed a pair at distance d(collision)
    is less or equal than the threshold

    Parameters
    -----
     collision: list
      the element with max priority in the motif queue;
     jacc: float
      the jaccard similarity between the dimensions;
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
  d = p(abs(collision[0]), r)

  # Check the condition
  return probability3(d, i, j, K, L, dim) <= threshold
  
def probabilitygraph(d, i, j, K, L, dim):
  i = K - i
  if i == K:
    return (np.power(1-np.power(d,K),j))
  else:
    return (np.power(1-np.power(d,((i))),j))*(np.power(1-np.power(d,((i+1))),(L-j)))

def stopgraph(collision, i, j, threshold, K, L, r, dim):
  '''
    Returns true if the probability of having missed a pair at distance d(collision)
    is less or equal than the threshold

    Parameters
    -----
     collision: list
      the element with max priority in the motif queue;
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
  ds = []
  for elem in collision[1][3]:
    ds.append(p(elem,r))
  
  prob = 1
  for d in ds:
    prob *= probabilitygraph(d, i, j, K, L, dim)

  # Check the condition
  return prob <= threshold

if __name__ == "__main__":
  print(((1-p(100,8)**(8))**(200))*((1-p(1,8)**(8))**(200))*((1-p(1,8)**(8))**(200)))
  print(((1-p(34,8)**(8))**(200))*((1-p(34,8)**(8))**(200))*((1-p(34,8)**(8))**(200)))
  print(((p(34,8))**(8*2))*(p(34,32)**8))
  print(p(100,32)**(8)*p(1,8)**(8)*p(1,8)**(8))

