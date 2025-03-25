import numpy as np
from numba import jit
from math import fabs, exp

NPY_SQRT1_2 = 1.0 / np.sqrt(2)


@jit(nopython=True, cache=True, fastmath=True)
def ndtr(x):
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    g = 0.2316419

    k = 1.0 / (1.0 + g * fabs(x))
    k2 = k * k
    k3 = k2 * k
    k4 = k3 * k
    k5 = k4 * k

    if x >= 0.0:
        c = a1 * k + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5
        phi = 1.0 - c * exp(-x * x / 2.0) * NPY_SQRT1_2
    else:
        phi = 1.0 - ndtr(-x)

    return phi
    """
    if np.isnan(a):
        return np.nan

    x = a * NPY_SQRT1_2
    z = fabs(x)

    if z < NPY_SQRT1_2:
        y = 0.5 + 0.5 * erf(x)

    else:
        y = 0.5 * erfc(z)

        if x > 0:
            y = 1.0 - y

    return y
  """


@jit(nopython=True, fastmath=True, cache=True)
def p(d, r):
    first_term = 1 - 2 * ndtr(-r / d)
    result = first_term + (
        -2 / (np.sqrt(2 * np.pi) * r / d) * (1 - np.exp(-(r**2) / (2 * d**2)))
    )
    return result


@jit(nopython=True, fastmath=True)
def probability(d, i, j, b, s, jacc, K, L, dim):
    if i == K:
        return (np.power(1 - np.power(d, (K * dim)), j)) * (
            np.power(1 - np.power(jacc, s), b)
        )
    else:
        return (
            (np.power(1 - np.power(d, ((K - i) * dim)), j))
            * (np.power(1 - np.power(d, ((K - i + 1) * dim)), L - j))
            * np.power((1 - np.power(np.power(jacc, s), b)), 2)
        )


def stop(collision, jacc, b, s, i, j, threshold, K, L, r, dim):
    """
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

    """
    # jacc is the vector with bool that indicate where teè dimensions match
    # jacc = sum(jacc)/len(jacc)
    # d is p(d) for the euclidean LSH
    d = p(abs(collision[1][3]), r)

    # Check the condition
    return probability(d, i, j, b, s, jacc, K, L, dim) <= threshold


def probability3(d, i, j, K, L, dim):
    i = K - i
    if i == K:
        print(np.power(1 - np.power(d, (K * dim)), j))
        return np.power(1 - np.power(d, ((K) * dim)), j)
    else:
        print(np.power(1 - np.power(d, ((i) * dim)), j)) * (
            np.power(1 - np.power(d, ((i + 1) * dim)), L - j)
        )
        return (np.power(1 - np.power(d, ((i) * dim)), j)) * (
            np.power(1 - np.power(d, ((i + 1) * dim)), L - j)
        )


def stop3(collision, i, j, threshold, K, L, r, dim):
    """
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

    """
    # jacc is the vector with bool that indicate where teè dimensions match
    # jacc = sum(jacc)/len(jacc)
    # d is p(d) for the euclidean LSH
    d = p(abs(collision[0]), r)

    # Check the condition
    return probability3(d, i, j, K, L, dim) <= threshold


@jit(nopython=True, fastmath=True)
def probabilitygraph(d, i, j, K, L, dim):
    i = K - i
    if i == K:
        return np.power(1 - np.power(d, K), j)
    else:
        return (np.power(1 - np.power(d, (i)), j)) * (
            np.power(1 - np.power(d, (i + 1)), (L - j))
        )


@jit(nopython=True, fastmath=True, cache=True)
def stopgraph(collision, i, j, threshold, K, L, r, dim):
    """
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

    """
    # jacc is the vector with bool that indicate where teè dimensions match
    # jacc = sum(jacc)/len(jacc)
    # d is p(d) for the euclidean LSH
    ds = []
    for id in collision:
        ds.append(p(id, r))

    prob = 1
    for d in ds:
        prob *= probabilitygraph(d, i, j, K, L, dim)
    # Check the condition
    return prob <= threshold


if __name__ == "__main__":
    print(
        ((1 - p(100, 8) ** (8)) ** (200))
        * ((1 - p(1, 8) ** (8)) ** (200))
        * ((1 - p(1, 8) ** (8)) ** (200))
    )
    print(
        ((1 - p(34, 8) ** (8)) ** (200))
        * ((1 - p(34, 8) ** (8)) ** (200))
        * ((1 - p(34, 8) ** (8)) ** (200))
    )
    print(((p(34, 8)) ** (8 * 2)) * (p(34, 32) ** 8))
    print(p(100, 32) ** (8) * p(1, 8) ** (8) * p(1, 8) ** (8))
