import numpy as np
import numba as nb
from numba import njit


# Faster than numpy
@njit(nb.bool(nb.int8[:], nb.int8[:]), fastmath=True)
def eq(a, b):
    return np.all(a == b)


@njit(nb.int8(nb.int8[:, :], nb.int8[:, :]), fastmath=True, cache=True)
def multi_eq(a, b):
    sum = 0
    for i in range(a.shape[0]):
        if eq(a[i], b[i]):
            sum += 1
    return sum


# Not faster
@njit(nb.float32[:](nb.float32[:, :]), fastmath=True, cache=True)
def comp_mean(a):
    res = np.empty(a.shape[1], dtype=np.float32)
    for i in range(a.shape[1]):
        res[i] = np.sum(a[i]) / a.shape[0]
    return res


@njit(nb.float32[:](nb.float32[:, :]), fastmath=True, cache=True)
def comp_std(a):
    res = np.empty(a.shape[1], dtype=np.float32)
    for i in range(a.shape[1]):
        res[i] = np.std(a[:, i])
    return res
