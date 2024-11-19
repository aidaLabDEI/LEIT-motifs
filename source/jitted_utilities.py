import numpy as np
import numba as nb
from numba import njit


# Faster than numpy
@njit(nb.bool(nb.int8[:], nb.int8[:]), fastmath=True)
def eq(a, b):
    return np.all(a == b)

def rolling_window(a, window):
    """
    Use strides to generate rolling/sliding windows for a numpy array.

    Parameters
    ----------
    a : numpy.ndarray
        numpy array

    window : int
        Size of the rolling window

    Returns
    -------
    output : numpy.ndarray
        This will be a new view of the original input array.
    """

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# Not faster

@njit(nb.int8(nb.int8[:, :], nb.int8[:, :]), fastmath=True, cache=True)
def multi_eq(a, b):
    sum = 0
    for i in range(a.shape[0]):
        if eq(a[i], b[i]):
            sum += 1
    return sum

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
