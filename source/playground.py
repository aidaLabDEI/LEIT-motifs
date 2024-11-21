import numpy as np
from numba import njit, prange
import numba as nb
from base import create_shared_array


@njit(nb.bool(nb.int8[:], nb.int8[:]), fastmath=True, cache=True)
def eq(a, b):
    return (a == b).all()


@njit(nb.int8(nb.int8[:, :], nb.int8[:, :]), fastmath=True, cache=True, parallel=True)
def multi_eq(a, b):
    sum = 0
    for i in prange(a.shape[0]):
        if np.all(a[i] == b[i]):
            sum += 1
    return sum


@njit(nb.float32[:](nb.float32[:, :]), fastmath=True, cache=True)
def comp_mean(a):
    res = np.empty(a.shape[1], dtype=np.float32)
    for i in range(a.shape[1]):
        res[i] = np.mean(a[i])
    return res


@njit(nb.float32[:](nb.float32[:, :]), fastmath=True, cache=True)
def comp_std(a):
    res = np.empty(a.shape[1], dtype=np.float32)
    for i in range(a.shape[0]):
        res[i] = np.std(a[i])
    return res


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

    shape = a.shape[:-1] + (a.shape[0] - window + 1, window)
    strides = a.strides + (a.strides[0],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


@njit
def sum(a_name):
    return np.sum(a_name)


if __name__ == "__main__":
    a_name, a = create_shared_array((5, 5), np.float32)
    a[:] = np.random.rand(5, 5)
    print(sum(a))
