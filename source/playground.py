import numpy as np
from numba import njit, prange
import numba as nb
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib


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
    matplotlib.use("WebAgg")
    x = np.array([1000, 5000, 10000, 50000])
    y = np.array([10.42, 40.18, 165.94, 4064.05])

    svm = SVR(kernel="poly", degree=3)
    svm.fit(x.reshape(-1, 1), y)
    print(svm.predict(np.array([500000]).reshape(-1, 1)))

    # Plot the function found by the SVM

    x_plot = np.linspace(
        500, 500000, 1000
    )  # Generate 1000 points between 500 and 500000
    y_plot = svm.predict(x_plot.reshape(-1, 1))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="red", label="Original Data", zorder=5)
    plt.plot(x_plot, y_plot, color="blue", label="Learned Function", linewidth=2)
    plt.title("SVM Regression with Polynomial Kernel")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
