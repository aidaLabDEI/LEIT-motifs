import numpy as np
from numba import njit, prange
import numba as nb
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
    r"""
    data = pd.read_csv("/home/monaco/bt_analysis/code/LEIT-motifs/Datasets/FOETAL_ECG.dat", sep=r"\s+")
    data = data.to_numpy()
    print(data.shape)
    window= 50
    distances = []
    figure,axs = plt.subplots(1,1)
    for dim in range(2):#data.shape[1]):
        for sub in range(0, data.shape[0]-window+1):
            dp = stumpy.mass(data[sub:sub+window, dim], data[:,dim])
            distances.append(dp)

    sns.kdeplot(np.array(distances).flatten(), ax=axs)
    plt.show()
    '''
    x = np.array([5000, 10000, 50000])
    y = np.array([11.07, 27.55, 31.39])

    svm = SVR(kernel="poly", degree=3)
    svm.fit(x.reshape(-1, 1), y)
    print(svm.predict(np.array([25132289]).reshape(-1, 1)))

    # Plot the function found by the SVM

    x_plot = np.linspace(
        5000, 25132289, 1000
    )  # Generate 1000 points between 5000 and 500000
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
    '''
    """
    a = np.pad(np.flip(np.array([1, 2, 3])), (0, 10))
    print(a)
    """
data_multi = [[0.7403949191793799, 0.509541136212647, 0.4503219509497285, 0.3500115992501378],
              [0.8709742920473218, 0.43884406983852386, 0.45436591748148203, 0.4526366014033556],
              [8.947684995830059, 8.947684995830059, 7.905856471508741, 6.5974732814356685],
              [33.37703644391149, 34.17523527145386, 36.39474736899137, 34.49558802973479]
              ]

for data in data_multi:

    # Compute statistics
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    n = len(data)

    # Confidence level
    confidence = 0.95
    alpha = 1 - confidence
    t_critical = t.ppf(1 - alpha/2, df=n-1)  # Two-tailed t critical value

    # Margin of error
    margin_of_error = t_critical * (std_dev / np.sqrt(n))

    # Confidence interval
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    print(f"Mean: {mean}")
    print(f"95% Confidence Interval: ({ci_lower}, {ci_upper})")
    """
