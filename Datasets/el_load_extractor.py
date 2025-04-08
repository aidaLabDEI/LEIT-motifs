import wfdb
import matplotlib


if __name__ == "__main__":
    matplotlib.use("WebAgg")
    path = "Datasets/FL010"
    data, fields = wfdb.rdsamp(path)
    print(fields)
    print(data.shape)
