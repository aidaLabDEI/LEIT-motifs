import matplotlib.pyplot as plt, pandas as pd, numpy as np, os, sys

if __name__ == "__main__":
    data = pd.read_csv( "results/r_dataset0.csv")

    K_data = data[1:5]
    print(K_data)
    plt.fill_between(K_data['K'], K_data['Time elapsed'], color="mediumseagreen", alpha=0.4)
    plt.plot(K_data['K'], K_data['Time elapsed'], color="seagreen", alpha=0.6, linewidth =1.2, marker='o')
    plt.show()