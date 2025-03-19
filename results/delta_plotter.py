import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    #matplotlib.use("WebAgg")
    #matplotlib.rcParams.update({"text.usetex":True, "text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath"})
    xfmt = ScalarFormatter()
    xfmt.set_scientific(True)
    xfmt.set_powerlimits((1, 2))
    names = ["potentials", "evaporator", "RUTH", "weather", "whales"]
    
    # !!!Violin delta plots
    data = pd.read_csv("results/delta_results.csv")
    # FInd the different values in the first column
    ds_values = data["Dataset"].unique()
    sns.boxenplot(data=data, x="distance", y="Dataset", hue="delta", palette="summer")
    sns.despine(trim=True)
    plt.show()