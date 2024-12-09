import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    matplotlib.use("WebAgg")
    xfmt = ScalarFormatter()
    xfmt.set_scientific(True)
    xfmt.set_powerlimits((1,2))
    names = ["potentials", "evaporator", "RUTH", "weather", "whales"]

    # !!!K plots
    data = pd.read_csv("results/K_results.csv")
    # FInd the different values in the first column
    ds_values = data["Dataset"].unique()
    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharex=True, layout="constrained")
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        K_data = data[data["Dataset"] == ds_val]
        sns.lineplot(data=K_data, x="K", y="Time elapsed", color= "mediumseagreen", ax=axs[i // 2, i % 2])
        axs[i // 2, i % 2].set_title(names[i])
    sns.despine(offset=10, trim=True)
    sns.set_context("paper")
    fig.supxlabel("Concatenations")
    fig.supylabel("Time elapsed (s)")
    plt.show()

    # !!!L plots
    data = pd.read_csv("results/L_results.csv")
    # FInd the different values in the first column
    ds_values = data["Dataset"].unique()

    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharex=True, layout="constrained")
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        L_data = data[data["Dataset"] == ds_val]
        axs[i // 2, i % 2].stackplot(
            L_data["L"],
            L_data["Time int"],
            color="peachpuff",
            alpha=0.8,
            labels=["Hash time"],
        )
        axs[i // 2, i % 2].stackplot(
            L_data["L"],
            L_data["Time elapsed"],
            color="wheat",
            alpha=0.4,
            labels=["Search time"],
        )
        axs[i // 2, i % 2].plot(
            L_data["L"],
            L_data["Time elapsed"],
            color="orange",
            alpha=0.6,
            linewidth=1.2,
            marker="o",
        )
        axs[i // 2, i % 2].set_title(names[i])
        if i == 1:
            axs[i // 2, i % 2].legend(loc="upper right")
    sns.despine(offset=10, trim=True)
    sns.set_context("paper")
    fig.supxlabel("Repetitions")
    fig.supylabel("Time elapsed (s)")
    plt.show()
    
    # !!!r plots
    data = pd.read_csv("results/R_results.csv")
    # FInd the different values in the first column
    ds_values = data["Dataset"].unique()
    r = [4, 8, 16, 32]
    r_dc = [6, 8, 15, 32]
    r_dist = [16, 312, 212, 38106]

    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharex=True, layout="constrained")
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        r_data = data[data["Dataset"] == ds_val]
        sns.lineplot(data=r_data, x="r", y="dist_computed", color= "coral", ax=axs[i // 2, i % 2])

        axs[i // 2, i % 2].vlines(
            r_dc[i], 0, r_dist[i], linestyle="dotted", color="crimson"
        )
        axs[i//2, i%2].scatter(r_dc[i], r_dist[i], color="crimson", zorder=5, s=50, label="Self-tuned r")
        axs[i // 2, i % 2].set_title(names[i])
        axs[i//2,i%2].yaxis.set_major_formatter(xfmt)
        
    plt.legend()
    sns.despine(offset=1, trim=False)
    sns.set_context("paper")
    fig.supxlabel("discretization parameter r")
    fig.supylabel("Compared couples")
    plt.show()

    # Time complex plots
    
