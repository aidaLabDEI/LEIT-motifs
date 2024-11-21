import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

if __name__ == "__main__":
    matplotlib.use("WebAgg")
    names = ["potentials", "evaporator", "RUTH", "weather", "whales"]

    # !!!K plots
    data = pd.read_csv("results/K_results.csv")
    # FInd the different values in the first column
    ds_values = data["Dataset"].unique()

    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        K_data = data[data["Dataset"] == ds_val]
        axs[i // 2, i % 2].fill_between(
            K_data["K"], K_data["Time elapsed"], color="mediumseagreen", alpha=0.4
        )
        axs[i // 2, i % 2].plot(
            K_data["K"],
            K_data["Time elapsed"],
            color="seagreen",
            alpha=0.6,
            linewidth=1.2,
            marker="o",
        )
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
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
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
    r_dc = [0, 0, 0, 0]

    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        r_data = data[data["Dataset"] == ds_val]
        axs[i // 2, i % 2].fill_between(
            r_data["r"], r_data["dist_computed"], color="coral", alpha=0.4
        )
        axs[i // 2, i % 2].plot(
            r_data["r"],
            r_data["dist_computed"],
            color="firebrick",
            alpha=0.6,
            linewidth=1.2,
            marker="o",
        )
        axs[i // 2, i % 2].vlines(
            r[i], 0, r_dc[i], linestyle="dotted", color="crimson", label="tuned r"
        )
        axs[i // 2, i % 2].set_title(names[i])
    sns.despine(offset=1, trim=False)
    sns.set_context("paper")
    fig.supxlabel("discretization parameter r")
    fig.supylabel("Distances computed")
    plt.show()
