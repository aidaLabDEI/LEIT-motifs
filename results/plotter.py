import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    matplotlib.use("WebAgg")
    xfmt = ScalarFormatter()
    xfmt.set_scientific(True)
    xfmt.set_powerlimits((1, 2))
    names = ["potentials", "evaporator", "RUTH", "weather", "whales"]

    r"""
    # !!!K plots
    data = pd.read_csv("results/K_results.csv")
    # FInd the different values in the first column
    ds_values = data["Dataset"].unique()
    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(2, 2, figsize=(5, 3), sharex=True, layout="constrained")
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        K_data = data[data["Dataset"] == ds_val]
        sns.lineplot(data=K_data, x="K", y="Time elapsed", color= "mediumseagreen", ax=axs[i // 2, i % 2], legend=False)
        axs[i // 2, i % 2].set_title(names[i])
        axs[i // 2, i % 2].set_xlabel('')
        axs[i // 2, i % 2].set_ylabel('')
        axs[i // 2, i % 2].spines["bottom"].set_bounds(4, 16)
        axs[i // 2, i % 2].set_xticks([4, 8, 12, 16])
    sns.despine(trim=True)
    sns.set_context("paper")
    fig.supxlabel("Concatenations - K")
    fig.supylabel("Time (s)")
    #plt.xticks(xticks)
    plt.show()

    # !!!L plots
    data = pd.read_csv("results/L_results.csv")
    # FInd the different values in the first column
    ds_values = data["Dataset"].unique()
    data = data.groupby(["Dataset", "L"]).mean().reset_index()
    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(2, 2, figsize=(5, 3), sharex=True, layout="constrained")
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        L_data = data[data["Dataset"] == ds_val]

        axs[i // 2, i % 2].stackplot(
            L_data["L"],
            L_data["Time elapsed"],
            color="cornflowerblue",
            alpha=0.65,
            labels=["Search"],
        )
        axs[i // 2, i % 2].stackplot(
            L_data["L"],
            L_data["Time int"],
            color="mediumslateblue",
            alpha=0.55,
            labels=["Hash"],
        )
        axs[i // 2, i % 2].set_title(names[i])
        axs[i // 2, i % 2].set_xlabel('')
        axs[i // 2, i % 2].set_ylabel('')

        if i == 0:
            axs[i // 2, i % 2].legend(loc="upper left")
    sns.despine(offset=10, trim=True)
    sns.set_context("paper")
    fig.supxlabel("Repetitions - L")
    fig.supylabel("Time (s)")
    plt.show()


    # !!!r plots
    data = pd.read_csv("results/R_results.csv")
    # FInd the different values in the first column
    ds_values = data["Dataset"].unique()
    r = [4, 8, 16, 32]
    r_dc = [6, 8, 15, 32]
    r_dist = [16, 312, 212, 38106]

    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(2, 2, figsize=(5, 3), sharex=True, layout="constrained")
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        r_data = data[data["Dataset"] == ds_val]
        sns.lineplot(data=r_data, x="r", y="dist_computed", color= "crimson", ax=axs[i // 2, i % 2])

        axs[i // 2, i % 2].vlines(
            r_dc[i], 0, r_dist[i], linestyle="dotted", color="coral"
        )
        axs[i//2, i%2].scatter(r_dc[i], r_dist[i], color="crimson", zorder=5, s=40, label="Self-tuned r")
        axs[i // 2, i % 2].set_title(names[i])
        axs[i//2,i%2].yaxis.set_major_formatter(xfmt)
        axs[i // 2, i % 2].set_xlabel('')
        axs[i // 2, i % 2].set_ylabel('')
        axs[i // 2, i % 2].spines["bottom"].set_bounds(2, 32)   
        axs[i // 2, i % 2].set_xticks([2, 4, 8, 16, 32])
    plt.legend()
    sns.despine(offset=1, trim=True)
    sns.set_context("paper")
    fig.supxlabel("Discretization parameter - r")
    fig.supylabel("Compared couples")
    plt.show()

    # Noise plot

    data = pd.read_csv("results/noise.csv")
    # FInd the different values in the first column
    ds_values = data["Dataset"].unique()
    fig, axs = plt.subplots(1, 1, figsize=(5, 3), layout="constrained")
    colors = ["palevioletred", "skyblue", "seagreen", "darkorange"]
    names = ["potentials", "evaporator", "ruth", "weather"]
    for val in ds_values:
        n_data = data[data["Dataset"] == val]
        sns.lineplot(
            data=n_data,
            x=" noise",
            y=" val",
            color=colors[val],
            alpha=0.7,
            label=names[val],
            ci= None,
        )
    axs.set_ylabel("Recall")
    axs.set_xlabel("Injected dimensions")
    sns.despine(offset=10.2)
    plt.ylim((0, 1.1))
    plt.show()

    # Multidim plot
    mstumptime=[3.65, 4.45, 37, 39] # 
    mtimeread = [3.65, 4.45,84.04, 1035.73, 2.7*24*60*60, 7.2*24*60*60, 8.4*24*60*60, 11.8*24*60*60]
    data = pd.read_csv("results/dist_time.csv")
    ds_values = data["dataset"].unique()
    fig, axs = plt.subplots(1, 1, figsize=(5, 6), layout="constrained")
    colors = ["crimson", "cornflowerblue", "mediumseagreen", "darkorange", "darkcyan","xkcd:dull green", "firebrick", "xkcd:marigold"]
    names = ["potentials", "evaporator", "ruth", "weather", "whales", "quake", "el_load", "ltmm"]
    for val in ds_values:
        n_data = data[data["dataset"] == val]
        sns.lineplot(
            data=n_data,
            x="Distance",
            y="Time (s)",
            color=colors[val],
            alpha=0.95,
            label=names[val],
            ci=None,
        )
        sns.scatterplot(
            data=n_data,
            x="Distance",
            y="Time (s)",
            color=colors[val],
            alpha=0.7,
            marker='X',
        )

        axs.axhline(mtimeread[val], color=colors[val], linestyle='dotted')
    plt.xscale("log")
    plt.yscale("log")
    sns.despine(trim=True)
    plt.minorticks_off()
    plt.show()
    #     if val == 0: 
    #         axs.axhline(mstumptime[val], color=colors[val], linestyle='dotted')
    #         axs.text(300, mstumptime[val]-2, f"MSTUMP: {mtimeread[val]} s", color=colors[val])
    #     elif val==1:
    #         axs.axhline(mstumptime[val], color=colors[val], linestyle='dotted')
    #         axs.text(300, mstumptime[val]+0.9, f"MSTUMP: {mtimeread[val]} s", color=colors[val])
    #     else:
    #         axs.text(300, mstumptime[val]-0.5, f"↑ MSTUMP: {mtimeread[val]} s", color=colors[val], backgroundcolor='white')
    # axs.set_ylim(0, 39)
    # axs.spines["left"].set_bounds(0, 35)
    # sns.despine(offset={"left": 10.2})
    # plt.show()
    """

    # Scalability plot
    data = pd.read_csv("results/scalability.csv")
    fig, axs = plt.subplots(1, 1, figsize=(5, 3), layout="constrained")
    colors = ["dimgray", "cornflowerblue", "crimson", "mediumseagreen"]
    names = ["MSTUMP","LEIT-motifs (medium)", "LEIT-motifs (hard)", "LEIT-motifs (easy)"]
    num = [0, 3, 1, 2]
    for val in num:
        n_data = data[data["Algo"] == val]
        if val == 1: 
            continue
        sns.lineplot(
            data=n_data,
            x="Size",
            y="Time (s)",
            color=colors[val],
            alpha=0.7,
            label=names[val],
            ci=None,
            linewidth=2,
        )

        sns.scatterplot(
            data=n_data,
            x="Size",
            y="Time (s)",
            color=colors[val],
            alpha=0.8,
            marker='o',
            s=50,
            )
        # Generate reference complexity curves
    size_range = np.logspace(np.log10(data["Size"].min()), np.log10(data["Size"].max()), 100)
    min_time = data["Time (s)"].min()
    max_time = data["Time (s)"].max()

    # Scale the reference curves for visualization
    linear_time = min_time * (size_range / size_range.min())  # O(n)
    quadratic_time = min_time * (size_range / size_range.min())**2  # O(n^2)

    # Plot reference lines
    plt.plot(size_range, linear_time, 'k-.', linewidth=1, label="O(n)", alpha= 0.8)  # Dashed black line
    plt.plot(size_range, quadratic_time, 'k:', linewidth=1, label="O(n²)", alpha=0.8)  # Dotted black line

    plt.yscale("log")
    plt.xscale("log")
    sns.despine(trim=True)
    plt.legend()
    plt.minorticks_off()
    plt.show()