import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    # matplotlib.use("WebAgg")
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath",
        }
    )
    xfmt = ScalarFormatter()
    xfmt.set_scientific(True)
    xfmt.set_powerlimits((1, 2))
    names = ["potentials", "evaporator", "RUTH", "weather", "whales"]
    r"""
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
            label=names[val],
            legend=False,
        )
    axs.set_ylabel("Recall")
    axs.set_xlabel("Injected dimensions")
    sns.despine(trim=True)
    plt.show()
    """
    # Multidim plot

    mstumptime = [3.65, 4.45, 37, 39]  #
    mtimeread = [
        3.65,
        4.45,
        84.04,
        1035.73,
        2.7 * 24 * 60 * 60,
        7.2 * 24 * 60 * 60,
        8.4 * 24 * 60 * 60,
        11.8 * 24 * 60 * 60,
    ]
    data = pd.read_csv("results/csv/dist_time.csv")
    ds_values = data["dataset"].unique()
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
    colors = [
        "crimson",
        "cornflowerblue",
        "mediumseagreen",
        "darkorange",
        "darkcyan",
        "xkcd:dull green",
        "firebrick",
        "xkcd:pale purple",
    ]
    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    names = [
        "potentials",
        "evaporator",
        "RUTH",
        "weather",
        "whales",
        "quake",
        "el_load",
        "LTMM",
    ]
    baseline_label_x = [
        0.3,
        100,
        0.3,
        0.3,
        0.3,
        30,
        3,
        0.3,
    ]

    for val in ds_values:
        n_data = data[data["dataset"] == val]
        minx = n_data["Distance"].min()
        miny = n_data["Time (s)"].min()
        label_x = minx - 0.2 * minx
        label_y = miny
        horizontal_alignment = "right"
        vertical_alignment = "center"
        if names[val] in ["LTMM", "potentials"]:
            label_x = minx
            label_y = miny + 0.2 * miny
            horizontal_alignment = "left"
            vertical_alignment = "bottom"
        plt.text(
            x=label_x,
            y=label_y,
            s=r"\textsc{" + names[val] + "}",
            color=colors[val],
            ha=horizontal_alignment,
            va=vertical_alignment,
        )
        sns.lineplot(
            data=n_data,
            x="Distance",
            y="Time (s)",
            color=colors[val],
            alpha=0.95,
            label=names[val],
            errorbar=None,
            linewidth=1,
        )
        sns.scatterplot(
            data=n_data,
            x="Distance",
            y="Time (s)",
            color=colors[val],
            alpha=0.7,
            marker="X",
        )

        axs.axhline(
            mtimeread[val],
            color=colors[val],
            # linestyle=(0, (1, 10)),
            linestyle=(0, (3, 1)),
            linewidth=0.7,
        )
        y_adj = dict(evaporator=0.3, potentials=-0.3, el_load=+50000, quake=-50000)
        plt.text(
            x=baseline_label_x[val],
            y=mtimeread[val] + y_adj.get(names[val], 0),
            s=r"\textsc{" + names[val] + "} (Mstump)",
            color=colors[val],
            ha="left",
            va="center",
            fontsize=7,
            bbox=dict(facecolor="white", linewidth=0, pad=0),
        )
    # legend = axs.legend(loc= (0.01, 0.2))
    # for text in legend.get_texts():
    #     text.set_text(r"\textsc{" + text.get_text() + "}")
    axs.get_legend().remove()
    axs.spines["left"].set_bounds(min(data["Time (s)"]), max(data["Time (s)"]))
    axs.spines["bottom"].set_bounds(min(data["Distance"]), max(data["Distance"]))
    plt.xscale("log")
    plt.yscale("log")
    # sns.despine(trim=True)
    sns.despine(top=True, right=True)
    plt.minorticks_off()
    if plt.isinteractive():
        plt.show()
    else:
        plt.savefig("figures/multisub_small.pdf")

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
