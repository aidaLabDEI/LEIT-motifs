import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    matplotlib.use("WebAgg")
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
    data = pd.read_csv("results/dist_time.csv")
    ds_values = data["dataset"].unique()
    fig, axs = plt.subplots(1, 1, figsize=(5, 6), layout="constrained")
    colors = [
        "crimson",
        "cornflowerblue",
        "mediumseagreen",
        "darkorange",
        "darkcyan",
        "xkcd:dull green",
        "firebrick",
        "xkcd:marigold",
    ]
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
            marker="X",
        )

        axs.axhline(mtimeread[val], color=colors[val], linestyle="dotted")
    legend = axs.legend()
    for text in legend.get_texts():
        text.set_text(r"\textsc{" + text.get_text() + "}")
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
