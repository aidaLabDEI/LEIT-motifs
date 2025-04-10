import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

if __name__ == "__main__":
    matplotlib.use("WebAgg")
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{siunitx} \usepackage{stix} \usepackage{sansmath} \sansmath",
        }
    )
    sns.set_theme(style="ticks", palette="muted")
    fig, axs = plt.subplots(1, 1, figsize=(5, 3), layout="constrained")
    recall_data = pd.read_csv("results/recall_results.csv")

    means = pd.DataFrame(
        columns=["Dataset", "delta", "time"],
        data=[
            ["potentials", 0.05, 0.51],
            ["potentials", 0.1, 0.45],
            ["potentials", 0.2, 0.42],
            ["potentials", 0.4, 0.325],
            ["potentials", 0.8, 0.32],
            ["evaporator", 0.05, 0.551],
            ["evaporator", 0.1, 0.55],
            ["evaporator", 0.2, 0.491],
            ["evaporator", 0.4, 0.49],
            ["evaporator", 0.8, 0.47],
            ["RUTH", 0.05, 8.10],
            ["RUTH", 0.1, 3.27],
            ["RUTH", 0.2, 3.24],
            ["RUTH", 0.4, 3.23],
            ["RUTH", 0.8, 3.22],
            ["weather", 0.05, 33.37],
            ["weather", 0.1, 32.45],
            ["weather", 0.2, 30.75],
            ["weather", 0.4, 17.32],
            ["weather", 0.8, 15.81],
            ["whales", 0.05, 6798.66],
            ["whales", 0.1, 1947.23],
            ["whales", 0.2, 1823.73],
            ["whales", 0.4, 1753.75],
            ["whales", 0.8, 1329],
            ["el_load", 0.05, 10080],
            ["el_load", 0.1, 10080.1],
            ["el_load", 0.2, 10080.2],
            ["el_load", 0.4, 5401],
            ["el_load", 0.8, 3120],
            ["quake", 0.05, 12960],
            ["quake", 0.1, 12400],
            ["quake", 0.2, 11520],
            ["quake", 0.4, 10138.1],
            ["quake", 0.8, 10138],
            ["LTMM", 0.05, 1902],
            ["LTMM", 0.1, 1865],
            ["LTMM", 0.2, 696.92],
            ["LTMM", 0.4, 696.91],
            ["LTMM", 0.8, 696.9],
        ],
    )
    means = pd.merge(
        means,
        recall_data,
        how="left",
        left_on=["Dataset", "delta"],
        right_on=["Dataset", "delta"],
    )
    means = means.rename(columns={"recall": "Recall", "time": "Time (s)"})
    
    #Min Max Sca
    means["Time (s)"] = means.groupby("Dataset")["Time (s)"].transform(
        lambda x: x / x.max()
    )

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

    sns.lineplot(
        data=means,
        x="Recall",
        y="Time (s)",
        hue="Dataset",
        palette=colors,
        legend=False,
        errorbar=None,
        alpha=0.4,
    )
    sns.scatterplot(
        data=means,
        x="Recall",
        y="Time (s)",
        style="delta",
        hue="Dataset",
        palette=colors,
        legend=False,
        s=65,
    )
    markers = [
        "$0.05$",
        "$0.1$",
        "$0.2$",
        "$0.4$",
        "$0.8$",
    ]
    mark = ["o", "X", "s", "P", "D"]

    for index, delta in enumerate(means["delta"].unique()):
        plt.text(
            0.23,
            index * 0.05 + 0.3,
            markers[4 - index],
            color="slategray",
            ha="left",
            va="center",
        )
        plt.scatter(
            0.2,
            index * 0.05 + 0.301,
            marker=mark[4 - index],
            color="slategray",
            alpha=0.9,
            s=20,
        )
    plt.text(
        0.25,
        0.05 * 5 + 0.3,
        "$\\delta$",
        color="slategray",
        ha="left",
        va="center",
    )

    plt.xlabel(r"Recall")
    plt.ylabel(r"Fraction of Time")
    #plt.xscale("log")
    # plt.xlim(0.7, 1.09)
    # plt.yscale("log")
    # legend = axs.legend()
    # for text in legend.get_texts():
    #     text.set_text(r"\textsc{" + text.get_text() + "}")
    plt.minorticks_off()
    sns.despine(trim=True)
    plt.show()
