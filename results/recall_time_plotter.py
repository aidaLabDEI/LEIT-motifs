import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

if __name__ == "__main__":
    matplotlib.use("WebAgg")
    matplotlib.rcParams.update({"text.usetex":True, "text.latex.preamble": r"\usepackage{siunitx} \usepackage{stix} \usepackage{sansmath} \sansmath"})
    sns.set_theme(style="ticks", palette="muted")
    fig, axs = plt.subplots(1, 1, figsize=(5, 3), layout="constrained")
    recall_data = pd.read_csv("results/recall_results.csv")

    means = pd.DataFrame(
        columns=["Dataset", "delta", "time"],
        data=[
            ["potentials", 0.01, 0.51],
            ["potentials", 0.1, 0.45],
            ["potentials", 0.2, 0.42],
            ["potentials", 0.5, 0.325],
            ["potentials", 0.8, 0.32],
            ["evaporator", 0.01, 0.551],
            ["evaporator", 0.1, 0.55],
            ["evaporator", 0.2, 0.491],
            ["evaporator", 0.5, 0.49],
            ["evaporator", 0.8, 0.47],
            ["RUTH", 0.01, 8.10],
            ["RUTH", 0.1, 3.27],
            ["RUTH", 0.2, 3.22],
            ["RUTH", 0.5, 3.221],
            ["RUTH", 0.8, 3.222],
            ["weather", 0.01, 33.37],
            ["weather", 0.1, 32.45],
            ["weather", 0.2, 30.75],
            ["weather", 0.5, 17.32],
            ["weather", 0.8, 15.81],
            ["whales", 0.01, 6798.66],
            ["whales", 0.1, 1447.23],
            ["whales", 0.2, 823.73],
            ["whales", 0.5, 753.75],
            ["whales", 0.8, 329],
            ["el_load", 0.01, 10080],
            ["el_load", 0.1, 10080.1],
            ["el_load", 0.2, 10080.2],
            ["el_load", 0.5, 5401],
            ["el_load", 0.8, 3120],
            ["quake", 0.01, 12960],
            ["quake", 0.1, 12400],
            ["quake", 0.2, 11520],
            ["quake", 0.5, 10138.1],
            ["quake", 0.8, 10138],
            ["LTMM", 0.01, 1902],
            ["LTMM", 0.1, 1865],
            ["LTMM", 0.2, 696.92],
            ["LTMM", 0.5, 696.91],
            ["LTMM", 0.8, 696.9],
        ]
    )
    means = pd.merge(
        means,
        recall_data,
        how="left",
        left_on=["Dataset", "delta"],
        right_on=["Dataset", "delta"],
    )
    means = means.rename(columns={"recall": "Recall", "time": "Time (s)"})
    
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
        x="Time (s)",
        y="Recall",
        hue="Dataset",
        palette=colors,
        legend=False,
        errorbar=None,
        alpha=0.8,
    )
    sns.scatterplot(
        data=means,
        x="Time (s)",
        y="Recall",
        style="delta",
        hue="Dataset",
        palette=colors,
        legend=False,
        alpha=0.7,
    )
    markers = ["$0.01$",
               "$0.1$",
               "$0.2$",
               "$0.5$",
               "$0.8$",
               ]
    mark = ["o", "X", "s", "P", "D"]
    
    for index, delta in enumerate(means["delta"].unique()):
        plt.text(
            99.8,
            index * 0.05 + 0.4,
            markers[4-index],
            color="slategray",
            ha="left",
            va="center",
        )
        plt.scatter(
            80.8,
            index * 0.05 + 0.401,
            marker=mark[4-index],
            color="slategray",
            alpha=0.9,
            s= 20
        )
    plt.text(
        99.8,
        0.05*5 +0.4,
        "$\\delta$",
        color="slategray",
        ha="left",
        va="center",
    )
        
    plt.xlabel(r"Time (s)")
    plt.ylabel(r"Recall")
    plt.xscale("log")
    #plt.xlim(0.7, 1.09)
    #plt.yscale("log")
    # legend = axs.legend()
    # for text in legend.get_texts():
    #     text.set_text(r"\textsc{" + text.get_text() + "}")
    plt.minorticks_off()
    sns.despine(trim=True)
    plt.show()
