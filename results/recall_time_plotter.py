from icecream import ic
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

if __name__ == "__main__":
    # matplotlib.use("WebAgg")
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{siunitx} \usepackage{stix} \usepackage{sansmath} \sansmath",
        }
    )
    sns.set_theme(style="ticks", palette="muted")
    fig, axs = plt.subplots(1, 1, figsize=(5, 3), layout="constrained")
    recall_data = pd.read_csv("results/csv/recall_results.csv")

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

    # colors = [
    #     "crimson",
    #     "cornflowerblue",
    #     "mediumseagreen",
    #     "darkorange",
    #     "darkcyan",
    #     "xkcd:dull green",
    #     "firebrick",
    #     "xkcd:pale purple",
    # ]
    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())

    # Drop the lines with delta 0.8
    means = means[means["delta"] != 0.8]
    
    label_data = means[means["delta"] == 0.4]
    for i, (_, row) in enumerate(label_data.iterrows()):
        dataset = row["Dataset"]
        vertical_alignment = "center"
        if dataset in ["el_load"]:
            vertical_alignment = "bottom"
        elif dataset in ["weather", "LTMM"]:
            vertical_alignment = "top"
        plt.text(
            x=row["Recall"] - 0.02,
            y=row["Time (s)"],
            s= r"\textsc{" + dataset + "}",
            color=colors[i],
            ha="right",
            va=vertical_alignment,
        )

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
    ]
    mark = ["o", "X", "s", "P"]
    

    base_x = 0.45
    for index, delta in enumerate(means["delta"].unique()):
        plt.text(
            base_x + 0.03,
            index * 0.05 + 0.3,
            markers[3- index],
            color="slategray",
            ha="left",
            va="center",
        )
        plt.scatter(
            base_x,
            index * 0.05 + 0.301,
            marker=mark[3- index],
            color="slategray",
            alpha=0.9,
            s=20,
        )
    plt.text(
        base_x + 0.05,
        0.05 * 4 + 0.3,
        "$\\delta$",
        color="slategray",
        ha="left",
        va="center",
    )
    # # Plot the legend for the datasets
    # for index, dataset in enumerate(reversed(means["Dataset"].unique())):
    #     plt.text(
    #         0.35,
    #         index * 0.06 + 0.56,
    #         r"\textsc{" + dataset + "}",
    #         #color=colors[index],
    #         ha="left",
    #         va="center",
    #         fontsize=11,
    #     )
    #     # A small line
    #     plt.plot(
    #         [0.3, 0.34],
    #         [index * 0.06 + 0.56, index * 0.06 + 0.56],
    #         color=colors[7-index],
    #         linewidth=1,
    #     )

    plt.xlabel(r"Recall ($\geq 1 ‑\delta$)")
    plt.ylabel(r"Fraction of Time")
    #plt.xscale("log")
    # plt.xlim(0.7, 1.09)
    # plt.yscale("log")
    # legend = axs.legend()
    # for text in legend.get_texts():
    #     text.set_text(r"\textsc{" + text.get_text() + "}")
    plt.minorticks_off()
    # sns.despine(trim=True)
    sns.despine(top=True, right=True)
    if plt.isinteractive():
        plt.show()
    else:
        plt.savefig("figures/time_recall_r.pdf")
