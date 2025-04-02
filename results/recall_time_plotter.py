import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

if __name__ == "__main__":
    matplotlib.use("WebAgg")
    # matplotlib.rcParams.update({"text.usetex":True, "text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath"})
    sns.set_theme(style="ticks", palette="muted")
    fig, axs = plt.subplots(1, 1, figsize=(5, 3), layout="constrained")
    recall_data = pd.read_csv("results/recall_results.csv")

    means = pd.DataFrame(
        columns=["Dataset", "delta", "time"],
        data=[
            ["potentials", 0.01, 0.51],
            ["potentials", 0.1, 0.45],
            ["potentials", 0.2, 0.42],
            ["evaporator", 0.01, 0.55],
            ["evaporator", 0.1, 0.55],
            ["evaporator", 0.2, 0.49],
            ["RUTH", 0.01, 8.10],
            ["RUTH", 0.1, 3.27],
            ["RUTH", 0.2, 3.22],
            ["weather", 0.01, 33.37],
            ["weather", 0.1, 32.45],
            ["weather", 0.2, 30.75],
            ["whales", 0.01, 6798.66],
            ["whales", 0.1, 1447.23],
            ["whales", 0.2, 823.73],
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

    sns.lineplot(
        data=means,
        x="Recall",
        y="Time (s)",
        hue="Dataset",
        legend=False,
        palette="pastel",
        alpha=0.6,
    )
    sns.scatterplot(
        data=means,
        x="Recall",
        y="Time (s)",
        hue="Dataset",
        style="delta",
        legend="brief",
    )
    # plt.xscale("log")
    plt.xlim(0.5, 1.09)
    # plt.yscale("log")
    plt.minorticks_off()
    sns.despine(trim=True)
    plt.show()
