import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np

if __name__ == "__main__":
    # matplotlib.use("WebAgg")
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath",
        }
    )
    # Scalability plot
    data = pd.read_csv("results/csv/scalability.csv")
    fig, axs = plt.subplots(1, 1, figsize=(5, 3), layout="constrained")
    colors = ["dimgray", "crimson", "mediumseagreen", "cornflowerblue", "dimgray"]
    names = [
        "MSTUMP (synth)",
        "MOMENTI (synth)",
        "MOMENTI (easy)",
        "MOMENTI (LTMM)",
        "MSTUMP (LTMM)",
    ]
    num = [0, 1, 2, 3, 4]
    # Generate reference complexity curves
    size_range = np.logspace(
        np.log10(data["Size"].min()), np.log10(data["Size"].max()), 100
    )
    min_time = data["Time (s)"].min()
    max_time = data["Time (s)"].max()

    # Scale the reference curves for visualization
    linear_time = min_time * (size_range / size_range.min())  # O(n)
    quadratic_time = min_time * (size_range / size_range.min()) ** 2  # O(n^2)

    # Plot reference lines
    # plt.plot(size_range, linear_time, 'k-.', linewidth=1, label="O(n)", alpha= 0.8)  # Dashed black line
    # plt.plot(size_range, quadratic_time, 'k:', linewidth=1, label="O(n²)", alpha=0.8)  # Dotted black line
    for val in num:
        if val == 2:
            continue
        n_data = data[data["Algo"] == val]
        if val < 3:
            sns.lineplot(
                data=n_data,
                x="Size",
                y="Time (s)",
                color=colors[val],
                alpha=0.7,
                label=names[val],
                ci=None,
                linewidth=1.5,
                linestyle="--",
            )
        else:
            sns.lineplot(
                data=n_data,
                x="Size",
                y="Time (s)",
                color=colors[val],
                alpha=0.7,
                label=names[val],
                ci=None,
                linewidth=1.5,
            )
        sns.scatterplot(
            data=n_data,
            x="Size",
            y="Time (s)",
            color=colors[val],
            alpha=0.8,
            marker="X",
            s=50,
        )

    plt.yscale("log")
    plt.xscale("log")
    # sns.despine(trim=True)
    sns.despine(top=True, right=True)
    legend = axs.legend()
    for text in legend.get_texts():
        text.set_text(r"\textsc{" + text.get_text() + "}")
    plt.minorticks_off()
    momenti_data = data[data["Algo"] == 3]
    axs.spines["left"].set_bounds(min(data["Time (s)"]), max(momenti_data["Time (s)"]))
    axs.spines["bottom"].set_bounds(min(data["Size"]), max(data["Size"]))
    # plt.show()
    plt.savefig("figures/scalability.pdf")
