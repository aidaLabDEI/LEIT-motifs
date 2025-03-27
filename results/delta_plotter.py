import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    matplotlib.use("WebAgg")
    #matplotlib.rcParams.update({"text.usetex":True, "text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath"})
    xfmt = ScalarFormatter()
    xfmt.set_scientific(True)
    xfmt.set_powerlimits((1, 2))
    sns.set_theme(style="ticks", palette="muted")
    
    # !!! Delta plots
    data = pd.read_csv("results/delta_results.csv")
    # Find the different values in the first column
    ds_values = data["Dataset"].unique()
    sns.boxplot(data=data, x="distance", y="Dataset", hue="delta", palette=["r", "g", "b"], width=.8, gap=0.1,)

    
    means = pd.DataFrame(columns=["Dataset", "delta", "time"], data=[
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
        ["whales", 0.2, 823.73]]     
    )
    colors = ["r", "g", "b"]
    for i, dataset in enumerate(data["Dataset"].unique()):
     for j, delta in enumerate(data["delta"].unique()):
        subset = means[(means["Dataset"] == dataset) & (means["delta"] == delta)]
        if not subset.empty:
            mean_time = subset["time"].values[0]
            plt.text(
                x= 19.1,  # Adjust x position
                y=i -0.3 + j*0.3,  # Align with y-axis category
                s=f"{mean_time:.2f}s",  # Format text
                ha='center', 
                va='center', 
                fontsize=10, 
                color=colors[j],
            )
    # sns.stripplot(
    # data=data, x="distance", y="Dataset", hue="delta",
    # dodge=True, alpha=.25, zorder=1, legend=False, palette="muted"
    # )

    # # Show the conditional means, aligning each pointplot in the
    # # center of the strips by adjusting the width allotted to each
    # # category (.8 by default) by the number of hue levels
    # sns.pointplot(
    #     data=data, x="distance", y="Dataset", hue="delta",
    #     dodge=.8 - .8 / 3, palette="muted", errorbar=None,
    #     markers="d", markersize=4, linestyle="none",
    # )
    sns.despine(trim=True)
    plt.show()