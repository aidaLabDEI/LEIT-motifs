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
    names = ["RUTH", "weather", "whales", "el_load"]
    mem_data = pd.read_csv("results/MemLK_results.csv")

    # !!!K plots
    data = pd.read_csv("results/K_results.csv")
    # FInd the different values in the first column
    ds_values = data["Dataset"].unique()
    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(2, 2, figsize=(5.9, 3.9), sharex=True, layout="constrained")
    for i, ds_val in enumerate(ds_values):
        if i < 2:
            continue
        # Get the data for the current value
        K_data = data[data["Dataset"] == ds_val]
        Mem_dataK = mem_data[mem_data["Dataset"] == ds_val]
        Mem_dataK = Mem_dataK[Mem_dataK["test"] == 0]
        i = i - 2
        sns.lineplot(
            data=K_data,
            x="K",
            y="Time elapsed",
            color="mediumseagreen",
            ax=axs[i // 2, i % 2],
            legend=False,
            zorder=100,
        )
        ax2 = axs[i // 2, i % 2].twinx()
        sns.lineplot(
            data=Mem_dataK,
            x="K",
            y="Memory (GB)",
            color="dimgray",
            ax=ax2,
            legend=False,
            linewidth=1,
            alpha=0.6,
            zorder=1,
        )
        axs[i // 2, i % 2].set_title(r"\textsc{" + names[i] + "}", fontsize=10)
        axs[i // 2, i % 2].set_xlabel("")
        axs[i // 2, i % 2].set_ylabel("")
        ax2.set_ylabel("")
        ax2.tick_params(axis="y", labelcolor="dimgray")

        if i % 2 == 1:
            for n, label in enumerate(ax2.get_yticklabels()):
                if n == 0 and i // 2 == 0:
                    label.set_visible(False)
                elif i // 2 == 1 and n > 2:
                    label.set_visible(False)

        axs[i // 2, i % 2].tick_params(axis="y", labelcolor="mediumseagreen")
        axs[i // 2, i % 2].spines["bottom"].set_bounds(4, 16)
        axs[i // 2, i % 2].set_xticks([4, 8, 12, 16])
    sns.despine(right=False, trim=True)
    sns.set_context("paper")
    fig.supxlabel("Concatenations - K")
    fig.supylabel("Time (s)", color="mediumseagreen")
    fig.text(
        0.98,
        0.5,
        r"Memory (GB)",
        ha="center",
        va="center",
        fontsize=10,
        rotation=270,
        color="dimgray",
    )

    plt.show()

    # !!!L plots
    # data = pd.read_csv("results/L_results.csv")
    # FInd the different values in the first column
    # ds_values = data["Dataset"].unique()
    # data = data.groupby(["Dataset", "L"]).mean().reset_index()
    # Create a plot with ds_values subplots
    # fig, axs = plt.subplots(2, 2, figsize=(5.9, 3.9), sharex=True, layout="constrained")
    # for i, ds_val in enumerate(ds_values):
    #     if i < 2:
    #         continue

    #     Get the data for the current value
    #     L_data = data[data["Dataset"] == ds_val]
    #     Mem_dataL = mem_data[mem_data["Dataset"] == ds_val]
    #     Mem_dataL = Mem_dataL[Mem_dataL["test"] == 1]
    #     i = i - 2
    #     sns.lineplot(
    #         data=L_data,
    #         x="L",
    #         y="Time elapsed",
    #         color="cornflowerblue",
    #         ax=axs[i // 2, i % 2],
    #         legend=False,
    #         errorbar=None,
    #     )
    #     ax2 = axs[i // 2, i % 2].twinx()
    #     sns.lineplot(
    #         data=Mem_dataL,
    #         x="L",
    #         y="Memory (GB)",
    #         color="dimgray",
    #         ax=ax2,
    #         legend=False,
    #         linewidth=1,
    #         alpha=0.6,
    #     )
    #     axs[i // 2, i % 2].spines["bottom"].set_bounds(10, 400)
    #     axs[i // 2, i % 2].set_xticks([10, 50, 100, 150, 200, 400])
    #     axs[i // 2, i % 2].stackplot(
    #         L_data["L"],
    #         L_data["Time elapsed"],
    #         color="cornflowerblue",
    #         alpha=0.65,
    #         labels=["Search"],
    #     )
    #     axs[i // 2, i % 2].stackplot(
    #         L_data["L"],
    #         L_data["Time int"],
    #         color="mediumslateblue",
    #         alpha=0.55,
    #         labels=["Hash"],
    #     )
    #     ax2.set_ylabel("")
    #     yticks = ax2.get_yticks()
    #     if len(yticks) > 2:
    #         ax2.set_yticks(yticks[1:-1])
    #     ax2.tick_params(axis="y", labelcolor="dimgray")
    #     ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    #     axs[i // 2, i % 2].tick_params(axis="y", labelcolor="cornflowerblue")
    #     axs[i // 2, i % 2].set_title(r"\textsc{" + names[i] + "}", fontsize=10)
    #     axs[i // 2, i % 2].set_xlabel("")
    #     axs[i // 2, i % 2].set_ylabel("")
    # sns.despine(right=False, trim=True)
    # sns.set_context("paper")
    # fig.supxlabel("Repetitions - L")
    # fig.supylabel("Time (s)", color="cornflowerblue")
    # fig.text(
    #     0.98,
    #     0.5,
    #     r"Memory (GB)",
    #     ha="center",
    #     va="center",
    #     fontsize=10,
    #     rotation=270,
    #     color="dimgray",
    # )
    # plt.show()

    # # !!!r plots
    # data = pd.read_csv("results/R_results.csv")
    # # FInd the different values in the first column
    # ds_values = data["Dataset"].unique()
    # r = [4, 8, 16, 32]
    # r_dc = [6, 8, 15, 32]
    # r_dist = [16, 312, 212, 38106]

    # # Create a plot with ds_values subplots
    # fig, axs = plt.subplots(2, 2, figsize=(5.9, 3.9), sharex=True, layout="constrained")
    # for i, ds_val in enumerate(ds_values):
    #     if i < 2:
    #         continue
    #     # Get the data for the current value
    #     r_data = data[data["Dataset"] == ds_val]
    #     i = i - 2

    #     sns.lineplot(data=r_data, x="r", y="dist_computed", color= "crimson", ax=axs[i // 2, i % 2])

    #     axs[i // 2, i % 2].vlines(
    #         r_dc[i], 0, r_dist[i], linestyle="dotted", color="coral"
    #     )
    #     axs[i//2, i%2].scatter(r_dc[i], r_dist[i], color="crimson", zorder=5, s=40, label="Self-tuned r")
    #     axs[i // 2, i % 2].set_title(r"\textsc{" + names[i] + "}")
    #     axs[i//2,i%2].yaxis.set_major_formatter(xfmt)
    #     axs[i // 2, i % 2].set_xlabel('')
    #     axs[i // 2, i % 2].set_ylabel('')
    #     axs[i // 2, i % 2].spines["bottom"].set_bounds(2, 32)
    #     axs[i // 2, i % 2].set_xticks([2, 4, 8, 16, 32])
    # plt.legend()
    # sns.despine(trim=True)
    # sns.set_context("paper")
    # fig.supxlabel("Discretization parameter - r")
    # fig.supylabel("Compared couples")
    # plt.show()

    # Fusion LK plot
    # K_data = pd.read_csv("results/K_results.csv")
    # L_data = pd.read_csv("results/L_results.csv")
    # ds_values = K_data["Dataset"].unique()
    # fig, axs = plt.subplots(2, 2, figsize=(6.5, 5), sharex=True, layout="constrained")

    # colors = {"K": "mediumseagreen", "L": "cornflowerblue"}

    # for i, ds_val in enumerate(ds_values):
    #     ax = axs[i // 2, i % 2]

    #     # Get subsets
    #     subset_K = K_data[K_data["Dataset"] == ds_val]
    #     subset_L = L_data[L_data["Dataset"] == ds_val]

    #     # Normalize the x-axis of K and L
    #     K_norm_x = (subset_K["K"] - 4) / 12  # Scale to 0-1
    #     L_norm_x = subset_L["L"] / 400  # Scale to 0-1

    #     # K Plot
    #     sns.lineplot(
    #         x=K_norm_x,
    #         y=subset_K["Time elapsed"],
    #         color=colors["K"],
    #         ax=ax,
    #         label="K",
    #         legend=False,
    #         errorbar=("pi", 50),
    #     )

    #     # L Plot (same y-axis, different x-axis)
    #     ax.set_xticks(np.linspace(0, 1, 4))
    #     ax.set_xticklabels(np.linspace(4, 16, 4, dtype=int))  # Restore original scale

    #     ax2 = ax.twiny()
    #     sns.lineplot(
    #         x=L_norm_x,
    #         y=subset_L["Time elapsed"],
    #         color=colors["L"],
    #         ax=ax,
    #         label="L",
    #         legend=False,
    #         errorbar=("pi", 50),
    #     )

    #     # Adjust x-ticks
    #     ax2.set_xticks(np.linspace(0, 1, 5))
    #     ax2.set_xticklabels(np.linspace(0, 400, 5, dtype=int))  # Restore original scale

    #     # Labels and titles
    #     ax.set_xlabel("Concatenations - K", color=colors["K"])
    #     ax2.set_xlabel("Repetitions - L", color=colors["L"])
    #     ax.set_ylabel("Time (s)")
    #     ax.set_title(r"\textsc{" + names[ds_val] + "}", fontsize=10)

    #     # Match tick colors
    #     ax.tick_params(axis="x", colors=colors["K"])
    #     ax2.tick_params(axis="x", colors=colors["L"])

    # sns.despine(top=False, trim=True)
    # sns.set_context("paper")

    # plt.show()
