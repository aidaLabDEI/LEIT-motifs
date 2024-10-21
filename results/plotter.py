import matplotlib.pyplot as plt, pandas as pd, numpy as np, os, sys, seaborn as sns

if __name__ == "__main__":
    names = ["potentials", "evaporator", "RUTH", "weather"]

    # !!!K plots
    data = pd.read_csv( "results/K_results.csv")
    # FInd the different values in the first column
    ds_values = data['Dataset'].unique()

    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(len(ds_values), 1, figsize=(10, 10), sharex=True)
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        K_data = data[data['Dataset'] == ds_val]
        axs[i].fill_between(K_data['K'], K_data['Time elapsed'], color="mediumseagreen", alpha=0.4)
        axs[i].plot(K_data['K'], K_data['Time elapsed'], color="seagreen", alpha=0.6, linewidth =1.2, marker='o')
        axs[i].set_title(names[i])
    sns.despine(offset=10, trim=True)
    sns.set_context("paper")
    fig.supxlabel("Concatenations")
    fig.supylabel("Time elapsed (s)")
    plt.show()

    # !!!L plots
    data = pd.read_csv( "results/L_results.csv")
    # FInd the different values in the first column
    ds_values = data['Dataset'].unique()

    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(len(ds_values), 1, figsize=(10, 10), sharex=True)
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        L_data = data[data['Dataset'] == ds_val]
        axs[i].fill_between(L_data['L'], L_data['Time elapsed'], color="wheat", alpha=0.4)
        axs[i].plot(L_data['L'], L_data['Time elapsed'], color="orange", alpha=0.6, linewidth =1.2, marker='o')
        axs[i].set_title(names[i])
    sns.despine(offset=10, trim=True)
    sns.set_context("paper")
    fig.supxlabel("Repetitions")
    fig.supylabel("Time elapsed (s)")
    plt.show()

    # !!!r plots
    data = pd.read_csv( "results/r_results.csv")
    # FInd the different values in the first column
    ds_values = data['Dataset'].unique()

    # Create a plot with ds_values subplots
    fig, axs = plt.subplots(len(ds_values), 1, figsize=(10, 10), sharex=True)
    for i, ds_val in enumerate(ds_values):
        # Get the data for the current value
        r_data = data[data['Dataset'] == ds_val]
        axs[i].fill_between(r_data['r'], r_data['dist_computed'], color="coral", alpha=0.4)
        axs[i].plot(r_data['r'], r_data['dist_computed'], color="firebrick", alpha=0.6, linewidth =1.2, marker='o')
        axs[i].set_title(names[i])
    sns.despine(offset=1, trim=True)
    sns.set_context("paper")
    fig.supxlabel("discretization parameter r")
    fig.supylabel("Distances computed")
    plt.show()