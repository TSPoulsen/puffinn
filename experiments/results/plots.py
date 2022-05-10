import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import h5py
from typing import List
import argparse

import os
types = {
    "mahalanobis_8_perm.hdf5": "Maha_M8",
    "mahalanobis_16_perm.hdf5": "Maha_M16",
    "mahalanobis_8_no_perm.hdf5":"Maha_M8" ,
    "mahalanobis_16_no_perm.hdf5": "Maha_M16",
    "euclidean_8_perm.hdf5": "Euc_M8",
    "euclidean_16_perm.hdf5": "Euc_M16",
    "euclidean_8_no_perm.hdf5":"Euc_M8",
    "euclidean_16_no_perm.hdf5": "Euc_M16" }
    #"mahalanobis_32_perm.hdf5": "Maha_M32",
    #"euclidean_32_perm.hdf5": "Euc_M32"}
lsh_types = {
    "lsh_2.hdf5" : "LSH_Bit",
    "lsh_16.hdf5" : "LSH_Time" }
    
d_prefix = ["glove-100-angular", "nytimes-256-angular", "deep-image-96-angular"]

SAMPLE_SIZE = 10000000
TOP = False # if True Gives only differences in estimation of top 100 inner products for each query
QUICK = False # If True quickly creates plot, with almost no data to get a visual of how the plot would look like

def infer_estimates(coll: np.array) -> np.array:
    """
    Calculates the inner products given collision probabilities between query sketch and data hashes
    """
    return np.cos(np.pi * (1-coll))




def plot_err(estimates: np.array, true: np.array, ax: plt.Axes, label: str) -> None:
    """
    top: bool
    """
    if (TOP):
        order = np.argsort(true)
        order = order[:,-100:]
        top_est = []
        top_true = []
        for i in range(true.shape[0]):
            top_true.append(true[i][order[i]])
            top_est.append(estimates[i][order[i]])
        top_est = np.concatenate(top_est, axis = 0)
        top_true = np.concatenate(top_true, axis = 0)
        diffs = top_est - top_true
    else:
        diffs = estimates - true
        diffs = diffs.reshape(-1)
    sample = np.random.choice(diffs, size = min(SAMPLE_SIZE,diffs.shape[0]), replace = False)
    sns.kdeplot(sample, label=label, ax = ax)
    ax.set(xlabel=None, ylabel=None)

def create_plot(data_f: str, files: List[str], ax: plt.Axes) -> None:

    for fn in files:
        data_path = data_f + "_" + fn
        if not os.path.isfile(data_path):
            return
        print(data_path)
        h5data = h5py.File(data_path, "r")
        if "lsh" in fn:
            estimates = infer_estimates(np.array(h5data["collision_prob"]))
        else:
            estimates = np.array(h5data["estimated_inner"])
        true = np.array(h5data["true_inner"])
        if QUICK:
            true = true[:,:500]
            estimates = estimates[:,:500]
        plot_err(estimates, true, ax,types[fn])
        h5data.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--top',
        action='store_true')
    parser.add_argument(
        '--quick',
        action='store_true')
    args = parser.parse_args()
    TOP = args.top
    QUICK = args.quick


    # Change rows to be perm vs noperm
    fig, axes = plt.subplots(nrows = 3, ncols = 2, sharey = True, figsize=(10,15))
    if len(axes.shape) == 1: axes = axes.reshape(1, axes.shape[0])
    for i in range(axes.shape[0]):
        ax_r =  axes[i]
        for j in range(axes.shape[1]):
            ax = axes[i,j]
            ax.axvline(0.0, linestyle="--")
            if TOP:
                ax.set_xlim(-0.9,0.1)
            else:
                ax.set_xlim(-0.5,0.5)
            ax.set_ylim(0,10)
            title = "No Permutation" if j == 0 else "Permuted Vectors"
            if i == 0: ax.set_title(title, fontweight="semibold")
            perm_filter = (lambda name: "no_perm" in name) if j == 0 else (lambda name: "no_perm" not in name)
            if QUICK: filter = lambda name: perm_filter(name) and "16" in name and "mahalanobis" in name
            else: filter = perm_filter
            files = [fn for fn in types if filter(fn)]
            create_plot(d_prefix[i], files, ax)

    #fig.suptitle("Estimation errors" + (" for top 100" if TOP else ""))
    fig.text(0.02, 0.5 ,"Density", va="center", rotation="vertical", fontsize=15)
    #fig.text(0.02, 0.75 ,"Density", va="center", rotation="vertical", fontsize=15)
    #fig.text(0.04, 0.95 ,"No Permutation", fontweight="semibold", fontsize=20)
    #fig.text(0.04, 0.49 ,"Permuted Vectors", fontweight="semibold", fontsize=20)
    #fig.text(0.48, 0.95 ,"GloVe1M", fontweight="semibold", fontsize=20)
    fig.text(0.52, 0.02 ,"Estimation Error", ha="center", fontsize=15)
    #fig.text(0.5, 0.04 ,"Estimation Error", ha="center", fontsize=15)
    fig.show()
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", fontsize=13)
    # set the spacing between subplots
    plt.subplots_adjust(left=0.08,
                        bottom=0.12, 
                        right=0.99, 
                        top=0.92, 
                        wspace=0.1,
                        hspace=0.25)
    plot_name = input("What is name of plot?")
    if plot_name:
        fig.savefig("../../plots/"+ plot_name + ("_top" if TOP else "") + ".svg", format="svg")