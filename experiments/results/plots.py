import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import h5py
from typing import List, Tuple
import argparse

import os
colors = plt.get_cmap("tab20c").colors

types = {
    "mahalanobis_4_perm.hdf5": {
        "label" : "$M=4$",
        "color" : colors[2]
    },
    "mahalanobis_8_perm.hdf5": {
        "label" : "$M=8$",
        "color" : colors[1]
    },
    "mahalanobis_16_perm.hdf5": { 
        "label" : "$M=16$",
        "color" : colors[0]
    },
    "mahalanobis_4_no_perm.hdf5": {
        "label" : "$M=4$",
        "color" : colors[2]
    },
    "mahalanobis_8_no_perm.hdf5": {
        "label" : "$M=8$" ,
        "color": colors[1]
    },
    "mahalanobis_16_no_perm.hdf5": { 
        "label" : "$M=16$",
        "color" : colors[0]
    },
    "euclidean_4_perm.hdf5": {
        "label" : "$M=4$",
        "color" : colors[6]
    },
    "euclidean_8_perm.hdf5": { 
        "label" : "$M=8$",
        "color" : colors[5]
    },
    "euclidean_16_perm.hdf5": { 
        "label" : "$M=16$",
        "color" : colors[4]
    },
    "euclidean_4_no_perm.hdf5": {
        "label" : "$M=4$",
        "color" : colors[6]
    },
    "euclidean_8_no_perm.hdf5": {
        "label" : "$M=8$",
        "color" : colors[5]
    },
    "euclidean_16_no_perm.hdf5": { 
        "label" : "$M=16$",
        "color" : colors[4]
    },
    "lsh_1_first.hdf5" : {
        "label": "LSH_Bit", 
        "n_sketches": 1,
        "color" : (0,0,0)
        },
    "lsh_5_first.hdf5" : {
        "label" : "LSH_Time",
        "n_sketches":  5,
        "color" : (1,0,0) 
    }
}
    
d_prefix = ["glove-100-angular", "nytimes-256-angular", "deep-image-96-angular"]

TOP = False # if True Gives only differences in estimation of top 100 inner products for each query
QUICK = True # If True quickly creates plot, with almost no data to get a visual of how the plot would look like

def infer_estimates(coll: np.array, n_sketches) -> np.array:
    """
    Calculates the inner products given collision probabilities between query sketch and data hashes
    """
    return np.cos(np.pi * (1-(coll/(64*n_sketches))))



def plot_err(estimates: np.array, true: np.array, ax: plt.Axes, label: str, col: Tuple[int,int,int]) -> None:
    """
    top: bool
    """
    SAMPLE_SIZE = 1_000_000
    if QUICK:
        estimates = estimates[:,:500]
        true = true[:,:500]
        SAMPLE_SIZE = 1000
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
    print(diffs.shape)
    sample = np.random.choice(diffs, size = min(SAMPLE_SIZE,diffs.shape[0]), replace = False)
    print("std:", np.std(diffs))
    sns.kdeplot(sample, label=label, ax = ax, color = col)
    ax.set(xlabel=None, ylabel=None)

def create_plot(data_f: str, files: List[str], ax: plt.Axes) -> None:

    for fn in files:
        data_path = data_f + "_" + fn
        if not os.path.isfile(data_path):
            print(data_path, "should exists but doesn't")
            return
        print(data_path)
        h5data = h5py.File(data_path, "r")

        if "section0" in h5data:
            all_ests = []
            all_true = []
            for section in h5data:
                all_ests.append(h5data[section]["estimated_inner"])
                all_true.append(h5data[section]["true_inner"])
            estimates = np.array(all_ests)
            del all_ests
            true = np.array(all_true)
            del all_true

        elif "lsh" in fn:
            estimates = infer_estimates(np.array(h5data["collisions"]), types[fn]["n_sketches"])
            true = np.array(h5data["true_inner"])
        else:
            estimates = np.array(h5data["estimated_inner"])
            true = np.array(h5data["true_inner"])
        plot_err(estimates, true, ax, label = types[fn]["label"], col = types[fn]["color"])
        h5data.close()
    


if __name__ == "__main__":
    no_perm = lambda ls: [fn for fn in ls if "no_perm" in fn and "lsh" not in fn]
    perm = lambda ls: [fn for fn in ls if "no_perm" not in fn and "lsh" not in fn]
    def set_legend(fig,ax):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels)
    fig, axes = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize=(10,4))
    all_euc = [fn for fn in types.keys() if "euclidean" in fn]
    all_maha = [fn for fn in types.keys() if "mahalanobis" in fn]

    QUICK = False
    TOP = False
    create_plot("glove-100-angular", no_perm(all_maha), axes[0])
    create_plot("glove-100-angular", perm(all_maha), axes[1])

    # Plot configuration
    fig.subplots_adjust(left=0.08,
                        bottom=0.12, 
                        right=0.99, 
                        top=0.92, 
                        wspace=0.05,
                        hspace=0.1)
    set_legend(fig, axes[0])
    axes[0].set_xlim(-0.5, 0.5)
    axes[1].set_xlim(-0.5, 0.5)
    axes[0].axvline(0, linestyle = "--", color="grey")
    axes[1].axvline(0, linestyle = "--", color="grey")
    axes[0].set_title("Non Permuted", fontweight="semibold")
    axes[1].set_title("Permuted", fontweight="semibold")
    axes[0].set_ylabel("Density", fontsize=15)
    fig.text(0.53, 0.02 ,"Estimation Error", ha="center", fontsize=15)

    fig.savefig("temp.svg", format = "svg")# "../../plots/glove-100-angular_euclidean.svg", format = "svg")
    exit()
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


    print(types)
    exit()
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