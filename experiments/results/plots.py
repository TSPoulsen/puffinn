import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import h5py
from typing import List
import argparse

import os
labels = ["M8_Perm", "M16_Perm", "M8_NoPerm", "M16_NoPerm", "Viktors Joker"]
types = {
    "mahalanobis_8_perm.hdf5": 0,
    "mahalanobis_16_perm.hdf5": 1,
    "mahalanobis_8_no_perm.hdf5": 2,
    "mahalanobis_16_no_perm.hdf5": 3,
    "euclidean_8_perm.hdf5": 0,
    "euclidean_16_perm.hdf5": 1,
    "euclidean_32_perm.hdf5": 4,
    "euclidean_8_no_perm.hdf5": 2,
    "euclidean_16_no_perm.hdf5": 3,
    "lsh_single.hdf5" : "LSH_S",
    "lsh_total.hdf5" : "LSH_T" }
    
d_prefix = ["glove-100-angular", "nytimes-256-angular", "deep-image-96-angular"]

SAMPLE_SIZE = 10000000
TOP = False # if True Gives only differences in estimation of top 100 inner products for each query

def infer_estimates(coll: np.array) -> np.array:
    """
    Calculates the inner products given collision probabilities between query sketch and data hashes
    """
    return np.cos(np.pi * (1-coll))




def plot_err(estimates: np.array, true: np.array, ax: plt.Axes, label: str) -> None:
    """
    top: bool
    """
    #estimates = estimates[:, :500]
    #true = true[:, :500]
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
        diffs = true-estimates
        diffs = diffs.reshape(-1)
    sample = np.random.choice(diffs, size = min(SAMPLE_SIZE,diffs.shape[0]), replace = False)
    sns.kdeplot(sample, label=label, ax = ax)
    ax.set(xlabel=None, ylabel=None)

def create_plot(data_f: str, files: List[str], ax: plt.Axes) -> None:

    for fn in files:
        data_path = data_f + "_" + fn
        assert(os.path.isfile(data_path), "file %s doesn't exist" % data_path)
        print(data_path)
        h5data = h5py.File(data_path, "r")
        if "lsh" in fn:
            estimates = infer_estimates(np.array(h5data["collision_prob"]))
        else:
            estimates = np.array(h5data["estimated_inner"])
        true = np.array(h5data["true_inner"])
        print(types[fn],labels)
        plot_err(estimates, true, ax, labels[types[fn]])
        h5data.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--top',
        action='store_true')
    args = parser.parse_args()
    TOP = args.top


    ax = plt.gca()
    fig, axes = plt.subplots(nrows = 2, ncols = 3, sharey = True, figsize=(20,10))
    for i, ax_r in enumerate(axes):
        for j, ax in enumerate(ax_r):
            ax.axvline(0.0, linestyle="--")
            ax.set_xlim(-0.5,0.5)
            ax.set_ylim(0,10)
            if i == 0: ax.set_title(d_prefix[j])
            loss = "mahalanobis" if i == 0 else "euclidean"
            files = [fn for fn in types if loss in fn and "no_perm" not in fn]
            create_plot(d_prefix[j], files, ax)
            break
        break

    #fig.suptitle("Estimation errors" + (" for top 100" if TOP else ""))
    fig.text(0.02, 0.25 ,"Density", va="center", rotation="vertical", fontsize=15)
    fig.text(0.02, 0.75 ,"Density", va="center", rotation="vertical", fontsize=15)
    #fig.text(0.5, 0.04 ,"Estimation Error", ha="center", fontsize=15)
    fig.show()
    fig.legend(loc="upper right")
    # set the spacing between subplots
    plt.subplots_adjust(left=0.05,
                        bottom=0.05, 
                        right=0.99, 
                        top=0.95, 
                        wspace=0.1,
                        hspace=0.2)
    plot_name = input("What is name of plot?")
    if plot_name:
        fig.savefig("../../plots/"+ plot_name + ("_top" if TOP else "") + ".svg", format="svg")