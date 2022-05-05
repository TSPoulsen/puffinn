import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import h5py
from typing import List
import argparse

import os

types = {
    "mahalanobis_8_perm.hdf5": "M8P",
    "mahalanobis_16_perm.hdf5": "M16P",
    "mahalanobis_8_no_perm.hdf5": "M8NP",
    "mahalanobis_16_no_perm.hdf5": "M16NP",
    "euclidean_8_no_perm.hdf5": "E8NP",
    "euclidean_16_no_perm.hdf5": "E16NP",
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
    print("Creating plot for",label)
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

def create_plot(data_f: str, files: List[str], ax: plt.Axes) -> None:

    input_d = h5py.File(data_f + "_" + list(types.keys())[4], "r")
    true = np.array(input_d["true_inner"])
    input_d.close()
    for fn in files:
        data_path = data_f + "_" + fn
        #assert(os.path.isfile(data_path), "file %s doesn't exist" % data_path)
        if not os.path.isfile(data_path): continue
        h5data = h5py.File(data_path, "r")
        if "lsh" in fn:
            estimates = infer_estimates(np.array(h5data["collision_prob"]))
        else:
            estimates = np.array(h5data["estimated_inner"])
        plot_err(estimates, true, ax, types[fn])
        h5data.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--top',
        action='store_true')
    args = parser.parse_args()
    TOP = args.top


    ax = plt.gca()
    fig, axes = plt.subplots(nrows = 1, ncols = 3)
    for i, ax in enumerate(axes):
        ax.set_xlabel("Error")
        ax.set_title(d_prefix[i])
        files = [fn for fn in types if "mahalanobis" not in fn and "euclidean" not in fn]
        create_plot(d_prefix[i], files, ax)
    plt.title("Estimation error" + (" for top 100" if TOP else ""))
    plt.legend()
    plt.show()