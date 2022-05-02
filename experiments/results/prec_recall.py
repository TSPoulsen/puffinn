import h5py
import numpy as np
import os
from typing import Tuple
import argparse


DELTA = 0.01
START = -1.0
END = 1.0
RAN = np.arange(START,END,DELTA)

filenames = { 
            "g100_pass_filter_euc_perm.hdf5": "EP",
            "g100_pass_filter_maha_perm.hdf5": "MP",
            "g100_pass_filter_euc16_perm.hdf5": "E16P",
            "g100_pass_filter_maha16_perm.hdf5": "MP16",
            "g100_pass_filter_euc_no_perm.hdf5": "ENP",
            "g100_pass_filter_maha_no_perm.hdf5": "MNP",
            "g100_pass_filter_euc16_no_perm.hdf5": "E16NP",
            "g100_pass_filter_maha16_no_perm.hdf5": "M16NP"}

def calcPrecRecallLSH(diffs: np.array, true_neigh) -> Tuple[np.array, np.array]:
    int_range = np.arange(0, 64, 1)
    recalls = np.zeros(int_range.shape[0]) 
    precisions = np.zeros(int_range.shape[0]) 
    diffs = diffs.astype("float32") / 32 # To get average difference as there are 32 hashes to each sketch
    order = np.argsort(diffs)
    for i, row in enumerate(diffs):
        print(i, end="-", flush = True)
        row_order = order[i]
        idcs = np.searchsorted(row, int_range, sorter = row_order)
        for j, p_i in enumerate(idcs):
            passed = row_order[p_i:] 
            tp = np.intersect1d(passed,true_neigh[i]).shape[0]
            fp = true_neigh.shape[1] - tp 
            fn = passed.shape[0] - tp
            recalls[j] += ((tp+0.01)/((tp+fn)+0.01))
            precisions[j] += ((tp+0.01)/((tp + fp)+0.01))
    print()
    recalls /= order.shape[0]
    precisions /= order.shape[0]
    return recalls, precisions


def calcPrecRecallPQ(estimates: np.array, true_neigh: np.array) -> Tuple[np.array, np.array]:
    recalls = np.zeros(RAN.shape[0]) 
    precisions = np.zeros(RAN.shape[0]) 
    order = np.argsort(estimates)
    for i, row in enumerate(estimates):
        print(i,end ="-", flush=True)
        row_order = order[i]
        idcs = np.searchsorted(row, RAN, sorter = row_order)
        for j, p_i in enumerate(idcs):
            passed = row_order[p_i:] 
            tp = np.intersect1d(passed,true_neigh[i]).shape[0]
            fp = true_neigh.shape[1] - tp 
            fn = passed.shape[0] - tp
            recalls[j] += ((tp+0.01)/((tp+fn)+0.01))
            precisions[j] += ((tp+0.01)/((tp + fp)+0.01))
    print()
    recalls /= order.shape[0]
    precisions /= order.shape[0]
    return recalls, precisions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--force',
        action='store_true')
    parser.add_argument(
        '--k',
        type=int,
        default = 10
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(__file__))
    rf = h5py.File("rec_prec_results.hdf5","r+")
    data = h5py.File("../../data/glove-100-angular.hdf5","r")
    true_neigh = np.array(data["neighbors"])
    true_neigh[:,:args.k] # They are given in sorted order
    for fn in filenames.keys():
        print(fn)
        if (filenames[fn] in rf.keys() and str(args.k) in rf[filenames[fn]]) and not args.force:  continue
        hf = h5py.File(fn, "r")
        rec, prec = calcPrecRecallPQ(np.array(hf["estimated_inner"]), true_neigh)
        g = rf.get(filenames[fn],None)
        if not g: g = rf.create_group(filenames[fn])
        if g.get(str(args.k), ""): del g[str(args.k)]
        gk = g.create_group(str(args.k))
        gk.create_dataset("recalls", data = rec)
        gk.create_dataset("precisions", data = prec)
        hf.close()

    #hf = h5py.File("g100_pass_filter_lsh.hdf5", "r")
    #rec, prec = calcPrecRecallLSH(np.array(hf["bit_diffs"]), true_neigh)
    #if "LSH" not in rf.keys() or args.force:
        #g = rf.get(filenames[fn],rf.create_group(filenames[fn]))
        #if g.get(str(args.k), ""): del g[str(args.k)]
        #gk = g.create_group(str(args.k))
        #gk.create_dataset("recalls", data = rec)
        #gk.create_dataset("precisions", data = prec)
        #g = rf.get("LSH",rf.create_group("LSH"))
        #g.create_dataset("recalls", data = rec)
        #g.create_dataset("precisions", data = prec)
    #hf.close()


    data.close()
    rf.close()

