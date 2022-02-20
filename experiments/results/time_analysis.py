import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Set file dir as cwd
os.chdir(os.path.dirname(__file__))

with h5py.File("time_kmeans_V1.hdf5", "r") as file:
    times = file["times_data"][:]
    plt.hist(times)
    plt.show()
