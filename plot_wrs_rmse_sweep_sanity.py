import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rmseBench = pd.read_csv("./wrs_rmse_sweep.csv")


method = "PSA2-0"
N = 1015637

rmseBench = rmseBench[rmseBench["group"] == method]

print(rmseBench["N"].unique())

rmseBench = rmseBench[rmseBench["N"] == N]

plt.plot(rmseBench["S"], rmseBench["rmse"])

plt.grid(which="both");
plt.xscale("log")
plt.yscale("log")

plt.show();

print(rmseBench)

