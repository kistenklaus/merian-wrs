import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rmseBench0 = pd.read_csv("./wrs_rmse_sweep_wrong.csv")
rmseBench1 = pd.read_csv("./wrs_rmse_sweep.csv")

rmseBench = pd.concat([rmseBench0, rmseBench1])

print(rmseBench["group"].unique())

# method = "PSA2-128"
N = 1015637

# rmseBench = rmseBench[rmseBench["group"] == method]

print(rmseBench["N"].unique())

rmseBench = rmseBench[rmseBench["N"] == N]

if (len(rmseBench) == 0):
    exit(-1)


for group in rmseBench["group"].unique():
    df = rmseBench[rmseBench["group"] == group]
    plt.plot(df["S"], df["rmse"], label=group)

# plt.plot(rmseBench["S"], rmseBench["rmse"])

plt.legend()
plt.grid(which="both");
plt.xscale("log")
plt.yscale("log")

plt.show();

print(rmseBench)

