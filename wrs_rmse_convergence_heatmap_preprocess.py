import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

# ------------------------------------------------------
# 1) Load data, compute throughput, select best method
# ------------------------------------------------------

bench_its_cutpoint = pd.read_csv("./wrs_benchmark_its_cutpoint_heatmap.csv")

bench_its = bench_its_cutpoint.loc[
    (bench_its_cutpoint["group"] == "ITS-0") |
    (bench_its_cutpoint["group"] == "ITS-128") |
    (bench_its_cutpoint["group"] == "ITS-128-pArray")
]
bench_its.loc[bench_its["group"] == "ITS-128-pArray", "group"] = "ITS-128"
bench_its.loc[bench_its["group"] == "ITS-0", "group"] = "ITS-128"

bench_psa = pd.read_csv("./wrs_benchmark.csv")

bench_cutpoint = bench_its_cutpoint[
    bench_its_cutpoint["group"] == "Cutpoint-128"
]

bench = pd.concat([bench_its, bench_cutpoint, bench_psa], ignore_index=True)

latencyBase = "latency"
bench["latency"] = bench["build_latency"] + bench["sampling_latency"]
bench["throughput"] = bench["S"] / (bench[latencyBase] * 1e-3) * 1e-9

print(bench["group"].unique())

maxLatency = np.min(bench.groupby(["N", "group"])[latencyBase].max())
minLatency = np.min(bench[latencyBase])
print(f"maxLatency:{maxLatency}ms,   minLatency:{minLatency}ms");

# bench = bench[bench[latencyBase] < maxLatency]

latencyBucketMargin = 0.005
bench["bin_latency"] = (bench[latencyBase] / latencyBucketMargin).astype(int);


rmseBench = pd.read_csv("./wrs_rmse_sweep.csv")

print(rmseBench["group"].unique())

# print("minmax-rmse:", minmaxRmse);

rmseBench = rmseBench[(rmseBench["rmse"] > 1e-7) & (rmseBench["rmse"] < 2e-6)]


merged = pd.merge(bench, rmseBench, how="inner", on=["group", "N", "S"])

# for group in rmseBench["group"].unique():
#     psa128 = rmseBench[rmseBench["group"] == group]
#     psa128 = psa128[psa128["N"] == 2855764]
#     plt.plot(psa128["S"], psa128["rmse"], label=group)
#
# plt.xscale("log")
# plt.yscale("log")
# plt.show()

print(rmseBench)

best = merged.loc[merged.groupby(["bin_latency", "N"])["rmse"].idxmin()].reset_index(drop=True)

print(best["N"].unique())

test = best[best["N"] == 642480]

test = test.sort_values("bin_latency");
print("max-bin:", np.max(best["bin_latency"]));
print(test[["N", "latency", "bin_latency", "rmse", "group"]].to_string())

cum_best_rmse = None
cum_best_group = None

# Iterate over each row (each latency bin) for current N.
for idx, row in test.iterrows():
    # If it's the first row or current RMSE is lower than the best seen so far, update the cumulative best.
    if cum_best_rmse is None or row["rmse"] < cum_best_rmse:
        cum_best_rmse = row["rmse"]
        cum_best_group = row["group"]
    
    # Update only the "group" property to reflect the best performing method so far.
    test.loc[idx, "group"] = cum_best_group
    test.loc[idx, "rmse"] = cum_best_rmse

print(test[["N", "S", "latency", "bin_latency", "rmse", "group"]].to_string())
