import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 1e7

rmseBench_unserious = pd.read_csv("./wrs_rmse_curve.csv")
rmseBench = pd.concat([rmseBench_unserious])

print(rmseBench)

timeBench_unserious = pd.read_csv("./wrs_benchmark_sample_throughput2.csv")
timeBench_psa2_0 = pd.read_csv("./wrs_benchmark_sample_throughput2_psa2-0.csv")
timeBench_psa2_128 = pd.read_csv("./wrs_benchmark_sample_throughput2_psa2-128.csv")
# timeBench_unserious = pd.read_csv("./psa_benchmark_sample_throughput_heatmap.csv")
timeBench = pd.concat([timeBench_psa2_0, timeBench_psa2_128, timeBench_unserious])

timeBench["latency"] = timeBench["build_latency"] + timeBench["sample_latency"]


# Sort both dataframes by S for merge_asof
rmseBench_sorted = rmseBench.sort_values("S").reset_index(drop=True)
timeBench_sorted = timeBench.sort_values("S").reset_index(drop=True)


# Merge by the common grouping keys (here "N", "method", "group") and nearest S
bench = pd.merge_asof(
    left=timeBench_sorted,
    right=rmseBench_sorted,
    on="S",                    # the column on which to do the near match
    by=["N", "group", "method"],  # additional grouping keys
    direction="nearest",        # pick the row in timeBench whose S is closest
    tolerance=1000,
)

latencyBase = "latency"
bench = bench[["N", "S", "group", latencyBase, "rmse", "flushL2"]]

print(bench["group"].unique())
print(bench)

print("N:", bench["N"].unique());



def plotMe(df, label, color, method):
    df = df.sort_values(by = latencyBase).reset_index();
    df = df[df["group"] == method]
    cached = df[df["flushL2"]]
    not_cached = df[~df["flushL2"]]
    # plt.plot(not_cached[latencyBase], not_cached["rmse"], ":", color=color)
    plt.plot(cached[latencyBase], cached["rmse"], "-", color=color, label=label)


plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))

plotMe(bench, "its-baseline", "tab:blue", "ITS-0")
plotMe(bench, "its-coop", "tab:orange", "ITS-128")
plotMe(bench, "cutpoint", "tab:green", "Cutpoint")
plotMe(bench, "psa-baseline", "tab:red", "PSA2-0")
plotMe(bench, "psa-sectioned", "tab:brown", "PSA2-128")




ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Latency (ms)")
plt.ylabel("RMSE")
# plt.xscale("log")
plt.yscale("log")
plt.grid(True)

plt.xlim((0, 10))


plt.tight_layout()
plt.savefig("wrs_rmse_speed_construction.pdf", format="pdf")

plt.show()



