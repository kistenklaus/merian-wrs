import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd

numPoints = 100
maxN = 2e8
minN = 1e5

# Load the data
flushL2 = pd.read_csv("block_scan_benchmark_flushL2.csv")
withL2 = pd.read_csv("block_scan_benchmark_L2.csv")
withL2_2 = pd.read_csv("block_scan_benchmark_L2_2.csv")
withL2_3 = pd.read_csv("block_scan_benchmark_L2_3.csv")
withL2_4 = pd.read_csv("block_scan_benchmark_L2_4.csv")
withL2_5 = pd.read_csv("block_scan_benchmark_L2_5.csv")

withL2 = pd.concat([withL2, withL2_2, withL2_3, withL2_4, withL2_5])

flushL2 = flushL2[(flushL2["N"] >= minN) & (flushL2["N"] <= maxN)]
withL2 = withL2[(withL2["N"] >= minN) & (withL2["N"] <= maxN)]

property = "memory_throughput"
aggregate = "max"


plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))

def merge_duplicates(df):
    methods = df.groupby(["method", "N", "group"]).agg(
            aggregate=(property, aggregate),
            )
    groups = methods.groupby(["group", "N"]).agg(
            aggregate=("aggregate", aggregate),
            )
    groups.sort_index()
    groups = groups.reset_index()
    groups = groups.rename(columns={"group": "method"})

    return (methods, groups)

methodsFlushL2,groupsFlushL2 = merge_duplicates(flushL2)
methodsWithL2,groupsWithL2 = merge_duplicates(withL2)

def binedAverage(df, B):
    bin_size = len(df) // B
    df_copy = df.copy()
    df_copy.loc[:, 'bin'] = np.arange(len(df_copy)) // bin_size
    df_copy.loc[:, 'bin'] = np.where(df_copy['bin'] >= B, B-1, df_copy['bin'])
    results = df_copy.groupby('bin').agg({
        'N': 'median',
        'aggregate': 'mean'
    }).reset_index(drop=True)
    return results
    

groupsFlushL2 = groupsFlushL2.groupby("N").agg(
        low=("aggregate", "min"),
        high=("aggregate", "max"),
        ).reset_index()


rakingWithL2 = binedAverage(groupsWithL2[groupsWithL2["method"] == "RAKING"], numPoints)
rankedWithL2 = binedAverage(groupsWithL2[groupsWithL2["method"] == "RANKED"], numPoints)
rankedStridedWithL2 = binedAverage(groupsWithL2[groupsWithL2["method"] == "RANKED-STRIDED"], numPoints)


# plt.plot(rakingFlushL2["N"], rakingFlushL2["aggregate"], label="raking")
# plt.plot(rankedFlushL2["N"], rankedFlushL2["aggregate"], label="ranked")
# plt.plot(rankedStridedFlushL2["N"], rankedStridedFlushL2["aggregate"], label="ranked-strided")

plt.plot(rakingWithL2["N"], rakingWithL2["aggregate"], label="raking")
plt.plot(rankedWithL2["N"], rankedWithL2["aggregate"], label="ranked")
plt.plot(rankedStridedWithL2["N"], rankedStridedWithL2["aggregate"], label="ranked-strided")


BinedFill = True
if not BinedFill:
    plt.fill_between(groupsFlushL2["N"], groupsFlushL2["low"], groupsFlushL2["high"], alpha=0.5, label="flushed L2")
else :
    low = binedAverage(groupsFlushL2.rename(columns={"low": "aggregate"}), 200)
    high = binedAverage(groupsFlushL2.rename(columns={"high": "aggregate"}), 200)
    plt.fill_between(low["N"], low["aggregate"], high["aggregate"], alpha=0.5, label="flushed L2")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Element Count")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log")
plt.grid(True)

plt.text(3e7, 525, "470Gb/s")


plt.tight_layout()
plt.savefig("block_scan.pdf", format="pdf")


plt.show()




