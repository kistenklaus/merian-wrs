from functools import singledispatch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from numpy._core.numerictypes import single
import pandas as pd

numPoints = 100
maxN = 2e8
minN = 1e5

flushL2_1 = pd.read_csv("./scan_benchmark_flushL2_1.csv")
flushL2_2 = pd.read_csv("./scan_benchmark_flushL2_2.csv")
flushL2_3 = pd.read_csv("./scan_benchmark_flushL2_3.csv")

withL2_1 = pd.read_csv("./scan_benchmark_withL2.csv")
withL2_2 = pd.read_csv("./scan_benchmark_withL2_2.csv")

flushL2 = pd.concat([flushL2_1, flushL2_2, flushL2_3])
withL2 = pd.concat([withL2_1, withL2_2])

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


singleDispatchFlushL2 = groupsFlushL2[groupsFlushL2["method"] == "SingleDispatch-RANKED-STRIDED"]
singleDispatchFlushL2 = binedAverage(singleDispatchFlushL2, numPoints)

blockWiseFlushL2 = groupsFlushL2[groupsFlushL2["method"] == "BlockWise-RANKED-STRIDED"]
blockWiseFlushL2 = binedAverage(blockWiseFlushL2, numPoints)

singleDispatchWithL2 = groupsWithL2[groupsWithL2["method"] == "SingleDispatch-RANKED-STRIDED"]
singleDispatchWithL2 = binedAverage(singleDispatchWithL2, numPoints)

blockWiseWithL2 = groupsWithL2[groupsWithL2["method"] == "BlockWise-RANKED-STRIDED"]
blockWiseWithL2 = binedAverage(blockWiseWithL2, numPoints)



plt.plot(singleDispatchFlushL2["N"], singleDispatchFlushL2["aggregate"], "-", color="tab:blue", label="single-dispatch")
plt.plot(singleDispatchWithL2["N"], singleDispatchWithL2["aggregate"], ':', color="tab:blue")
plt.plot(blockWiseFlushL2["N"], blockWiseFlushL2["aggregate"], "-", color="tab:orange", label="block-wise")
plt.plot(blockWiseWithL2["N"], blockWiseWithL2["aggregate"], ":", color="tab:orange")

plt.text(4.2e7, 500, "465Gb/s")
plt.text(4.2e7, 260, "230Gb/s")


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


plt.tight_layout()
plt.savefig("scan.pdf", format="pdf")


plt.show()




