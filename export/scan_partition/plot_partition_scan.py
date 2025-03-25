import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

numPoints = 10000
maxN = 2e8
minN = 1e5

# bench_1 = pd.read_csv("./partition_scan_benchmark_1.csv")
# bench_2 = pd.read_csv("./partition_scan_benchmark_2.csv")
# bench_3 = pd.read_csv("./partition_scan_benchmark_3.csv")

dirname = os.path.dirname(__file__)
bench = pd.read_csv(os.path.join(dirname, "./partition_scan_benchmark.csv"))

benchWithoutElem = bench[~bench["write-partition"]]
benchWithoutElem = benchWithoutElem[(benchWithoutElem["N"] >= minN) & (benchWithoutElem["N"] <= maxN)]

property = "memory_throughput"
aggregate = "max"

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))

def merge_duplicates(df, property):
    methods = df.groupby(["flushL2", "method", "N", "group"]).agg(
            aggregate=(property, aggregate),
            )
    groups = methods.groupby(["flushL2", "group", "N"]).agg(
            aggregate=("aggregate", aggregate),
            )
    groups.sort_index()
    groups = groups.reset_index()
    groups = groups.rename(columns={"group": "method"})

    return (methods, groups)

def merge_duplicates_memcpy(df):
    methods = df.groupby(["method", "N"]).agg(
            aggregate=(property, aggregate),
            ).sort_index().reset_index()
    return methods

methods,groups = merge_duplicates(benchWithoutElem, property)

def selectMethod(df, method):
    df = df[(df["method"] == method)]
    withL2 = df[~df["flushL2"]]
    flushL2 = df[df["flushL2"]]

    return (withL2, flushL2)

def binedAverage(df, B):
    if (len(df) <= B):
        return df
    bin_size = len(df) // B
    df_copy = df.copy()
    df_copy.loc[:, 'bin'] = np.arange(len(df_copy)) // bin_size
    df_copy.loc[:, 'bin'] = np.where(df_copy['bin'] >= B, B-1, df_copy['bin'])
    results = df_copy.groupby('bin').agg({
        'N': 'median',
        'aggregate': 'mean'
    }).reset_index(drop=True)
    return results

singleDispatchWithL2, singleDispatchFlushL2 = selectMethod(groups, "SingleDispatch-RANKED-STRIDED")
singleDispatchFlushL2 = binedAverage(singleDispatchFlushL2, numPoints)
singleDispatchWithL2 = binedAverage(singleDispatchWithL2, numPoints)

blockWiseWithL2, blockWiseFlushL2 = selectMethod(groups, "BlockWise-RANKED-STRIDED")
blockWiseFlushL2 = binedAverage(blockWiseFlushL2, numPoints)
blockWiseWithL2 = binedAverage(blockWiseWithL2, numPoints)




plt.plot(singleDispatchFlushL2["N"], singleDispatchFlushL2["aggregate"], "-", color="tab:blue", label="single-dispatch")
plt.plot(singleDispatchWithL2["N"], singleDispatchWithL2["aggregate"], ':', color="tab:blue")

plt.plot(blockWiseFlushL2["N"], blockWiseFlushL2["aggregate"], "-", color="tab:orange", label="block-wise")
plt.plot(blockWiseWithL2["N"], blockWiseWithL2["aggregate"], ':', color="tab:orange")

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
# plt.ylim(top=500)
plt.xscale("log")
plt.grid(True)


plt.tight_layout()
# plt.savefig("partition_scan.pdf", format="pdf")


property = "throughput"

_,groupsWithoutElem = merge_duplicates(benchWithoutElem, property)

benchWithElem = bench[bench["write-partition"]]
_, groupsWithElem = merge_duplicates(benchWithElem, property)

withElemWithL2, withElemFlushL2 = selectMethod(groupsWithElem, "SingleDispatch-RANKED-STRIDED")
withoutElemWithL2, withoutElemFlushL2 = selectMethod(groupsWithoutElem, "SingleDispatch-RANKED-STRIDED")

withElemWithL2 = binedAverage(withElemWithL2, numPoints)
withElemFlushL2 = binedAverage(withElemFlushL2, numPoints)
withoutElemWithL2 = binedAverage(withoutElemWithL2, numPoints)
withoutElemFlushL2 = binedAverage(withoutElemFlushL2, numPoints)


plt.figure(figsize=(8, 3))

plt.plot(withElemWithL2["N"], withElemWithL2["aggregate"], ":", color="tab:green")
plt.plot(withElemFlushL2["N"], withElemFlushL2["aggregate"], "-",color="tab:green", label="write elements")

plt.plot(withoutElemWithL2["N"], withoutElemWithL2["aggregate"], ":", color="tab:blue")

plt.plot(withoutElemFlushL2["N"], withoutElemFlushL2["aggregate"], "-", color="tab:blue", label="only indices");


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
plt.ylabel("Billion Items / Second")
# plt.ylim(top=500)
plt.xscale("log")
plt.grid(True)

plt.tight_layout()
# plt.savefig("partition_scan_with_elem.pdf", format="pdf")

plt.show()
