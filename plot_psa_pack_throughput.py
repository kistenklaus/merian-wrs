import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd

numPoints = 100
maxN = 2e8
minN = 1e5

# Load the data
bench_unserious = pd.read_csv("psa_pack_benchmark_throughput.csv")

bench = pd.concat([bench_unserious])

property = "latency"
aggregate = "mean"


plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))

def merge_duplicates(df):
    methods = df.groupby(["method", "flushL2", "group", "N"]).agg(
            latency=("latency", "mean"),
            )
    groups = methods.groupby(["group", "flushL2", "N"]).agg(
            latency=("latency", "mean"),
            )
    groups.sort_index()
    groups = groups.reset_index()
    groups = groups.rename(columns={"group": "method"})

    return (methods, groups)

_,groups = merge_duplicates(bench)

groups["throughput"] = groups["N"] / (groups["latency"] * 1e-3) * 1e-9;

def binedAverage(df, B):
    bin_size = len(df) // B
    df_copy = df.copy()
    df_copy.loc[:, 'bin'] = np.arange(len(df_copy)) // bin_size
    df_copy.loc[:, 'bin'] = np.where(df_copy['bin'] >= B, B-1, df_copy['bin'])
    results = df_copy.groupby('bin').agg({
        'N': 'median',
        'throughput': 'mean',
        'latency': 'mean',
    }).reset_index(drop=True)
    return results

def selectMethod(df, method):
    df = df[(df["method"] == method)]
    withL2 = df[~df["flushL2"]]
    flushL2 = df[df["flushL2"]]
    withL2 : pd.DataFrame
    flushL2 : pd.DataFrame
    return (withL2, flushL2)

_, pack1 = selectMethod(groups, "pack1")
_, pack2 = selectMethod(groups, "pack2")
_, pack4 = selectMethod(groups, "pack4")
_, pack8 = selectMethod(groups, "pack8")
_, pack16 = selectMethod(groups, "pack16")
_, pack32 = selectMethod(groups, "pack32")

pack1 = binedAverage(pack1, numPoints)
pack2 = binedAverage(pack2, numPoints)
pack4 = binedAverage(pack4, numPoints)
pack8 = binedAverage(pack8, numPoints)
pack16 = binedAverage(pack16, numPoints)
pack32 = binedAverage(pack32, numPoints)

property = "latency"

plt.plot(pack1["N"], pack1[property], color="tab:blue", label="1-thread")
plt.plot(pack2["N"], pack2[property], color="tab:orange", label="2-thread")
plt.plot(pack4["N"], pack4[property], color="tab:green", label="4-thread")
plt.plot(pack8["N"], pack8[property], color="tab:red", label="8-thread")
plt.plot(pack16["N"], pack16[property], color="tab:purple", label="16-thread")
plt.plot(pack32["N"], pack32[property], color="tab:brown", label="32-thread")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Amount of items")
plt.ylabel("Billion Packs / Second")
plt.xscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig("psa_pack_throughput.pdf", format="pdf")


plt.show()




