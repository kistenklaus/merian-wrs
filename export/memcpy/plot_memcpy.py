from functools import singledispatch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from numpy._core.numerictypes import single
import pandas as pd
import os

numPoints = 1000
maxN = 1e8
minN = 1e5

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))


dirname = os.path.dirname(__file__)
bench = pd.read_csv(os.path.join(dirname, "./memcpy_benchmark.csv"))

bench = bench[(bench["N"] >= minN) & (bench["N"] <= maxN)]

property = "memory_throughput"
aggregate = "max" # We are interessted only in peeks (as memcpy only acts as a reference)

def merge_duplicates(df):
    methods = df.groupby(["method", "N"]).agg(
            aggregate=(property, aggregate),
            ).sort_index().reset_index()
    return methods

bench = merge_duplicates(bench)

def binedAverage(df, B):
    if B > len(df):
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

computeWithL2 = bench[bench["method"] == "Memcpy"]
computeWithL2 = binedAverage(computeWithL2, numPoints)

computeFlushL2 = bench[bench["method"] == "Memcpy-FlushL2"]
computeFlushL2 = binedAverage(computeFlushL2, numPoints)


plt.plot(computeWithL2["N"], computeWithL2["aggregate"], ":", label="without flushing", color="tab:blue")
plt.plot(computeFlushL2["N"], computeFlushL2["aggregate"], "-", label="with flusing", color="tab:blue")


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("4-Byte Element Count")
plt.ylabel("Throughput (GB/s)")
plt.xscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig("memcpy.pdf", format="pdf")


plt.show()
