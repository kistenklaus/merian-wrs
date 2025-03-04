import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

numPoints = 100
maxN = 1e10
minN = 1e5

bench_unserious = pd.read_csv("./psa_split_benchmark_fixed_N.csv")

bench = pd.concat([bench_unserious])

bench = bench[(bench["N"] >= minN) & (bench["N"] <= maxN)]

plt.rcParams.update({'font.size': 12})
plt.style.use('_mpl-gallery')
plt.figure(figsize=(8, 3))

def merge_duplicates(df):
    methods = df.groupby(["flushL2", "method", "N", "splitSize", "group"]).agg(
            latency=("latency", "min"),
            var=("std_derivation", "mean"),
            )
    groups = methods.groupby(["flushL2", "group", "N", "splitSize"]).agg(
            latency=("latency", "min"),
            var=("var", "mean"),
            )
    groups.sort_index()
    groups = groups.reset_index()
    groups = groups.rename(columns={"group": "method"})

    return (methods, groups)


_,groups = merge_duplicates(bench)


def selectMethod(df, method):
    df = df[(df["method"] == method)]
    withL2 = df[~df["flushL2"]]
    flushL2 = df[df["flushL2"]]

    return (withL2, flushL2)

scalarSplitWithL2, scalarSplitFlushL2 = selectMethod(groups, "ScalarSplit")

print(scalarSplitWithL2)

scalarSplitFlushL2["throughput"] = ((scalarSplitFlushL2["N"] / (scalarSplitFlushL2["splitSize"])) / (scalarSplitFlushL2["latency"] * 1e-3)) * 1e-9;


scalarSplitFlushL2["throughput_min"] = ((scalarSplitFlushL2["N"] / (scalarSplitFlushL2["splitSize"])) / ((scalarSplitFlushL2["latency"] - scalarSplitFlushL2["var"]) * 1e-3)) * 1e-9;
scalarSplitFlushL2["throughput_max"] = ((scalarSplitFlushL2["N"] / (scalarSplitFlushL2["splitSize"])) / ((scalarSplitFlushL2["latency"] + scalarSplitFlushL2["var"]) * 1e-3)) * 1e-9;

plt.plot(scalarSplitFlushL2["splitSize"], scalarSplitFlushL2["throughput"]);

plt.fill_between(scalarSplitFlushL2["splitSize"], scalarSplitFlushL2["throughput_min"], scalarSplitFlushL2["throughput_max"],
                 alpha=0.2, color="tab:blue")


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')


# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# ax.legend(handles=handle, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Split size")
plt.ylabel("Billion Splits / Second")
plt.grid(True)


plt.tight_layout()
plt.savefig("psa_split_throughput.pdf", format="pdf")

plt.show()



