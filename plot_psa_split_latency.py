import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

numPoints = 100
maxN = 1e10
minN = 1e7

bench_unserious = pd.read_csv("./psa_split_benchmark_latency.csv")

bench = pd.concat([bench_unserious])

bench = bench[(bench["N"] >= minN) & (bench["N"] <= maxN)]

plt.rcParams.update({'font.size': 12})
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

_, split2 = selectMethod(groups, "ScalarSplit-2")
_, split4 = selectMethod(groups, "ScalarSplit-4")
_, split8 = selectMethod(groups, "ScalarSplit-8")
_, split32 = selectMethod(groups, "ScalarSplit-32")
_, split128 = selectMethod(groups, "ScalarSplit-128")
_, split1024 = selectMethod(groups, "ScalarSplit-1024")
print(split2)
print(split4)

split4["a"] = split2["latency"];
print(split4)

plt.plot(split4["N"], split4["speedup"])

# plt.plot(split2["N"], split2["latency"], label="split-2");
# plt.plot(split4["N"], split4["latency"], label="split-4");
# plt.plot(split8["N"], split8["latency"], label="split-8");
# plt.plot(split32["N"], split32["latency"], label="split-32");
# plt.plot(split128["N"], split128["latency"], label="split-128");
# plt.plot(split1024["N"], split1024["latency"], label="split-1024");

# plt.fill_between(scalarSplitFlushL2["splitSize"], scalarSplitFlushL2["throughput_min"], scalarSplitFlushL2["throughput_max"],
#                  alpha=0.2, color="tab:blue")


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')


# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Amount of items")
plt.ylabel("Latency (ms)")
# plt.xscale("log")
# plt.yscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig("psa_split_latency.pdf", format="pdf")

plt.show()



