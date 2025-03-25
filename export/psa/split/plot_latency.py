import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

numPoints = 100
maxN = 1e10
minN = 1e7

dirname = os.path.dirname(__file__)
bench = pd.read_csv(os.path.join(dirname, "./benchmark_latency.csv"))

bench = bench[(bench["N"] >= minN) & (bench["N"] <= maxN)]
flushL2 = False
if flushL2:
    bench = bench[bench["flushL2"]]
else:
    bench = bench[~bench["flushL2"]]

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

    return df

split2 = selectMethod(groups, "ScalarSplit-2")
split4 = selectMethod(groups, "ScalarSplit-4")
split8 = selectMethod(groups, "ScalarSplit-8")
split32 = selectMethod(groups, "ScalarSplit-32")
split128 = selectMethod(groups, "ScalarSplit-128")
split1024 = selectMethod(groups, "ScalarSplit-1024")

def binedAverage(df, B):
    if (len(df) <= B):
        return df
    bin_size = len(df) // B
    df_copy = df.copy()
    df_copy.loc[:, 'bin'] = np.arange(len(df_copy)) // bin_size
    df_copy.loc[:, 'bin'] = np.where(df_copy['bin'] >= B, B-1, df_copy['bin'])
    results = df_copy.groupby('bin').agg({
        'N': 'median',
        'latency': 'mean'
    }).reset_index(drop=True)
    return results

split2 = binedAverage(split2, numPoints)
split4 = binedAverage(split4, numPoints)
split8 = binedAverage(split8, numPoints)
split32 = binedAverage(split32, numPoints)
split128 = binedAverage(split128, numPoints)
split1024 = binedAverage(split1024, numPoints)

print(split2)
print(split4)

split4["speedup"] = (split2["latency"] / split4["latency"]) / 1
# split8["speedup"] = (split2["latency"] / split8["latency"]) / 4;
# split32["speedup"] = (split2["latency"] / split32["latency"]) / 16;
# split128["speedup"] = (split2["latency"] / split128["latency"]) / 64;
split1024["speedup"] = (split2["latency"] / split1024["latency"]) / 128;

# plt.plot(split4["N"], split4["speedup"], label="split-4")
# plt.plot(split8["N"], split8["speedup"], label="split-8")
# plt.plot(split32["N"], split32["speedup"], label="split-32")
# plt.plot(split128["N"], split128["speedup"], label="split-128")
# plt.plot(split1024["N"], split1024["speedup"], label="split-1024")

plt.plot(split2["N"], split2["latency"], label="split-2");
plt.plot(split4["N"], split4["latency"], label="split-4");
plt.plot(split8["N"], split8["latency"], label="split-8");
plt.plot(split32["N"], split32["latency"], label="split-32");
plt.plot(split128["N"], split128["latency"], label="split-128");
plt.plot(split1024["N"], split1024["latency"], label="split-1024");

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



