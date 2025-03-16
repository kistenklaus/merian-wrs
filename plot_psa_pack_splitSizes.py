import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

numPoints = 100
maxN = 1e10
minN = 1e5

maxSplitSize = 2
minSplitSize = 128

smCount = 46;
maxOccupantThreads = 48 * 46 * 32

bench_unserious = pd.read_csv(f'./psa_pack_benchmark_splitSizes.csv')

bench1 = pd.read_csv('./psa_pack_benchmark_splitSizes1.csv')  # it = 250 w = 1
bench2 = pd.read_csv('./psa_pack_benchmark_splitSizes2.csv') # it = 250 w = 2
bench3 = pd.read_csv('./psa_pack_benchmark_splitSizes3.csv') # it = 250 w = 2
bench4 = pd.read_csv('./psa_pack_benchmark_splitSizes4.csv') # it = 250 w = 2
bench5 = pd.read_csv('./psa_pack_benchmark_splitSizes5.csv') # it = 250 w = 2
bench6 = pd.read_csv('./psa_pack_benchmark_splitSizes6.csv') # it = 250 w = 2
bench7 = pd.read_csv('./psa_pack_benchmark_splitSizes7.csv') # it = 250 w = 2

# bench = pd.concat([bench4, bench3]) # w = 10
# bench = pd.concat([bench1,bench2,bench3,bench4, bench5, bench6, bench7]) # w = 5
bench = pd.concat([bench_unserious])

# bench = bench[bench["N"] < 1.8e6]

# bench = bench[(bench["splitSize"] >= ) & (bench["splitSize"] <= maxN)]

bench["throughput"] = (bench["N"] / (bench["latency"] * 1e-3)) * 1e-9;
bench["throughput_std"] = (bench["N"] / (bench["latency"]**2 * 1e6)) * bench["std_derivation"]

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))

def weighted_mean(series, weights):
    """Compute the weighted mean."""
    return np.sum(series * weights) / np.sum(weights)

def weighted_variance(mean_series, std_series, weights):
    """Compute the weighted variance (corrected for different means)."""
    mean_w = weighted_mean(mean_series, weights)
    return np.sum(weights * (std_series**2 + mean_series**2)) / np.sum(weights) - mean_w**2

def merge_duplicates2(df):
    # First aggregation: compute weighted mean and variance for each method
    methods = df.groupby(["flushL2", "method", "splitSize", "threadsPerPack", "N", "workgroupSize", "group"]).apply(
        lambda g: pd.Series({
            "latency": weighted_mean(g["latency"], g["std_derivation"]**-2),  # Weighted by precision (1/variance)
            "var_sum": weighted_variance(g["latency"], g["std_derivation"], g["std_derivation"]**-2),  # Weighted variance
            "count": g["latency"].count(),
            "throughput":  weighted_mean(g["throughput"], g["throughput_std"]**-2),
            "throughput_var_sum": weighted_variance(g["throughput"], g["throughput_std"], g["throughput_std"]**-2),
        })
    )

    # Compute standard deviation from variance
    methods["std_derivation"] = np.sqrt(methods["var_sum"])
    methods["throughput_std"] = np.sqrt(methods["throughput_var_sum"])

    # Second aggregation: compute weighted mean and variance for groups
    groups = methods.groupby(["flushL2", "group", "splitSize", "threadsPerPack", "N", "workgroupSize"]).apply(
        lambda g: pd.Series({
            "latency": weighted_mean(g["latency"], g["std_derivation"]**-2),  # Weighted by precision
            "var_sum": weighted_variance(g["latency"], g["std_derivation"], g["std_derivation"]**-2),  # Weighted variance
            "count": g["count"].sum(),
            "throughput":  weighted_mean(g["throughput"], g["throughput_std"]**-2),
            "throughput_var_sum": weighted_variance(g["throughput"], g["throughput_std"], g["throughput_std"]**-2),
        })
    )

    # Compute standard deviation from variance
    groups["std_derivation"] = np.sqrt(groups["var_sum"])
    groups["throughput_std"] = np.sqrt(groups["throughput_var_sum"])

    # Sort and reset index
    groups = groups.reset_index()
    groups = groups.rename(columns={"group": "method"})

    return methods, groups


def merge_duplicates(df):
    methods = df.groupby(["flushL2", "method", "splitSize", "workgroupSize", "group"]).agg(
            latency=("latency", "mean"),
            count=("std_derivation", "count"),  # Get count of elements per group
            var_sum=("std_derivation", lambda x: (x**2).sum())  # Sum of squared std deviations
            )
    methods["var"] = methods["var_sum"] / (methods["count"]**2)
    methods["std_derivation"] = np.sqrt(methods["var"])
    groups = methods.groupby(["flushL2", "group", "workgroupSize", "splitSize"]).agg(
            latency=("latency", "mean"),
            count=("std_derivation", "count"),  # Get count of elements per group
            var_sum=("std_derivation", lambda x: (x**2).sum())  # Sum of squared std deviations
            )
    groups["var"] = groups["var_sum"] / (groups["count"]**2)
    groups["std_derivation"] = np.sqrt(groups["var"])
    groups.sort_index()
    groups = groups.reset_index()
    groups = groups.rename(columns={"group": "method"})

    return (methods, groups)


_,groups = merge_duplicates2(bench)

# groups = groups[groups["std_derivation"] < 0.1]


# groups["threads"] = (groups["N"] / groups["splitSize"]) * groups["threadsPerPack"];
# groups["dispatchSize"] = np.ceil(groups["threads"] / groups["workgroupSize"])
# groups["workgroupsPerSM"] = groups["dispatchSize"] / smCount
# groups["w"] = groups["threads"] / maxOccupantThreads;

# groups = groups[groups["std_derivation"] < 0.1]


def selectMethod(df, method):
    df = df[(df["method"] == method)]
    withL2 = df[~df["flushL2"]]
    flushL2 = df[df["flushL2"]]
    withL2 : pd.DataFrame
    flushL2 : pd.DataFrame

    return (flushL2.reset_index(), withL2.reset_index())

pack1Flush, pack1 = selectMethod(groups, "Pack-1")
pack2Flush, pack2 = selectMethod(groups, "Pack-2")
pack4Flush, pack4 = selectMethod(groups, "Pack-4")
pack8Flush, pack8 = selectMethod(groups, "Pack-8")
pack16Flush, pack16 = selectMethod(groups, "Pack-16")
pack32Flush, pack32 = selectMethod(groups, "Pack-32")

# a bit evil but it makes no sence if the measurements with L2 perform worse than when flushing the L2 
# It has to be noise (the patterns we see here are also not repeatable)
Cheat = False
if Cheat:
    pack1["throughput"] = np.maximum(pack1Flush["throughput"], pack1["throughput"])
    pack2["throughput"] = np.maximum(pack2Flush["throughput"], pack2["throughput"])
    pack4["throughput"] = np.maximum(pack4Flush["throughput"], pack4["throughput"])
    pack8["throughput"] = np.maximum(pack8Flush["throughput"], pack8["throughput"])
    pack16["throughput"] = np.maximum(pack16Flush["throughput"], pack16["throughput"])
    pack32["throughput"] = np.maximum(pack32Flush["throughput"], pack32["throughput"])

# print(pack8Flush[["N", "splitSize", "latency", "std_derivation", "w"]].to_string())

property = "throughput"
axis = "splitSize"
# axis = "workgroupsPerSM"
# axis = "dispatchSize"

def plotMe(pack, packFlush, label, color):
    # plt.plot(pack[axis], pack[property], "--", color=color);

    packLow = packFlush[packFlush["N"] <= 1.8e6]
    packHigh = packFlush[packFlush["N"] >= 1.5e6]

    plt.plot(packLow[axis], packLow[property], "--", color=color);
    plt.plot(packHigh[axis], packHigh[property], "-", label=label, color=color);



    if property == "throughput":
        plt.fill_between(packFlush[axis], packFlush["throughput"] - packFlush["throughput_std"],
                         packFlush["throughput"] + packFlush["throughput_std"],
                         color=color, alpha=0.2)
    if property == "latency":
        plt.fill_between(packFlush[axis], packFlush["latency"] - packFlush["std_derivation"],
                         packFlush["latency"] + packFlush["std_derivation"],
                         color=color, alpha=0.2)

    packBestSplit = packFlush.iloc[packFlush["throughput"].idxmax()]
    plt.plot(packBestSplit[axis], packBestSplit[property], "x", color=color)

    print(label, ":best split-size=",packBestSplit["splitSize"], "N=", packBestSplit["N"])

plotMe(pack1, pack1Flush, "1-invocation", "tab:blue")
plotMe(pack2, pack2Flush, "2-invocations", "tab:orange")
plotMe(pack4, pack4Flush, "4-invocations", "tab:green")
plotMe(pack8, pack8Flush, "8-invocations", "tab:red")
plotMe(pack16, pack16Flush, "16-invocations", "tab:purple")
plotMe(pack32, pack32Flush, "32-invocations", "tab:brown")


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')


ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
# ax.legend(frameon=False)

plt.xlabel("Subproblem size")
plt.ylabel("Billion Packs / Second")
plt.grid(True)


plt.tight_layout()
plt.savefig(f'psa_pack_throughput_splitSizes.pdf', format="pdf")

plt.show()



