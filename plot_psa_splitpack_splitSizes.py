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

bench_unserious = pd.read_csv(f'./psa_splitpack_benchmark_splitSizes.csv')

# bench4 = pd.read_csv('./psa_splitpack_benchmark_splitSizes4.csv') # it = 250 w = 4

# bench = pd.concat([bench4, bench3]) # w = 10
# bench = pd.concat([bench1,bench2,bench3,bench4, bench5, bench6, bench7]) # w = 5
bench = pd.concat([bench_unserious])

print(bench)

# bench = bench[(bench["splitSize"] >= ) & (bench["splitSize"] <= maxN)]



bench["packLatency"] = bench["latency"]
bench["packStd"] = bench["std_derivation"]
bench["latency"] = bench["latency"] + bench["splitLatency"]

bench["throughput"] = (bench["N"] / (bench["latency"] * 1e-3)) * 1e-9;
bench["throughput_std"] = (bench["N"] / (bench["latency"]**2 * 1e6)) * bench["std_derivation"]

bench["splitTpp"] = ((bench["splitLatency"] * 1e-3) / bench["N"]) * 1e9
bench["splitTpp_std"] = (1e6 / bench["N"]) * bench["splitStd"]

bench["packTpp"] = ((bench["packLatency"] * 1e-3) / bench["N"]) * 1e9
bench["packTpp_std"] = (1e6 / bench["N"]) * bench["packStd"]

plt.rcParams.update({'font.size': 12})
plt.style.use('_mpl-gallery')
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
    methods = df.groupby(["flushL2", "method", "splitSize", "threadsPerPack","N", "workgroupSize", "group"]).apply(
        lambda g: pd.Series({
            "count": g["latency"].count(),
            "splitTpp": weighted_mean(g["splitTpp"], g["splitTpp_std"]**-2),  # Weighted by precision (1/variance)
            "splitTpp_var_sum": weighted_variance(g["splitTpp"], g["splitTpp_std"], g["splitTpp_std"]**-2),  # Weighted variance
            "packTpp": weighted_mean(g["packTpp"], g["packTpp_std"]**-2),  # Weighted by precision (1/variance)
            "packTpp_var_sum": weighted_variance(g["packTpp"], g["packTpp_std"], g["packTpp_std"]**-2),  # Weighted variance
            "throughput":  weighted_mean(g["throughput"], g["throughput_std"]**-2),
            "throughput_var_sum": weighted_variance(g["throughput"], g["throughput_std"], g["throughput_std"]**-2),
            "packLatency":  weighted_mean(g["packLatency"], g["packStd"]**-2),
            "pack_var_sum": weighted_variance(g["packLatency"], g["packStd"], g["packStd"]**-2),
            "splitLatency":  weighted_mean(g["splitLatency"], g["splitStd"]**-2),
            "split_var_sum": weighted_variance(g["splitLatency"], g["splitStd"], g["splitStd"]**-2),
        })
    )

    # Compute standard deviation from variance
    methods["splitTpp_std"] = np.sqrt(methods["splitTpp_var_sum"])
    methods["packTpp_std"] = np.sqrt(methods["packTpp_var_sum"])
    methods["throughput_std"] = np.sqrt(methods["throughput_var_sum"])
    methods["packLatency_std"] = np.sqrt(methods["pack_var_sum"])
    methods["splitLatency_std"] = np.sqrt(methods["split_var_sum"])

    # Second aggregation: compute weighted mean and variance for groups
    groups = methods.groupby(["flushL2", "group", "splitSize", "N", "threadsPerPack", "workgroupSize"]).apply(
        lambda g: pd.Series({
            "count": g["count"].sum(),
            "splitTpp": weighted_mean(g["splitTpp"], g["splitTpp_std"]**-2),  # Weighted by precision (1/variance)
            "splitTpp_var_sum": weighted_variance(g["splitTpp"], g["splitTpp_std"], g["splitTpp_std"]**-2),  # Weighted variance
            "packTpp": weighted_mean(g["packTpp"], g["packTpp_std"]**-2),  # Weighted by precision (1/variance)
            "packTpp_var_sum": weighted_variance(g["packTpp"], g["packTpp_std"], g["packTpp_std"]**-2),  # Weighted variance
            "throughput":  weighted_mean(g["throughput"], g["throughput_std"]**-2),
            "throughput_var_sum": weighted_variance(g["throughput"], g["throughput_std"], g["throughput_std"]**-2),
            "packLatency":  weighted_mean(g["packLatency"], g["packLatency_std"]**-2),
            "pack_var_sum": weighted_variance(g["packLatency"], g["packLatency_std"], g["packLatency_std"]**-2),
            "splitLatency":  weighted_mean(g["splitLatency"], g["splitLatency_std"]**-2),
            "split_var_sum": weighted_variance(g["splitLatency"], g["splitLatency_std"], g["splitLatency_std"]**-2),
        })
    )

    # Compute standard deviation from variance
    groups["splitTpp_std"] = np.sqrt(groups["splitTpp_var_sum"])
    groups["packTpp_std"] = np.sqrt(groups["packTpp_var_sum"])
    groups["throughput_std"] = np.sqrt(groups["throughput_var_sum"])
    groups["packLatency_std"] = np.sqrt(groups["pack_var_sum"])
    groups["splitLatency_std"] = np.sqrt(groups["split_var_sum"])
    

    # Sort and reset index
    groups = groups.reset_index()
    groups = groups.rename(columns={"group": "method"})

    return methods, groups


_,groups = merge_duplicates2(bench)

groups["tpp"] = groups["packTpp"] + groups["splitTpp"];
groups["throughput"] = 1.0 / groups["tpp"]
groups["latency"] = groups["splitLatency"] + groups["packLatency"]
groups["packPercentage"] = groups["packLatency"] / groups["latency"]

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

print(pack8)

# print(pack8Flush[["N", "splitSize", "latency", "std_derivation", "w"]].to_string())

property = "throughput"
axis = "splitSize"
stackMe = False
# axis = "workgroupsPerSM"
# axis = "dispatchSize"

def plotMe(pack, packFlush, label, color):
    if (len(packFlush) == 0):
        return
    if stackMe:
        labels = ["Splitting", "Packing"]
        plt.stackplot(packFlush[axis], packFlush["splitTpp"], packFlush["packTpp"],
                      labels=labels)
        # plt.stackplot(packFlush[axis], packFlush["latency
        pass
    else:
        # plt.plot(pack[axis], pack[property], "--", color=color);
        plt.plot(packFlush[axis], packFlush[property], "-", label=label, color=color);
        if property == "throughput":
            plt.fill_between(packFlush[axis], packFlush["throughput"] - packFlush["throughput_std"],
                             packFlush["throughput"] + packFlush["throughput_std"],
                             color=color, alpha=0.2)
        # if property == "latency":
        #     plt.fill_between(packFlush[axis], packFlush["latency"] - packFlush["std_derivation"],
        #                      packFlush["latency"] + packFlush["std_derivation"],
        #                      color=color, alpha=0.2)

        packBestSplit = packFlush.iloc[packFlush["throughput"].idxmax()]
        plt.scatter(packBestSplit[axis], packBestSplit[property], s=50, marker="x", color=color)

        print(label, ":best split-size=",packBestSplit["splitSize"])


if not stackMe:
    plotMe(pack1, pack1Flush, "1-thread", "tab:blue")
    plotMe(pack2, pack2Flush, "2-thread", "tab:orange")
    plotMe(pack4, pack4Flush, "4-thread", "tab:green")
    plotMe(pack8, pack8Flush, "8-thread", "tab:red")
    plotMe(pack16, pack16Flush, "16-thread", "tab:purple")
    plotMe(pack32, pack32Flush, "32-thread", "tab:brown")
else:
    plotMe(pack16, pack16Flush, "16-thread", "tab:purple")



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
plt.ylabel("Billion packs / Second")
# plt.yscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig(f'psa_splitpack_splitSizes.pdf', format="pdf")

plt.show()



