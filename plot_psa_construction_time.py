import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


numPoints = 1000
serial = True
scale = "linear"
axis = "N";
threadsPerPack = 1

optimizeFor = "splitPackLatency"

maxOccupantThreads = 70656

onlyCached = False

l2_size = 36e6 # 36MB
memory_per_item_partition_indices = 4
memory_per_item_partition_prefix = 4
memory_per_item_alias_table = 8
memory_per_item_weights = 4
memory_per_split_entry = 4 * 3

memory_per_item = (
      memory_per_item_partition_indices 
    + memory_per_item_partition_prefix 
    + memory_per_item_alias_table
    + memory_per_item_weights
    )
if serial:
    memory_per_split = memory_per_split_entry
else:
    memory_per_split = 0

memory_requirements_heavyCount = 4
memory_requirements_mean = 4
memory_requirements_constant = (
          memory_requirements_heavyCount
        + memory_requirements_mean
        )



# bench_unserious = pd.read_csv("./psa_inline_splitpack_benchmark_optimal.csv")

# bench1 = pd.read_csv("./psa_inline_splitpack_benchmark_optimal1.csv")
benche5e6_s2s128 = pd.read_csv("./psa_inline_splitpack_benchmark_optimal_e5_e6_s2_s128.csv")
benche6e7_s2s128 = pd.read_csv("./psa_inline_splitpack_benchmark_optimal_e6e7_s2s128.csv")
benche5e7_s2s128_serial = pd.read_csv("./psa_inline_splitpack_benchmark_optimal_serial_log_e5e7_s2s128.csv")
benche5e7_s2s128_inline = pd.read_csv("./psa_inline_splitpack_benchmark_optimal_inline_log_e5e7_s2s128.csv")
benche5e7_s128s256 = pd.read_csv("psa_inline_splitpack_benchmark_inline_log_e5e6_s128s256.csv")

# bench = pd.concat([benche5e6_s2s128, benche6e7_s2s128])
bench = pd.concat([benche5e7_s2s128_inline, benche5e7_s2s128_serial,
                   benche5e7_s128s256])


# print("SplitSizes: ", bench["splitSize"].unique())

print(bench.columns)

# print(bench)

bench["K"] = bench["N"] / bench["splitSize"]
bench["threads"] = bench["K"] * bench["threadsPerPack"]
bench["w"] = bench["threads"] / maxOccupantThreads
bench["dispatchSize"] = bench["threads"] / bench["workgroupSize"]
bench["subgroupCount"] = bench["dispatchSize"] * (bench["workgroupSize"] / 32)
bench["workgroupsPerSM"] = bench["dispatchSize"] / 46
bench["subgroupsPerSM"] = bench["subgroupCount"] / 46

bench["throughput"] = bench["N"] / (bench[optimizeFor] * 1e-3) * 1e-9
bench["inv_split"] = 1 / bench["splitSize"]
bench["memory_requirements"] = (
        memory_per_item * bench["N"] + 
        memory_per_item * bench["K"] + 
        memory_requirements_constant
        )

bench["fully_cached"] = bench["memory_requirements"] < l2_size

if onlyCached:
    bench = bench[bench["fully_cached"]]




bench = bench[(bench["N"] > 1e6) & (bench["N"] < 1e7)]


optimal_idx = bench.groupby(["N"])[optimizeFor].idxmin()
optimal = bench.loc[optimal_idx].copy()
optimal = optimal.sort_values(by = "N")



# if serial:
#     sp = optimal[optimal["group"] == f"SerialSplitPack-{threadsPerPack}"]
# else:
#     sp = optimal[optimal["group"] == f"InlineSplitPack-{threadsPerPack}"]

def binnedAverage(df, numPoints):
    # If we don't have enough data to bin, just return the original df
    if len(df) <= numPoints:
        return df.copy()
    
    df_copy = df.copy()
    
    # For safety, ensure no NaN or non-positive if using log scale
    # (only if your data might contain such values)
    if scale == 'log' and (df_copy[axis] <= 0).any():
        raise ValueError("Log scale binning requires strictly positive values in '{}'."
                         .format(axis))

    # Compute min and max of your axis values
    x_min = df_copy[axis].min()
    x_max = df_copy[axis].max()

    # If the data range is degenerate, just return df
    if x_min == x_max:
        return df.copy()

    # Build bin edges
    if scale == 'linear':
        # equally spaced bins in linear space
        bin_edges = np.linspace(x_min, x_max, numPoints+1)
    elif scale == 'log':
        # equally spaced bins in log space
        bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), numPoints+1)
    else:
        raise ValueError("scale must be either 'linear' or 'log'")

    # Assign each row to a bin (labels=False returns integer bin indices)
    df_copy['bin'] = pd.cut(df_copy[axis], bins=bin_edges, labels=False, include_lowest=True)
    df_copy = df_copy.reset_index();

    min_indices = df_copy.groupby('bin')[optimizeFor].idxmin()

    return df_copy.loc[min_indices].reset_index(drop=True)
    
    # Group by the assigned bin and take the median of the relevant columns
    # results = df_copy.groupby('bin', 'fully_cached').agg({
    #     axis: 'median',
    #     property: 'median'
    # }).reset_index(drop=True)
    #
    # return results

sp = optimal
sp = binnedAverage(sp, numPoints)



plt.rcParams.update({'font.size': 12})

plt.style.use('_mpl-gallery')
plt.figure(figsize=(8, 3))


def plotMe(df):
    labels = ["mean", "partition-scan", "split-packing"]
    plt.stackplot(sp["N"], sp["meanLatency"], sp["parScanLatency"], sp["splitPackLatency"], labels=labels)
    pass

plotMe(sp)


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
# ax.legend(frameon=False)

plt.xlabel("Amount of Items (N)")
plt.xscale(scale)
plt.ylabel("Latency (ms)")
plt.grid(True)
ax.set_xticks([1e6, 1e7])
ax.set_xticklabels([r"$10^6$", r"$10^7$"])


plt.tight_layout()
plt.savefig(f'psa_construction_time.pdf', format="pdf")

plt.show()




