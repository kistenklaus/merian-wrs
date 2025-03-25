import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



# Look at serial or inline implementation (i.e. split-then-pack or split-packing)
serial = True

# X-scale
scale = "log"

# ========== Select x and y axis and optimization target ===========
# Possible properties 
# - N : Amount of items
# - splitSize : split size (i.e. subproblem size)
# - K : Amount of splits
# - threads : Total amount of threads
# - dispatchSize : total amount of workgroups
# - workgroupSize: workgroup size of the configuration (is probably constant).
# - threadsPerPack: Amount of threads cooperating on one subproblem (is probably constant).
# - writeElements: True, if we write the partition elements during scan-partition.
# - subgroupCount : total amount of subgroups
# - workgroupsPerSm : Average amount of workgroups per SM (i.e. SIMT-processor).
# - subgroupsPerSm : Average amount of subgroups per SM (i.e. SIMT-processor).
# - throughput : Amount of alias table entries written per second.
# - memory_requirements: Total amount of memory required
# - meanLatency : Latency of the mean calculation
# - meanLatency_std: Standard derivation of the mean calculation
# - parScanLatency : scan-partition latency 
# - parScanLatency_std : scan-partition latency standard derivation.
# - splitPackLatency : combined latency of the splitting and packing steps
# - splitPackLatency_std : Standard derivation of the combined latency of the splitting and packing steps
# - splitLatency : Latency of the split step, only available if serial=False
# - splitLatency_std : Standard derivation of the latency of the split step, only available if serial=False
# - packLatency : Latency of the pack step, only available if serial=False
# - packLatency_std : Standard derivation of the latency of the pack step, only available if serial=False

# Select the x axis
axis = "N";
# Select the y axis
property = "throughput"
# Select what to optmize for 
optimizeFor = "throughput" 
highIsGood = True # otherwise minimizes
# Select latency base for the throughput!
latencyBase = "splitPackLatency" # <- has to be a latency 

# ===== Hardware information (HAS TO BE FILLED OUT) =====
# Amount of maximum occupant threads on your device.
maxOccupantThreads = 70656
# Amount of SMs 
amountOfSm = 46
# Threads within one subgroup
subgroupSize = 32


# Filter config!
onlyCached = False # refers to if everything fits into the L2-cache not if the measurements are actually cached.
maxN = 1e10

# How many points to show (if lower then we interpolate can help with large evals to filter some noise)
numPoints = 1000


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

dirname = os.path.dirname(__file__)
bench = pd.read_csv(os.path.join(dirname, "./sweep.csv"))

print("GROUPS:", bench["group"].unique())

bench = bench[bench["N"] < maxN]


# print("SplitSizes: ", bench["splitSize"].unique())

print(bench.columns)

# print(bench)

bench["K"] = bench["N"] / bench["splitSize"]
bench["threads"] = bench["K"] * bench["threadsPerPack"]
bench["w"] = bench["threads"] / maxOccupantThreads
bench["dispatchSize"] = bench["threads"] / bench["workgroupSize"]
bench["subgroupCount"] = bench["dispatchSize"] * (bench["workgroupSize"] / subgroupSize)
bench["workgroupsPerSM"] = bench["dispatchSize"] / amountOfSm
bench["subgroupsPerSM"] = bench["subgroupCount"] / amountOfSm

bench["throughput"] = bench["N"] / (bench[latencyBase] * 1e-3) * 1e-9
bench["inv_split"] = 1 / bench["splitSize"]
bench["memory_requirements"] = (
        memory_per_item * bench["N"] + 
        memory_per_item * bench["K"] + 
        memory_requirements_constant
        )

bench["fully_cached"] = bench["memory_requirements"] < l2_size

if onlyCached:
    bench = bench[bench["fully_cached"]]


# bench = bench[bench["w"] > 1]

if highIsGood:
    optimal_idx = bench.groupby(["group", "N"])[optimizeFor].idxmax()
else:
    optimal_idx = bench.groupby(["group", "N"])[optimizeFor].idxmin()
optimal = bench.loc[optimal_idx].copy()
optimal = optimal.sort_values(by = "N")



if serial:
    sp1 = optimal[optimal["group"] == "SerialSplitPack-1"]
    sp2 = optimal[optimal["group"] == "SerialSplitPack-2"]
    sp4 = optimal[optimal["group"] == "SerialSplitPack-4"]
    sp8 = optimal[optimal["group"] == "SerialSplitPack-8"]
    sp16 = optimal[optimal["group"] == "SerialSplitPack-16"]
    sp32 = optimal[optimal["group"] == "SerialSplitPack-32"]
else:
    sp1 = optimal[optimal["group"] == "InlineSplitPack-1"]
    sp2 = optimal[optimal["group"] == "InlineSplitPack-2"]
    sp4 = optimal[optimal["group"] == "InlineSplitPack-4"]
    sp8 = optimal[optimal["group"] == "InlineSplitPack-8"]
    sp16 = optimal[optimal["group"] == "InlineSplitPack-16"]
    sp32 = optimal[optimal["group"] == "InlineSplitPack-32"]

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

sp1 = binnedAverage(sp1, numPoints)
sp2 = binnedAverage(sp2, numPoints)
sp4 = binnedAverage(sp4, numPoints)
sp8 = binnedAverage(sp8, numPoints)
sp16 = binnedAverage(sp16, numPoints)
sp32 = binnedAverage(sp32, numPoints)



plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))


def plotMe(df, label, color):
    fully_cached = df[df["fully_cached"]]
    partialy_cached = df[~df["fully_cached"]]
    if onlyCached:
        plt.plot(fully_cached[axis], fully_cached[property], "-", markersize=2,label=label, color=color)
    else:
        plt.plot(fully_cached[axis], fully_cached[property], ":", markersize=2,color=color)
        plt.plot(partialy_cached[axis], partialy_cached[property], "-", markersize=2, color=color, label=label)

if property == "threads":
    plt.hlines(y=maxOccupantThreads, xmin=optimal["N"].min(), xmax=optimal["N"].max(), linewidth=1, color='black')
    plt.hlines(y=2*maxOccupantThreads, xmin=optimal["N"].min(), xmax=optimal["N"].max(), linewidth=1, color='black')
    plt.hlines(y=3*maxOccupantThreads, xmin=optimal["N"].min(), xmax=optimal["N"].max(), linewidth=1, color='black')

plotMe(sp1, "1-invocation", "tab:blue")
plotMe(sp2, "2-invocations", "tab:orange")
plotMe(sp4, "4-invocations", "tab:green")
plotMe(sp8, "8-invocations", "tab:red")
plotMe(sp16, "16-invocations", "tab:purple")
plotMe(sp32, "32-invocations", "tab:brown")

if property == "threads":
    plt.text(x=1.2e5, y=maxOccupantThreads*1.1, s="1x Max. occupant invocations")
    plt.text(x=1.2e5, y=maxOccupantThreads + maxOccupantThreads*1.1, s="2x Max. occupant invocations")
    plt.text(x=1.2e5, y=2*maxOccupantThreads + maxOccupantThreads*1.1, s="3x Max. occupant invocations")


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')


ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, markerscale=10)
# ax.legend(frameon=False)

plt.xlabel(axis)
plt.xscale(scale)
plt.ylabel(property)
plt.grid(True)


fully_cached = optimal[optimal["fully_cached"]]
if onlyCached:
    plt.xlim((fully_cached["N"].min(), fully_cached["N"].max()))

plt.tight_layout()

plt.show()




