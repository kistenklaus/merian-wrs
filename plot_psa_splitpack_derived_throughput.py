import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
splitBench = pd.read_csv("./psa_split_benchmark_splitSizes.csv")
packBench = pd.read_csv("./psa_pack_benchmark_splitSizes.csv")

print(splitBench)


# Calculate K and throughput for splitBench
splitBench["K"] = np.ceil(splitBench["N"] / splitBench["splitSize"]).astype(int)
splitBench["splitThroughput"] = (splitBench["K"] / (splitBench["latency"] * 1e-3)) * 1e-9
# Calculate the standard deviation of splitThroughput
splitBench["splitThroughput_std"] = (splitBench["K"] / (splitBench["splitThroughput"]**2 * 1e6)) * splitBench["std_derivation"]

# Calculate K for packBench and ensure it's consistent
packBench["K"] = np.ceil(packBench["N"] / packBench["splitSize"]).astype(int)
packBench = packBench.rename(columns={"latency": "packLatency", "std_derivation" : "packLatency_std"})

# Perform the merge
splitPackBench = pd.merge(
        packBench[["K", "N", "flushL2", "threadsPerPack", "splitSize", "packLatency", "packLatency_std"]], 
        splitBench[["splitSize", "flushL2", "splitThroughput", "splitThroughput_std"]], 
        on=["splitSize", "flushL2"], 
        how="left")
splitPackBench["splitLatency"] = (splitPackBench["K"] / (splitPackBench["splitThroughput"] * 1e9)) * 1e3
splitPackBench["splitLatency_std"] = splitPackBench["splitLatency"] * (splitPackBench["splitThroughput_std"] / splitPackBench["splitThroughput"])

# splitPackBench["splitIps"] = (splitPackBench["splitLatency"] / splitPackBench["N"]) * 1e6
# splitPackBench["packIps"] = (splitPackBench["packLatency"] / splitPackBench["N"]) * 1e6

splitPackBench["latency"] = splitPackBench["packLatency"] + splitPackBench["splitLatency"]
splitPackBench["latency_std"] = np.sqrt(
    splitPackBench["packLatency_std"]**2 + splitPackBench["splitLatency_std"]**2
)


splitPackBench["throughput"] = splitPackBench["N"] / (splitPackBench["latency"] * 1e-3) * 1e-9

splitPackBench["throughput_std"] = (splitPackBench["N"] / (splitPackBench["throughput"]**2 * 1e6)) * splitPackBench["latency_std"]

splitPackBench["packPercentage"] = splitPackBench["packLatency"] / splitPackBench["latency"] 
splitPackBench["packPercentage_std"] = abs(splitPackBench["packPercentage"]) * np.sqrt(
    (splitPackBench["packLatency_std"] / splitPackBench["packLatency"])**2 +
    (splitPackBench["latency_std"] / splitPackBench["latency"])**2
)


def selectIpp(df, ipp):
    df = df[df["threadsPerPack"] == ipp]

    return (df[df["flushL2"]].reset_index(), df[~df["flushL2"]].reset_index())

splitPack1,_ = selectIpp(splitPackBench, 1)
splitPack2,_ = selectIpp(splitPackBench, 2)
splitPack4,_ = selectIpp(splitPackBench, 4)
splitPack8,_ = selectIpp(splitPackBench, 8)
splitPack16,_ = selectIpp(splitPackBench, 16)
splitPack32,_ = selectIpp(splitPackBench, 32)

print(splitPack1)

## Visualize

plt.rcParams.update({'font.size': 12})
plt.style.use('_mpl-gallery')
plt.figure(figsize=(8, 3))

property = "throughput"

def plotMe(splitPack, label, color):
    plt.plot(splitPack["splitSize"], splitPack[property], "-", label=label, color=color)

    plt.fill_between(splitPack["splitSize"], splitPack[property] - splitPack[f"{property}_std"],
                     splitPack[property] + splitPack[f"{property}_std"], color=color,
                     alpha=0.25)

    packBestSplit = splitPack.iloc[splitPack["throughput"].idxmax()]
    plt.plot(packBestSplit["splitSize"], packBestSplit[property], "x", color=color)

    print(label, ":best split-size=",packBestSplit["splitSize"])

plotMe(splitPack1, "1-thread", "tab:blue")
plotMe(splitPack2, "2-thread", "tab:orange")
plotMe(splitPack4, "4-thread", "tab:green")
plotMe(splitPack8, "8-thread", "tab:red")
plotMe(splitPack16, "16-thread", "tab:purple")
plotMe(splitPack32, "32-thread", "tab:brown")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Split Size")
plt.ylabel("Billion Packs / Second")
plt.grid(True)

plt.tight_layout()

plt.savefig(f'psa_splitpack_throughput_splitSizes.pdf', format="pdf")


plt.show()


