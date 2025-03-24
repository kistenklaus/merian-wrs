from re import A
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

numPoints = 10000
N = 1e7

bench0 = pd.read_csv("./wrs_benchmark_sample_throughput2_its_cutpoint.csv")
bench1 = pd.read_csv("./wrs_benchmark_sample_throughput2_psa2-0.csv")
bench2 = pd.read_csv("./wrs_benchmark_sample_throughput2_psa2-128.csv")

bench3 = pd.read_csv("./wrs_benchmark_sample_throughput2.csv")

bench = pd.concat([bench1, bench2, bench0])

print(bench["N"].unique())
bench = bench[bench["N"] == N]

print(bench["group"].unique())

property = "sample_throughput"
aggregate = "median";
latencyBase = "sample_latency"

bench["latency"] = bench["build_latency"] + bench["sample_latency"]
bench["sample_throughput"] = bench["S"] / (bench[latencyBase] * 1e-3) * 1e-9


def binedAverage(df, B):
    if (len(df) <= B):
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

def plotMe(df, method, color, label):
    df = df[df["group"] == method]
    withL2 = df[~df["flushL2"]]
    flushL2 = df[df["flushL2"]]

    withL2 = binedAverage(withL2, numPoints)
    flushL2 = binedAverage(flushL2, numPoints)

    # plt.plot(withL2["S"], withL2[property], ":", color=color)
    plt.plot(flushL2["S"], flushL2[property], "-", label=label, color=color)
    
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))

print(bench["group"].unique())
plotMe(bench, "ITS-0", "tab:blue", "its-baseline")
plotMe(bench, "ITS-128", "tab:orange", "its-coop")
plotMe(bench, "Cutpoint", "tab:green", "cutpoint")
plotMe(bench, "PSA2-0", "tab:red", "psa-baseline")
plotMe(bench, "PSA2-128", "tab:brown", "psa-sectioned")


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Amount of Samples")
plt.ylabel("Billion Samples / Second")
# plt.ylim(top=60)
plt.xscale("log")
# plt.yscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig("wrs_sample_throughput2.pdf", format="pdf")

plt.show()



