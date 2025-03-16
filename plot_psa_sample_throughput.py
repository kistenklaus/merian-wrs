import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuration
latencyBase = "sample_latency"
property = "throughput"
min_N = 1 << 16
max_N = 1 << 28
min_S = 1 << 16
max_S = 1 << 28

l2_size = 36e6;

chosen_S = 10**7  # e.g., 1 million

memory_requirements_alias_table_entry = 8

# Load evaluation results
# bench_unserious = pd.read_csv("./wrs_benchmark_sample_throughput2_psa2-0.csv")
bench_unserious = pd.read_csv("./wrs_benchmark_sample_throughput.csv")

bench = pd.concat([bench_unserious])
print(bench["method"].unique())
print(bench["S"].unique())

# Implicit columns

bench["latency"] = bench["build_latency"] + bench["sample_latency"]
bench["throughput"] = bench["S"] / (bench[latencyBase] * 1e-3) * 1e-9
bench["memory_requirements"] = memory_requirements_alias_table_entry * bench["N"]
bench["cached"] = bench["memory_requirements"] < l2_size

bench = bench[(bench["N"] >= min_N) & (bench["N"] <= max_N) 
                & (bench["S"] >= min_S) & (bench["S"] <= max_S)]

# bench = bench[bench["flushL2"]]

# --- Example: We'll try to choose a certain S. ---

# 1) Find the S in 'method' that is closest to chosen_S
unique_S = np.unique(bench['S'])
closest_S = unique_S[np.abs(unique_S - chosen_S).argmin()]

# 2) Filter the dataset to only rows having that S
subset = bench[bench['S'] == closest_S].copy()

# 3) Sort by N to produce a clean line plot
subset.sort_values(by='N', inplace=True)

# Select methods
psa2_128 = subset[subset["method"] == "PSA2-128"]
psa2_0 = subset[subset["method"] == "PSA2-0"]

# 4) Plot throughput vs. N
plt.figure(figsize=(8, 3))
plt.rcParams.update({'font.size': 12})

def plotMe(df, label,color, method):
    df = df[df["method"] == method]
    not_cached = df[df["flushL2"]]
    fully_cached = df[~df["flushL2"]]
    plt.plot(fully_cached['N'], fully_cached[property], "--", color=color)  # marker optional
    plt.plot(not_cached['N'], not_cached[property], "-",color=color, label=label)  # marker optional

plotMe(subset, "psa-baseline", "tab:blue", "PSA2-0")
plotMe(subset, "psa-sectioned", "tab:orange", "PSA2-128")
# plotMe(subset, "its-baseline", "tab:green", "ITS-0")
# plotMe(subset, "its-coop", "tab:red", "ITS-128")
# plotMe(subset, "cutpoint", "tab:brown", "Cutpoint-128")



ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Amount of Items (N)")
plt.ylabel("Billion Samples / Second")
# plt.ylim(top=60)
plt.xscale("log")
# plt.yscale("log")
plt.grid(True, axis="both", which="major")
plt.ylim(bottom=0)
# plt.grid(True, axis="y", which="minor")


plt.tight_layout()
plt.savefig("alias_sample_throughput.pdf", format="pdf")

plt.show()



