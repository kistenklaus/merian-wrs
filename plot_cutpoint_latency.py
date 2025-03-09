import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

numPoints = 100
maxN = 1e10
minN = 1e5

bench_unserious = pd.read_csv("./cutpoint_benchmark_latency.csv")
# bench_3 = pd.read_csv("./partition_scan_benchmark_3.csv")

bench = pd.concat([bench_unserious])

bench = bench[(bench["N"] >= minN) & (bench["N"] <= maxN)]

plt.rcParams.update({'font.size': 12})
plt.style.use('_mpl-gallery')
plt.figure(figsize=(8, 3))

def merge_duplicates(df):
    methods = df.groupby(["flushL2", "method", "N", "group"]).agg(
            scan_latency=("scan_latency", "max"),
            guide_latency=("guide_latency", "max"),
            sample_latency=("sample_latency", "max"),
            )
    groups = methods.groupby(["flushL2", "group", "N"]).agg(
            scan_latency=("scan_latency", "max"),
            guide_latency=("guide_latency", "max"),
            sample_latency=("sample_latency", "max"),
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

cutpointWithL2, cutpointFlushL2 = selectMethod(groups, "Cutpoint")

# print(cutpointFlushL2)

def stacky(cutpoint):
    cutpoint["step1"] = cutpoint["scan_latency"]
    cutpoint["step2"] = cutpoint["guide_latency"]
    cutpoint["step3"] = cutpoint["sample_latency"]
    return cutpoint;

cutpointWithL2 = stacky(cutpointWithL2)
cutpointFlushL2 = stacky(cutpointFlushL2)

def binedAverage(df, B):
    if (len(df) <= B):
        return df
    bin_size = len(df) // B
    df_copy = df.copy()
    df_copy.loc[:, 'bin'] = np.arange(len(df_copy)) // bin_size
    df_copy.loc[:, 'bin'] = np.where(df_copy['bin'] >= B, B-1, df_copy['bin'])
    results = df_copy.groupby('bin').agg({
        'N': 'median',
        'step1' : 'mean',
        'step2': 'mean',
        'step3' : 'mean',
    }).reset_index(drop=True)
    return results


cutpointWithL2 = binedAverage( cutpointWithL2, numPoints)
cutpointFlushL2 = binedAverage(cutpointFlushL2, numPoints)

labels = ["scan", "guiding table", "sampling"]
# plt.stackplot(cutpointFlushL2["N"], cutpointFlushL2["step1"], cutpointFlushL2["step2"],
#               cutpointFlushL2["step3"], labels=labels)
handle = plt.stackplot(cutpointWithL2["N"], cutpointWithL2["step1"], cutpointWithL2["step2"],
              cutpointWithL2["step3"], labels=labels)


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

# ax.legend(handles=handle, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Number of items")
plt.ylabel("Latency (ms)")
# plt.ylim(top=60)
plt.xscale("log")
# plt.yscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig("cutpoint_stackplot.pdf", format="pdf")

plt.show()



