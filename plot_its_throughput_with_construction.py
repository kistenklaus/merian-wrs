import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

numPoints = 100
maxN = 1e8
minN = 1e5

# bench_unserious = pd.read_csv("./wrs_benchmark_sample_throughput_2.csv")

bench_1 = pd.read_csv("./wrs_benchmark_sample_throughput_1.csv")
bench_2 = pd.read_csv("./wrs_benchmark_sample_throughput_2.csv")
# bench_3 = pd.read_csv("./partition_scan_benchmark_3.csv")

bench = pd.concat([bench_1, bench_2])

bench = bench[(bench["N"] >= minN) & (bench["N"] <= maxN)]

bench["latency"] = (bench["build_latency"] + bench["sample_latency"]) * 1e-3;
bench["sample_throughput_with_construction"] = bench["S"] / bench["latency"]

property = "sample_throughput_with_construction"
aggregate = "max";

def merge_duplicates(df, property):
    methods = df.groupby(["flushL2", "method", "N", "group"]).agg(
            aggregate=(property, aggregate),
            )
    groups = methods.groupby(["flushL2", "group", "N"]).agg(
aggregate=("aggregate", aggregate),
            )
    groups.sort_index()
    groups = groups.reset_index()
    groups = groups.rename(columns={"group": "method"})

    return (methods, groups)

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))

_,groups = merge_duplicates(bench, property)

def selectMethod(df, method):
    df = df[(df["method"] == method)]
    withL2 = df[~df["flushL2"]]
    flushL2 = df[df["flushL2"]]

    return (withL2, flushL2)

binaryWithL2, binaryFlushL2 = selectMethod(groups, "ITS-Binary")
coopWithL2, coopFlushL2 = selectMethod(groups, "ITS-BinaryCoop")
pArrayWithL2, pArrayFlushL2 = selectMethod(groups, "ITS-pArrayCoop")
cutpointWithL2, cutpointFlushL2 = selectMethod(groups, "Cutpoint")
aliasInlineWithL2, aliasInlineFlushL2 = selectMethod(groups, "PSA-Inline")

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

binaryWithL2 = binedAverage(binaryWithL2, numPoints)
binaryFlushL2 = binedAverage(binaryFlushL2, numPoints)

coopWithL2 = binedAverage(coopWithL2, numPoints)
coopFlushL2 = binedAverage(coopFlushL2, numPoints)

cutpointWithL2 = binedAverage( cutpointWithL2, numPoints)
cutpointFlushL2 = binedAverage(cutpointFlushL2, numPoints)

aliasInlineWithL2 = binedAverage(aliasInlineWithL2, numPoints)
aliasInlineFlushL2 = binedAverage(aliasInlineFlushL2, numPoints)


binaryWithL2["aggregate"] = binaryWithL2["aggregate"] * 1e-9;
binaryFlushL2["aggregate"] = binaryFlushL2["aggregate"] * 1e-9;

coopWithL2["aggregate"] = coopWithL2["aggregate"] * 1e-9;
coopFlushL2["aggregate"] = coopFlushL2["aggregate"] * 1e-9;

pArrayWithL2["aggregate"] = pArrayWithL2["aggregate"] * 1e-9;
pArrayFlushL2["aggregate"] = pArrayFlushL2["aggregate"] * 1e-9;

cutpointWithL2["aggregate"] =  cutpointWithL2["aggregate"] * 1e-9;
cutpointFlushL2["aggregate"] = cutpointFlushL2["aggregate"] * 1e-9;

aliasInlineWithL2["aggregate"] =  aliasInlineWithL2["aggregate"] * 1e-9;
aliasInlineFlushL2["aggregate"] = aliasInlineFlushL2["aggregate"] * 1e-9;

plt.plot(binaryFlushL2["N"], binaryFlushL2["aggregate"], "-", color="tab:blue", label="baseline")
plt.plot(binaryWithL2["N"], binaryWithL2["aggregate"], ':', color="tab:blue")

plt.plot(coopFlushL2["N"], coopFlushL2["aggregate"], "-", color="tab:orange", label="coop-128")
plt.plot(coopWithL2["N"],  coopWithL2["aggregate"], ':', color="tab:orange")

plt.plot(cutpointFlushL2["N"], cutpointFlushL2["aggregate"], "-", color="tab:red", label="cutpoint-128")
plt.plot(cutpointWithL2["N"],  cutpointWithL2["aggregate"], ':', color="tab:red")


# plt.plot(pArrayFlushL2["N"], pArrayFlushL2["aggregate"], "-", color="tab:green", label="coop-pArray-128")
# plt.plot(pArrayWithL2["N"],  pArrayWithL2["aggregate"], ':', color="tab:green")

plt.plot(aliasInlineFlushL2["N"], aliasInlineFlushL2["aggregate"], "-", color="tab:green", label="alias-inline")
plt.plot(aliasInlineWithL2["N"],  aliasInlineWithL2["aggregate"], ':', color="tab:green")



# plt.plot(memcpyWithoutElemFlushL2["N"], memcpyWithoutElemFlushL2["aggregate"], "-", color="tab:red", label="memcpy")
# plt.plot(blockWiseFlushL2["N"], blockWiseFlushL2["aggregate"], "-", color="tab:orange", label="block-wise")
# plt.plot(blockWiseWithL2["N"], blockWiseWithL2["aggregate"], ":", color="tab:orange")


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Number of items")
plt.ylabel("Billion Samples / Second")
plt.ylim(top=60)
plt.xscale("log")
# plt.yscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig("its_sampling_throughput_with_construction.pdf", format="pdf")

plt.show()



