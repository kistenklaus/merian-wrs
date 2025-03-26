from re import A
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

numPoints = 10000
maxN = 1e8
minN = 1e5

dirname = os.path.dirname(__file__)
bench = pd.read_csv(os.path.join(dirname, "./benchmark2.csv"))

bench = bench[(bench["N"] >= minN) & (bench["N"] <= maxN)]

property = "throughput"
aggregate = "median";
latencyBase = "latency" # <- "latency" if you also want to consider the construction time as part of sampling.

print(bench)

bench["latency"] = bench["build_latency"] + bench["sample_latency"]
bench["throughput"] = bench["S"] / (bench[latencyBase] * 1e-3) * 1e-9

def merge_duplicates(df, property):
    methods = df.groupby(["flushL2", "method", "S", "group"]).agg(
            aggregate=(property, aggregate),
            )
    groups = methods.groupby(["flushL2", "group", "S"]).agg(
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

print(groups["method"].unique())

binaryWithL2, binaryFlushL2 = selectMethod(groups, "ITS-0")
coopWithL2, coopFlushL2 = selectMethod(groups, "ITS-128")
cutpointWithL2, cutpointFlushL2 = selectMethod(groups, "Cutpoint-128")
alias0, alias0FlushL2 = selectMethod(groups, "PSA2-0")
alias128, alias128FlushL2 = selectMethod(groups, "PSA2-128")


def binedAverage(df, B):
    if (len(df) <= B):
        return df.reset_index(drop=True)
    bin_size = len(df) // B
    df_copy = df.copy()
    df_copy.loc[:, 'bin'] = np.arange(len(df_copy)) // bin_size
    df_copy.loc[:, 'bin'] = np.where(df_copy['bin'] >= B, B-1, df_copy['bin'])
    results = df_copy.groupby('bin').agg({
        'S': 'median',
        'aggregate': 'mean'
    }).reset_index(drop=True)
    return results

binaryWithL2 = binedAverage(binaryWithL2, numPoints)
binaryFlushL2 = binedAverage(binaryFlushL2, numPoints)

coopWithL2 = binedAverage(coopWithL2, numPoints)
coopFlushL2 = binedAverage(coopFlushL2, numPoints)

cutpointWithL2 = binedAverage( cutpointWithL2, numPoints)
cutpointFlushL2 = binedAverage(cutpointFlushL2, numPoints)

alias0 = binedAverage(alias0, numPoints)
alias0FlushL2 = binedAverage(alias0FlushL2, numPoints)

alias128 = binedAverage(alias128, numPoints)
alias128FlushL2 = binedAverage(alias128FlushL2, numPoints)



plt.plot(binaryFlushL2["S"], binaryFlushL2["aggregate"], "-", color="tab:blue", label="baseline")
plt.plot(binaryWithL2["S"], binaryWithL2["aggregate"], ':', color="tab:blue")

plt.plot(coopFlushL2["S"], coopFlushL2["aggregate"], "-", color="tab:orange", label="coop-128")
plt.plot(coopWithL2["S"],  coopWithL2["aggregate"], ':', color="tab:orange")

plt.plot(cutpointFlushL2["S"], cutpointFlushL2["aggregate"], "-", color="tab:red", label="cutpoint-128")
plt.plot(cutpointWithL2["S"],  cutpointWithL2["aggregate"], ':', color="tab:red")

plt.plot(alias0FlushL2["S"], alias0FlushL2["aggregate"], "-", color="tab:green", label="baseline-alias")
plt.plot(alias0["S"],  alias0["aggregate"], ':', color="tab:green")

plt.plot(alias128FlushL2["S"], alias128FlushL2["aggregate"], "-", color="tab:brown", label="section-sampling-128")
plt.plot(alias128["S"],  alias128["aggregate"], ':', color="tab:brown")


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
# plt.ylim(top=60)
plt.xscale("log")
# plt.yscale("log")
plt.grid(True)


plt.tight_layout()

plt.show()



