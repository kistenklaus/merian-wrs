import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

numPoints = 100
maxN = 1e10
minN = 1e5

maxOccupantThreads = 48 * 46 * 32

bench_unserious = pd.read_csv("./psa_pack_benchmark_latency.csv")

# bench_1 = pd.read_csv("./psa_pack_benchmark_latency_1.csv")
# bench_2 = pd.read_csv("./psa_pack_benchmark_latency_2.csv")
# bench_3 = pd.read_csv("./psa_pack_benchmark_latency_3.csv")
bench_hugeW = pd.read_csv("./psa_pack_benchmark_latency_hugeW.csv")

bench = pd.concat([bench_unserious])

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))

#
def merge_duplicates(df):
    methods = df.groupby(["flushL2", "method", "group", "invocPerPack", "splitSize", "w", "N"]).agg(
            latency=("latency", "median"),
            var=("std_derivation", "mean"),
            )
    groups = methods.groupby(["flushL2", "group", "invocPerPack", "splitSize", "w", "N"]).agg(
            latency=("latency", "min"),
            var=("var", "mean"),
            )
    groups.sort_index()
    groups = groups.reset_index()
    groups = groups.rename(columns={"group": "method"})

    return (methods, groups)


# _,groups = merge_duplicates(bench)
groups = bench
groups["method"] = groups["group"]
print(groups)

groups["throughput"] = (groups["N"] / (groups["latency"] * 1e-3)) * 1e-9

groups["K"] = groups["N"] / groups["splitSize"];
groups["threads"] = groups["K"] * groups["invocPerPack"]
groups["dispatchSize"] = groups["threads"] / 512

# print(groups)

def selectMethod(df, method):
    df = df[(df["method"] == method)]
    withL2 = df[~df["flushL2"]]
    flushL2 = df[df["flushL2"]]

    return (withL2, flushL2)
#
scalarPack1_2_L2, scalarPack1_2 = selectMethod(groups, "Pack1-2")
subgroupPack2_2_L2, subgroupPack2_2 = selectMethod(groups, "Pack2-2")
subgroupPack2_4_L2, subgroupPack2_4 = selectMethod(groups, "Pack2-4")
subgroupPack2_8_L2, subgroupPack2_8 = selectMethod(groups, "Pack2-8")

subgroupPack4_4_L2, subgroupPack4_4 = selectMethod(groups, "Pack4-4")
subgroupPack4_8_L2, subgroupPack4_8 = selectMethod(groups, "Pack4-8")
subgroupPack4_16_L2, subgroupPack4_16 = selectMethod(groups, "Pack4-16")
subgroupPack4_32_L2, subgroupPack4_32 = selectMethod(groups, "Pack4-32")

subgroupPack8_16_L2, subgroupPack8_16 = selectMethod(groups, "Pack8-16")
subgroupPack8_32_L2, subgroupPack8_32 = selectMethod(groups, "Pack8-32")
subgroupPack8_64_L2, subgroupPack8_64 = selectMethod(groups, "Pack8-64")
subgroupPack8_128_L2, subgroupPack8_128 = selectMethod(groups, "Pack8-128")

subgroupPack16_32_L2, subgroupPack16_32 = selectMethod(groups, "Pack16-32")
subgroupPack16_128_L2, subgroupPack16_128 = selectMethod(groups, "Pack16-128")
subgroupPack16_256_L2, subgroupPack16_256 = selectMethod(groups, "Pack16-256")

subgroupPack32_32_L2, subgroupPack32_32 = selectMethod(groups, "Pack32-32")
subgroupPack32_128_L2, subgroupPack32_128 = selectMethod(groups, "Pack32-128")
subgroupPack32_256, subgroupPack32_256 = selectMethod(groups, "Pack32-256")
subgroupPack32_512, subgroupPack32_512 = selectMethod(groups, "Pack32-512")

pack1 = scalarPack1_2;
pack1_L2 = scalarPack1_2_L2;
print(max(pack1["N"]))

pack2 = subgroupPack2_4;
pack2_L2 = subgroupPack2_8_L2;

pack4 = subgroupPack4_8; # TODO try different split sizes
pack4_L2 = subgroupPack4_8_L2;

pack8 = subgroupPack8_16; # TODO try different split sizes
pack8_L2 = subgroupPack8_32_L2;

pack16 = subgroupPack16_256
pack16_L2 = subgroupPack16_128_L2

pack32 = subgroupPack32_512;
pack32_L2 = subgroupPack32_128_L2;

# pack32["K"] = pack32["N"] / pack32["splitSize"]
# pack32["subgroups"] = pack32["K"]
# pack32["w"] = pack32["subgroups"] / (2208)
print(pack32)

# print(scalarPack1_2)

showL2 = True

property = "throughput"
axis = "w"

plt.plot(pack1[axis], pack1[property], "-", label="1-thread with\nsplit-size 2", color="tab:blue")
if showL2:
    plt.plot(pack1_L2[axis], pack1_L2[property], ":", color="tab:blue")

plt.plot(pack2[axis], pack2[property], "-", label="2-threads with\nsplit-size 4", color="tab:orange")
if showL2: 
    plt.plot(pack2_L2[axis], pack2_L2[property], ":", color="tab:orange")

plt.plot(pack4[axis], pack4[property], "-", label="4-threads with\nsplit-size 32", color="tab:green")
if showL2:
    plt.plot(pack4_L2[axis], pack4_L2[property], ":", color="tab:green")

plt.plot(pack8[axis], pack8[property], "-", label="8-threads with\nsplit-size 32", color="tab:red")
if showL2:
    plt.plot(pack8_L2[axis], pack8_L2[property], ":", color="tab:red")

plt.plot(pack16[axis], pack16[property], "-", label="16-threads", color="tab:purple")
if showL2:
    plt.plot(pack16_L2[axis], pack16_L2[property], ":", color="tab:purple")

plt.plot(pack32[axis], pack32[property], "-", label="32-threads", color="tab:brown")
if showL2:
    plt.plot(pack32_L2[axis], pack32_L2[property], ":", color="tab:brown")


# plt.plot(subgroupPack4_4["w"], subgroupPack4_4["throughput"], "-", label="4-threads")
# plt.plot(subgroupPack8_32["w"], subgroupPack8_32["throughput"], "-", label="8-threads")
# plt.plot(subgroupPack16_128["w"], subgroupPack16_128["throughput"], "-", label="16-threads")

# plt.plot(subgroupPack2_4["w"], subgroupPack2_4["throughput"], ":", label="pack2-4")
# plt.plot(subgroupPack4_8["w"], subgroupPack4_8["throughput"], "-", label="pack4-8")
# plt.plot(subgroupPack4_32["w"], subgroupPack4_32["throughput"], ":", label="pack4-32")
# plt.plot(subgroupPack8_128["w"], subgroupPack8_128["throughput"], ":", label="pack8-128")
# plt.plot(subgroupPack16_32["w"], subgroupPack16_32["throughput"], ":", label="pack16-32")
# plt.plot(subgroupPack16_256["w"], subgroupPack16_256["throughput"], ":", label="pack16-256")
# plt.plot(subgroupPack32_32["w"], subgroupPack32_32["throughput"], ":", label="pack32-32")
# plt.plot(subgroupPack32_128["w"], subgroupPack32_128["throughput"], ":", label="pack32-128")
# plt.plot(subgroupPack32_256["w"], subgroupPack32_256["throughput"], ":", label="pack32-256")
# plt.plot(subgroupPack32_512["w"], subgroupPack32_512["throughput"], ":", label="pack32-512")


# _, pack2 = selectMethod(groups, "Pack-2")
# _, pack4 = selectMethod(groups, "Pack-4")
# _, pack8 = selectMethod(groups, "Pack-8")
# _, pack16 = selectMethod(groups, "Pack-16")
#
# pack1["throughput"] = (pack1["N"] / (pack1["latency"] * 1e-3)) * 1e-9;
# pack2["throughput"] = (pack2["N"] / (pack2["latency"] * 1e-3)) * 1e-9;
# pack4["throughput"] = (pack4["N"] / (pack4["latency"] * 1e-3)) * 1e-9;
# pack8["throughput"] = (pack8["N"] / (pack8["latency"] * 1e-3)) * 1e-9;
# pack16["throughput"] = (pack16["N"] / (pack16["latency"] * 1e-3)) * 1e-9;
#
# pack1["threadsPerPack"] = 1;
# pack2["threadsPerPack"] = 2;
# pack4["threadsPerPack"] = 4;
# pack8["threadsPerPack"] = 8;
# pack16["threadsPerPack"] = 16;
#
# def inferData(pack):
#     pack["K"] = pack["N"] / pack["splitSize"]
#     pack["threads"] = pack["K"] * pack["threadsPerPack"]
#     pack["w"] = pack["threads"] / maxOccupantThreads
#     return pack
#
# pack1 = inferData(pack1)
#
#
# # pack2["threads"] = pack2["N"] * pack2["threadsPerPack"];
# # pack4["threads"] = pack4["N"] * pack4["threadsPerPack"];
# # pack8["threads"] = pack8["N"] * pack8["threadsPerPack"];
# # pack16["threads"] = pack16["N"] * pack16["threadsPerPack"];
#
#
# # scalarSplitFlushL2["throughput_min"] = ((scalarSplitFlushL2["N"] / (scalarSplitFlushL2["splitSize"])) / ((scalarSplitFlushL2["latency"] - scalarSplitFlushL2["var"]) * 1e-3)) * 1e-9;
# # scalarSplitFlushL2["throughput_max"] = ((scalarSplitFlushL2["N"] / (scalarSplitFlushL2["splitSize"])) / ((scalarSplitFlushL2["latency"] + scalarSplitFlushL2["var"]) * 1e-3)) * 1e-9;
#
# plt.plot(pack1["w"], pack1["throughput"], "bx-", label="pack-1");
# # plt.plot(pack2["splitSize"], pack2["throughput"], label="pack-2");
# # plt.plot(pack4["splitSize"], pack4["throughput"], label="pack-4");
# # plt.plot(pack8["splitSize"], pack8["throughput"], label="pack-8");
# # plt.plot(pack16["splitSize"], pack16["throughput"], label="pack-16");
#
# # plt.fill_between(scalarSplitFlushL2["splitSize"], scalarSplitFlushL2["throughput_min"], scalarSplitFlushL2["throughput_max"],
# #                  alpha=0.2, color="tab:blue")
#
#
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')


ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel(axis)
plt.ylabel("Billion Packs / Second")
# plt.xscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig("psa_pack_work.pdf", format="pdf")

plt.show()
#
#
#
