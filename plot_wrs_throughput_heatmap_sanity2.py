import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

# bench_its_cutpoint = pd.read_csv("./wrs_benchmark.csv")
#
# bench_its = bench_its_cutpoint.loc[
#     (bench_its_cutpoint["group"] == "ITS-0") |
#     (bench_its_cutpoint["group"] == "ITS-128") |
#     (bench_its_cutpoint["group"] == "ITS-128-pArray")
# ]
# bench_its.loc[bench_its["group"] == "ITS-128-pArray", "group"] = "ITS-128"
# bench_cutpoint = bench_its_cutpoint[
#     bench_its_cutpoint["group"] == "Cutpoint-128"
# ]

bench_psa = pd.read_csv("./wrs_benchmark_psa_e28.csv")
bench0 = pd.read_csv("./wrs_benchmark_baseline.csv")
# bench1 = pd.read_csv("./wrs_benchmark.csv")

bench = pd.concat([bench_psa, bench0], ignore_index=True)

print(bench["N"].unique())
# S = 65536
# N = 65536
# N = 105047
N = 1001640
# N = 10095907
# N = 9419161

bench = bench[bench["N"] == N]

latencyBase = "latency"
property = "throughput"
bench["latency"] = bench["build_latency"] + bench["sampling_latency"]
bench["throughput"] = bench["S"] / (bench[latencyBase] * 1e-3) * 1e-9

bench = bench.sort_values(by = "S")



for group in bench["group"].unique():
    df = bench[bench["group"] == group]
    plt.plot(df["S"], df[property], label=group)

plt.xscale("log")

plt.grid()
plt.xlabel("Amount of Samples")
plt.ylabel(property)

plt.legend()
plt.show()


