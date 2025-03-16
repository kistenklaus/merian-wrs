import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

bench_its_cutpoint = pd.read_csv("./wrs_benchmark_its_cutpoint_heatmap.csv")

bench_its = bench_its_cutpoint.loc[
    (bench_its_cutpoint["group"] == "ITS-0") |
    (bench_its_cutpoint["group"] == "ITS-128") |
    (bench_its_cutpoint["group"] == "ITS-128-pArray")
]
bench_its.loc[bench_its["group"] == "ITS-128-pArray", "group"] = "ITS-128"

bench_psa = pd.read_csv("./wrs_benchmark.csv")

bench_cutpoint = bench_its_cutpoint[
    bench_its_cutpoint["group"] == "Cutpoint-128"
]

bench = pd.concat([bench_its, bench_cutpoint, bench_psa], ignore_index=True)

print(bench["S"].unique())
# S = 65536
# S = 67108863
S = 10452301

bench = bench[bench["S"] == S]

latencyBase = "latency"
bench["latency"] = bench["build_latency"] + bench["sampling_latency"]
bench["throughput"] = bench["S"] / (bench[latencyBase] * 1e-3) * 1e-9

bench = bench.sort_values(by = "N")



for group in bench["group"].unique():
    df = bench[bench["group"] == group]
    plt.plot(df["N"], df["throughput"], label=group)

plt.xscale("log")

plt.legend()
plt.show()


