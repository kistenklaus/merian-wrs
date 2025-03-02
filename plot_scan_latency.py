import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("scan_benchmark.csv")

property = "memory_throughput";

# Group by N and method, then take the minimum latency for each group
df_min_latency = df.groupby(["N", "method"], as_index=False)[property].max()

# Pivot to get a clearer format for plotting
plt.figure(figsize=(10, 6))
for method in df_min_latency["method"].unique():
    subset = df_min_latency[df_min_latency["method"] == method]
    plt.plot(subset["N"], subset[property], label=method)

# Formatting
plt.xlabel("N")
plt.xscale("log")
plt.ylabel("Latency (min)")
plt.title("Minimum Latency over N for Different Methods")
plt.legend(title="Method")
plt.grid(True)
plt.show()
