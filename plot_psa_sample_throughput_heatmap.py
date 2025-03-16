import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

# Configuration
latencyBase = "latency"
method_name = "PSA2-0"
min_N = 1 << 16
max_N = 1 << 28
min_S = 1 << 16
max_S = 1 << 28

# Load evaluation results
bench_unserious = pd.read_csv("./wrs_benchmark_sample_throughput.csv")

bench = pd.concat([bench_unserious])
print(bench["method"].unique())

# Implicit columns
bench["latency"] = bench["build_latency"] + bench["sample_latency"]
bench["throughput"] = bench["S"] / (bench[latencyBase] * 1e-3) * 1e-9


# Select methods
method = bench[bench["method"] == method_name]

# Filter data
method = method[(method["N"] >= min_N) & (method["N"] <= max_N) 
                & (method["S"] >= min_S) & (method["S"] <= max_S)]
print(method)

# Plotting
x_scale = "log"
y_scale = "log"
# TODO: Create a heatmap with N on the x-axis and S on the Y-axis
# The input data (i.e. method) can have any distribution of N and S values.
# So for example it might be that the N's are distributed logartihmically
# and the S linearly.
# In the heatmap both should be display with a logarithmic scale.
# Also the input data is very large the ticks on the x and y-axis should 
# only display exact powers of 10. 
# The dataset might not contain the powers of 10 exactly if let's say 10^6 is not 
# contained in the dataset i want the heatmap to still have a tick at approximatly this position
# saying 10^6. 

# 1) Define log-spaced bins for N and S
n_bins = 50  # Adjust as needed
N_min, N_max = method["N"].min(), method["N"].max()
S_min, S_max = method["S"].min(), method["S"].max()

binsN = np.logspace(np.log10(N_min), np.log10(N_max), n_bins)
binsS = np.logspace(np.log10(S_min), np.log10(S_max), n_bins)

# 2) Compute weighted 2D histogram:
#    - 'h' accumulates the sum of throughput in each (N,S) bin
#    - 'counts' counts how many samples fall into each bin
h, xedges, yedges = np.histogram2d(
    method["N"], method["S"],
    bins=[binsN, binsS],
    weights=method["throughput"]
)
counts, _, _ = np.histogram2d(
    method["N"], method["S"],
    bins=[binsN, binsS]
)

# 3) Avoid division-by-zero when computing average throughput
avg_throughput = h / (counts + 1e-9)

# 4) Plot heatmap
plt.figure(figsize=(8, 6))
# pcolormesh expects the data in [y,x] order so we transpose
mesh = plt.pcolormesh(xedges, yedges, avg_throughput.T, shading='auto')

# 5) Apply log scale to axes
plt.xscale('log')
plt.yscale('log')

# 6) Format major ticks at powers of 10
ax = plt.gca()
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))

# 7) Labels and colorbar
plt.xlabel('N')
plt.ylabel('S')
cbar = plt.colorbar(mesh)
cbar.set_label('Billion Samples / Second')

plt.title(f'Average throughput heatmap')
plt.tight_layout()
plt.show()
