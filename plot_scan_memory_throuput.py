import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

# Load the data
df = pd.read_csv("scan_benchmark.csv")
property = "memory_throughput"

# Group by N and method, then take the maximum throughput for each group
df_min_latency = df.groupby(["N", "method"], as_index=False)[property].max()

# Interpolation settings
num_points = 59  # Number of interpolated points
interpolation_method = "linear"  # Options: 'linear', 'savgol', 'lowess'

plt.figure(figsize=(10, 6))
plt.axhline(y=504, color='r', linestyle=':')
plt.text(df['N'].max(), 512, f"504", fontsize=10, verticalalignment='center', color='red')

for method in df_min_latency["method"].unique():
    subset = df_min_latency[df_min_latency["method"] == method]
    x, y = subset["N"].values, subset[property].values

    if len(x) > 5:  # Ensure enough points for interpolation
        x_new = np.logspace(np.log10(x.min()), np.log10(x.max()), num_points)
        
        if interpolation_method == "linear":
            f = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")
            y_smooth = f(x_new)
            plt.plot(x_new, y_smooth, label=method, marker=".")  # Add markers to points
        elif interpolation_method == "savgol":
            y_smooth = savgol_filter(y, window_length=min(7, len(y)), polyorder=2, mode='nearest')
            x_new, y_smooth = x, y_smooth  # Keep original x points for consistency
            plt.plot(x_new, y_smooth, label=method)
        elif interpolation_method == "lowess":
            smoothed = lowess(y, x, frac=0.3)
            x_new, y_smooth = smoothed[:, 0], smoothed[:, 1]
            plt.plot(x_new, y_smooth, label=method)
        else:
            raise ValueError("Invalid interpolation method")
        
        plt.text(x_new[-1] * 1.2, y_smooth[-1], f"{y_smooth[-1]:.1f}", fontsize=10, verticalalignment='center')
    else:
        plt.plot(x, y, label=method, marker="x")  # Fallback for small datasets

# Formatting
plt.xlabel("Element count (N)")
plt.xscale("log")
plt.xlim(df['N'].min(), df['N'].max() * 5)
plt.ylabel("Memory Throughput (GB/s)")
plt.legend(title="Method")
plt.grid(True)
plt.show()
