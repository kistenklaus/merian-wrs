import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

# -------------------- Data Loading & Processing (Vectorized) --------------------

# Load the CSV file (ensure proper dtypes if possible)
file_path = "./wrs_benchmark_1000.csv"
df = pd.read_csv(file_path)
plt.style.use('ggplot')

font = {'family' : 'normal',
        'size'   : 22}

mplt.rc('font', **font)

# Keep only the needed columns.
df = df[['N', 'S', 'method', 'total_latency']]

# Group by (N, S, method) and compute the minimum latency per group.
grouped = df.groupby(['N', 'S', 'method'])['total_latency'].min().reset_index()

# Define explicitly our base colors.
base_colors = {
    'ITS-COOP': (1.0, 0.0, 0.0),                # Red
    'PSA-InlineSplitPack': (0.0, 1.0, 0.0),     # Green
    'PSA-SerialSplitSubgroupPack': (0.0, 0.0, 1.0)  # Blue
}
methods = list(base_colors.keys())

# Pivot the DataFrame so each (N, S) pair is a row and each method is a column.
pivot_df = grouped.pivot_table(index=['N','S'], columns='method', values='total_latency')

# Restrict to our desired methods (if there are extra ones, drop them)
pivot_df = pivot_df[methods]

# Compute best and worst latencies and corresponding methods (vectorized)
best_latency = pivot_df.min(axis=1)
worst_latency = pivot_df.max(axis=1)
best_method = pivot_df.idxmin(axis=1)
worst_method = pivot_df.idxmax(axis=1)
max_latency = pivot_df.max(axis=1)

# Set bonus factor for the best method.
best_bonus = 1.0

# Compute raw weights for each method (vectorized)
weights = {}
for m in methods:
    # raw weight = max_latency - latency
    raw = max_latency - pivot_df[m]
    # Multiply by bonus factor where this method is the best
    bonus = np.where(best_method == m, best_bonus, 1.0)
    weights[m] = raw * bonus

# Build a DataFrame of weights
weights_df = pd.DataFrame(weights, index=pivot_df.index)
sum_weights = weights_df.sum(axis=1)
normalized_weights = weights_df.div(sum_weights, axis=0)

# Compute final color components as the weighted sum of base colors.
color_r = sum(normalized_weights[m] * base_colors[m][0] for m in methods)
color_g = sum(normalized_weights[m] * base_colors[m][1] for m in methods)
color_b = sum(normalized_weights[m] * base_colors[m][2] for m in methods)

final_color = pd.DataFrame({'r': color_r, 'g': color_g, 'b': color_b}, index=pivot_df.index)
# Create a tuple for each row (each color)
final_color_tuple = list(zip(final_color['r'], final_color['g'], final_color['b']))

# Assemble an aggregated DataFrame.
aggregated_df = pd.DataFrame({
    'N': pivot_df.index.get_level_values('N'),
    'S': pivot_df.index.get_level_values('S'),
    'best_method': best_method.values,
    'worst_method': worst_method.values,
    'best_latency': best_latency.values,
    'worst_latency': worst_latency.values,
    'interpolated_color': final_color_tuple
})

# -------------------- Visualization (Same as Before) --------------------

# Get sorted unique values for N and S.
unique_N = np.sort(aggregated_df['N'].unique())
unique_S = np.sort(aggregated_df['S'].unique())

# Map the unique N and S values to pixel indices (x-axis is N, y-axis is S).
N_to_idx = {n: i for i, n in enumerate(unique_N)}
S_to_idx = {s: j for j, s in enumerate(unique_S)}

# Build an image array (rows correspond to S, columns to N).
img = np.zeros((len(unique_S), len(unique_N), 3))
for _, row in aggregated_df.iterrows():
    x = N_to_idx[row['N']]   # x-axis (N)
    y = S_to_idx[row['S']]   # y-axis (S)
    img[y, x, :] = row['interpolated_color']

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img, origin='lower', aspect='auto')
ax.set_xlabel("Number of weights (N)")
ax.set_ylabel("Amount of samples (S)")
# ax.set_title("B")
ax.grid(False)

# Limit tick marks to 10 evenly spaced values per axis.
n_ticks = 5
x_indices = np.linspace(0, len(unique_N) - 1, n_ticks, dtype=int) if len(unique_N) > n_ticks else np.arange(len(unique_N))
y_indices = np.linspace(0, len(unique_S) - 1, n_ticks, dtype=int) if len(unique_S) > n_ticks else np.arange(len(unique_S))
ax.set_xticks(x_indices)
ax.set_yticks(y_indices)

# Custom formatters to display tick labels in scientific notation like "2.3·10^7".
def format_log_tick(val, pos, unique_vals):
    if pos == 0: 
        return ""
    idx = int(round(val))
    if 0 <= idx < len(unique_vals):
        v = unique_vals[idx]
        if v <= 0:
            return "0"
        exponent = int(np.floor(np.log10(v)))
        mantissa = v / (10 ** exponent)
        return f"${mantissa:.1f}\\cdot10^{{{exponent}}}$"
    return ""

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_log_tick(x, pos, unique_N)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: format_log_tick(y, pos, unique_S)))

# Create legend handles using the explicitly defined base_colors.
legend_handles = [mpatches.Patch(color=base_colors[m], label=m) for m in base_colors]
# ax.legend(handles=legend_handles, title="Methods", loc='upper right')

plt.show()
