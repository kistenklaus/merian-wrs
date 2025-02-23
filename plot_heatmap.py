import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Data Loading & Processing --------------------

# Load the CSV file
file_path = "./wrs_benchmark.csv"
df = pd.read_csv(file_path)

# Select only the needed columns
df = df[['N', 'S', 'method', 'total_latency']]

# Group by (N, S, method) and compute the minimum latency per group.
grouped_df = df.groupby(['N', 'S', 'method'])['total_latency'].min().reset_index()

# Define the explicit base colors for each method (RGB values in [0,1])
base_colors = {
    'ITS-COOP': (1.0, 0.0, 0.0),                # Red
    'PSA-InlineSplitPack': (0.0, 1.0, 0.0),     # Green
    'PSA-SerialSplitSubgroupPack-16': (0.0, 0.0, 1.0),  # Blue
    'PSA-SerialSplitSubgroupPack-32': (0.0, 0.0, 1.0)   # Yellow
}

def interpolate_color(method_latencies, base_colors):
    """
    Compute a weighted color by blending the base colors of each method,
    where each method's weight is computed as (max_latency - latency).
    
    Parameters:
      method_latencies (dict): Mapping from method names to latency values.
      base_colors (dict): Mapping from method names to base RGB color tuple.
    
    Returns:
      tuple: Interpolated RGB color in [0, 1].
    """
    max_latency = max(method_latencies.values())
    raw_weights = {method: max_latency - latency for method, latency in method_latencies.items()}
    total_weight = sum(raw_weights.values())
    if total_weight == 0:
        normalized_weights = {method: 1 / len(raw_weights) for method in raw_weights}
    else:
        normalized_weights = {method: weight / total_weight for method, weight in raw_weights.items()}
    
    blended_color = np.array([0.0, 0.0, 0.0])
    for method, weight in normalized_weights.items():
        if method in base_colors:
            blended_color += weight * np.array(base_colors[method])
    return tuple(blended_color)

# Prepare a list to collect aggregated results for each (N, S) pair.
aggregated_data = []

for (N_val, S_val), group in grouped_df.groupby(['N', 'S']):
    method_latencies = dict(zip(group['method'], group['total_latency']))
    interp_color = interpolate_color(method_latencies, base_colors)
    
    best_idx = group['total_latency'].idxmin()
    worst_idx = group['total_latency'].idxmax()
    best_method = group.loc[best_idx, 'method']
    best_latency = group.loc[best_idx, 'total_latency']
    worst_method = group.loc[worst_idx, 'method']
    worst_latency = group.loc[worst_idx, 'total_latency']
    
    aggregated_data.append({
        'N': N_val,
        'S': S_val,
        'best_method': best_method,
        'best_latency': best_latency,
        'worst_method': worst_method,
        'worst_latency': worst_latency,
        'interpolated_color': interp_color
    })

aggregated_df = pd.DataFrame(aggregated_data)

# -------------------- Visualization --------------------

# Get sorted unique values for N and S.
unique_N = np.sort(aggregated_df['N'].unique())
unique_S = np.sort(aggregated_df['S'].unique())

# Create mappings from N and S values to pixel indices (zero-indexed).
N_to_idx = {n: i for i, n in enumerate(unique_N)}
S_to_idx = {s: j for j, s in enumerate(unique_S)}

# Build the image array using the pixel indices.
img = np.zeros((len(unique_N), len(unique_S), 3))
for _, row in aggregated_df.iterrows():
    i = N_to_idx[row['N']]
    j = S_to_idx[row['S']]
    img[i, j, :] = row['interpolated_color']

fig2, ax2 = plt.subplots(figsize=(10, 8))
cax = ax2.imshow(img, origin='lower', aspect='auto')
ax2.set_xlabel("S (index)")
ax2.set_ylabel("N (index)")
ax2.set_title("Performance Heatmap (Image Plot)")

# Limit tick marks to 10 evenly spaced values per axis.
n_ticks = 10

# For the x-axis (S values)
if len(unique_S) > n_ticks:
    x_indices = np.linspace(0, len(unique_S) - 1, n_ticks, dtype=int)
else:
    x_indices = np.arange(len(unique_S))
ax2.set_xticks(x_indices)
ax2.set_xticklabels(unique_S[x_indices])

# For the y-axis (N values)
if len(unique_N) > n_ticks:
    y_indices = np.linspace(0, len(unique_N) - 1, n_ticks, dtype=int)
else:
    y_indices = np.arange(len(unique_N))
ax2.set_yticks(y_indices)
ax2.set_yticklabels(unique_N[y_indices])

plt.show()
