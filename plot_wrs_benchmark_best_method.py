import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load CSV file
file_path = "wrs_benchmark.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Step 1: Remove duplicates for (N, S, method), keeping only the lowest total_latency
df = df.loc[df.groupby(['N', 'S', 'method'])['total_latency'].idxmin()]

# Step 2: Find the best method for each (N, S) pair (the one with the lowest total_latency)
best_methods_df = df.loc[df.groupby(['N', 'S'])['total_latency'].idxmin(), ['N', 'S', 'method']]

# Convert method names to categorical values
method_categories = best_methods_df['method'].astype('category')
best_methods_df['method_code'] = method_categories.cat.codes  # Convert methods to integer codes
method_labels = method_categories.cat.categories  # Get unique method names

# Extract unique N and S values (sorted for proper mapping)
unique_N = np.sort(best_methods_df['N'].unique())
unique_S = np.sort(best_methods_df['S'].unique())

# Ensure we pick exactly 10 ticks, even if there are more values
num_ticks = 10
tick_N = np.linspace(0, len(unique_N) - 1, num_ticks, dtype=int)  # Select 10 evenly spaced indices
tick_S = np.linspace(0, len(unique_S) - 1, num_ticks, dtype=int)  # Select 10 evenly spaced indices

# Get actual tick labels
tick_N_labels = unique_N[tick_N]
tick_S_labels = unique_S[tick_S]

# Pivot the data to match the (N, S) grid
pivot_table = best_methods_df.pivot(index='S', columns='N', values='method_code')

# Define colormap
cmap = ListedColormap(plt.cm.get_cmap('Paired', len(method_labels)).colors)

# Create figure and plot heatmap
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = ax.imshow(pivot_table, aspect='auto', cmap=cmap, origin='lower')

# Set fixed number of ticks (10 on each axis)
ax.set_xticks(tick_N)
ax.set_xticklabels(tick_N_labels, rotation=90)

ax.set_yticks(tick_S)
ax.set_yticklabels(tick_S_labels)

# Add colorbar with method labels
cbar = plt.colorbar(heatmap, ax=ax)
cbar.set_ticks(np.arange(len(method_labels)))
cbar.set_ticklabels(method_labels)

# Labels and title
ax.set_xlabel("N (fixed 10 ticks)")
ax.set_ylabel("S (fixed 10 ticks)")
ax.set_title("Best Method Heatmap with Fixed 10 Ticks")

# Show the plot
plt.show()
