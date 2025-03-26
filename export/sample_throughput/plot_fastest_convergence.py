import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
import os

numPoints = 125

dirname = os.path.dirname(__file__)
bench = pd.read_csv(os.path.join(dirname, "./sweep.csv"))

# bench = bench[bench["N"] < 2e7]

latencyBase = "latency"
bench["latency"] = bench["build_latency"] + bench["sampling_latency"]
bench["throughput"] = bench["S"] / (bench[latencyBase] * 1e-3) * 1e-9

print(bench["group"].unique())

maxLatency = np.min(bench.groupby(["N", "group"])[latencyBase].max())
minLatency = np.min(bench[latencyBase])
print(f"maxLatency:{maxLatency}ms,   minLatency:{minLatency}ms");

bench = bench[bench[latencyBase] < maxLatency]

latencyBucketMargin = 0.01
bench["bin_latency"] = (bench[latencyBase] / latencyBucketMargin).astype(int);

print(bench["group"].unique())


dirname = os.path.dirname(__file__)
rmseBench = pd.read_csv(os.path.join(dirname, "../rmse/sweep.csv"))

print(rmseBench)

merged = pd.merge(bench, rmseBench, how="inner", on=["group", "N", "S"])

best = merged.loc[merged.groupby(["bin_latency", "N"])["rmse"].idxmin()].reset_index();

print(best[[latencyBase, "N", "rmse", "group", "S"]])

# For each (N, group), average build_latency
# bench["build_latency"] = (
#     bench.groupby(["N", "group"])["build_latency"].transform("mean")
# )

# Compute total latency + throughput


def foo(df, xAxis, yAxis, property, colormap, labels, xscale, yscale, xPixels, yPixels, xticks, yticks,
        xlabel, ylabel):
    df = df.copy()
    if xscale == "log":
        # Filter out non-positive values because log scaling requires positive numbers.
        df = df[df[xAxis] > 0]
        x_min = df[xAxis].min()
        x_max = df[xAxis].max()
        # Compute bin indices in log space.
        df['bin_x'] = ((np.log(df[xAxis]) - np.log(x_min)) / (np.log(x_max) - np.log(x_min)) * xPixels).astype(int)
    elif xscale == "linear":
        x_min = df[xAxis].min()
        x_max = df[xAxis].max()
        # Compute bin indices in linear space.
        df['bin_x'] = ((df[xAxis] - x_min) / (x_max - x_min) * xPixels).astype(int)
    else:
        raise ValueError("xscale must be either 'linear' or 'log'")

    if yscale == "log":
        # Filter out non-positive values because log scaling requires positive numbers.
        df = df[df[yAxis] > 0]
        y_min = df[yAxis].min()
        y_max = df[yAxis].max()
        # Compute bin indices in log space.
        df['bin_y'] = ((np.log(df[yAxis]) - np.log(y_min)) / (np.log(y_max) - np.log(y_min)) * yPixels).astype(int)
    elif yscale == "linear":
        y_min = df[yAxis].min()
        y_max = df[yAxis].max()
        # Compute bin indices in linear space.
        df['bin_y'] = ((df[yAxis] - y_min) / (y_max - y_min) * yPixels).astype(int)
    else:
        raise ValueError("yscale must be either 'linear' or 'log'")

    df['bin_x'] = df['bin_x'].clip(upper=xPixels - 1)
    df['bin_y'] = df['bin_y'].clip(upper=yPixels - 1)

    def aggregate_group(group):
        # Use mode to get the most common property (assuming property is a string).
        mode_property = group[property].mode()[0]
        # Compute representative x and y values for the bin (using the mean of the original coordinates).
        rep_x = group[xAxis].median()
        rep_y = group[yAxis].median()
        bin_x = group['bin_x'].iloc[0]
        bin_y = group['bin_y'].iloc[0]
        return pd.Series({xAxis: rep_x, yAxis: rep_y, 'bin_x': bin_x, 'bin_y': bin_y, property: mode_property})

    grouped_df = df.groupby(['bin_x', 'bin_y']).apply(aggregate_group).reset_index(drop=True)

    print("Binned DataFrame:\n", df)
    print("\nAggregated DataFrame by bin:\n", grouped_df)

    # Create a blank image with a white background.
    image = np.zeros((yPixels, xPixels, 3))
    
    # Convert the colormap to RGB.
    color_map_rgb = {key: to_rgb(color) for key, color in colormap.items()}
    
    # Populate the image: use bin_x and bin_y as pixel coordinates.
    for _, row in grouped_df.iterrows():
        x_idx = int(row['bin_x'])
        y_idx = int(row['bin_y'])
        prop_value = row[property]
        rgb = color_map_rgb.get(prop_value, (0, 0, 0))
        image[y_idx, x_idx, :] = rgb

    # Determine the data coordinate extent from the representative x and y values.
    # (These should reflect the overall data coordinate range.)
    if not grouped_df.empty:
        data_x_min = grouped_df[xAxis].min()
        data_x_max = grouped_df[xAxis].max()
        data_y_min = grouped_df[yAxis].min()
        data_y_max = grouped_df[yAxis].max()
    else:
        data_x_min, data_x_max = 0, xPixels
        data_y_min, data_y_max = 0, yPixels

    # Transform the extent based on the scaling.
    if xscale == "log":
        extent_x_min = np.log10(data_x_min)
        extent_x_max = np.log10(data_x_max)
    else:
        extent_x_min = data_x_min
        extent_x_max = data_x_max

    if yscale == "log":
        extent_y_min = np.log10(data_y_min)
        extent_y_max = np.log10(data_y_max)
    else:
        extent_y_min = data_y_min
        extent_y_max = data_y_max

    # Create the plot, using extent to map pixel coordinates to data coordinates.
    plt.figure(figsize=(14, 8))
    plt.imshow(image, origin='lower', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max),
               interpolation="none",aspect=(extent_x_max - extent_x_min) / (extent_y_max - extent_y_min))
    
    # Build the legend: one entry per key in the colormap.
    legend_handles = []
    for key, color in colormap.items():
        label_text = labels.get(key, key)
        patch = mpatches.Patch(color=to_rgb(color), label=label_text)
        legend_handles.append(patch)
    plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    
    # Get current axes to set custom ticks.
    ax = plt.gca()

    if xscale == "log":
        # Transform tick positions to log space.
        xtick_positions = [np.log10(t) for t in xticks]
        x_tick_labels = [r'$10^{%d}$' % int(np.log10(t)) for t in xticks]
    else:
        xtick_positions = xticks
        x_tick_labels = [str(t) for t in xticks]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(x_tick_labels)

    # Set custom y ticks.
    if yscale == "log":
        ytick_positions = [np.log10(t) for t in yticks]
        y_tick_labels = [r'$10^{%d}$' % int(np.log10(t)) for t in yticks]
    else:
        ytick_positions = yticks
        y_tick_labels = [str(t) for t in yticks]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(y_tick_labels)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


plt.rcParams.update({'font.size': 24})

colormap = {
        "ITS-128" : "tab:orange",
        "Cutpoint-128" : "tab:green",
        "PSA2-0" : "tab:red",
        "PSA2-128" : "tab:brown",
        }
labels ={
        "ITS-128" : "its-coop",
        "Cutpoint-128" : "cutpoint",
        "PSA2-0" : "psa-baseline",
        "PSA2-128" : "psa-sectioned",
        }
# Call the function to plot the image.
foo(best, "N", latencyBase, "group", colormap, labels, "log", "linear", numPoints, numPoints,
    [1e5,1e6,1e7], [0.1,0.2,0.3,0.4,0.5,0.6,0.7], xlabel="Amount of Items (N)", ylabel="Latency (ms)")


plt.savefig(f'wrs_rmse_convergence_heatmap.pdf', format="pdf")
plt.show()
