import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

numPoints = 250
serial = False
scale = "log"
axis = "N"
property = "splitSize"

if serial:
    method_prefix = "Serial"
else:
    method_prefix = "Inline"

latencyBase = "splitPackLatency"

benche5e7_s2s128_serial = pd.read_csv("./psa_inline_splitpack_benchmark_optimal_serial_log_e5e7_s2s128.csv")
benche5e7_s2s128_inline = pd.read_csv("./psa_inline_splitpack_benchmark_optimal_inline_log_e5e7_s2s128.csv")

# For demonstration, let's just use the 'inline' data
bench = pd.concat([benche5e7_s2s128_inline])
bench["throughput"] = bench["N"] / (bench[latencyBase] * 1e-3) * 1e-9

sp1  = bench[bench["group"] == f"{method_prefix}SplitPack-1"]
sp2  = bench[bench["group"] == f"{method_prefix}SplitPack-2"]
sp4  = bench[bench["group"] == f"{method_prefix}SplitPack-4"]
sp8  = bench[bench["group"] == f"{method_prefix}SplitPack-8"]
sp16 = bench[bench["group"] == f"{method_prefix}SplitPack-16"]
sp32 = bench[bench["group"] == f"{method_prefix}SplitPack-32"]

plt.rcParams.update({'font.size': 16})

def edges_from_centers(centers):
    """
    Convert 'center' coordinates to bin edges for pcolormesh.
    If centers=[c0, c1, ..., c_{n-1}],
    returns edges=[e0, e1, ..., e_n] with e_i 
    ~ halfway between c_{i} and c_{i+1}.
    """
    edges = np.zeros(len(centers) + 1, dtype=float)
    if len(centers) > 1:
        edges[1:-1] = 0.5 * (centers[1:] + centers[:-1])
        edges[0]  = centers[0]  - 0.5 * (centers[1] - centers[0])
        edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    else:
        # if there's only one center, pick a small +/- buffer
        edges[0] = centers[0] - 0.5
        edges[-1] = centers[0] + 0.5
    return edges

def plot_all_methods(sp1, sp2, sp4, sp8, sp16, sp32):
    """
    Create a single figure with 6 heatmaps (3x2 grid). 
    We'll set both X and Y to logarithmic scales,
    and display ticks at powers-of-10 only if they exist in the data.
    """
    # 1) Global color scale
    all_throughputs = pd.concat([sp1, sp2, sp4, sp8, sp16, sp32])["throughput"]
    vmin, vmax = all_throughputs.min(), all_throughputs.max()
    
    method_data = [
        ("1-invocation",  sp1),
        ("2-invocations", sp2),
        ("4-invocations", sp4),
        ("8-invocations", sp8),
        ("16-invocations",sp16),
        ("32-invocations",sp32),
    ]
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 19))
    
    # Powers of 10 for the X- and Y-axis
    desired_x_values = [1e5, 1e6, 1e7]
    desired_y_values = [2,4,8,16,32,64,128]
    
    last_mesh = None
    
    # 2) Loop over each method
    for ax, (method_name, df) in zip(axes.flat, method_data):
        pivoted = df.pivot_table(
            values='throughput',
            index='splitSize',
            columns='N',
            aggfunc='mean'
        )
        
        # Sort so X, Y ascend
        pivoted = pivoted.sort_index()
        pivoted = pivoted[pivoted.columns.sort_values()]
        
        X_vals = pivoted.columns.values
        Y_vals = pivoted.index.values
        Z = pivoted.values
        
        X_edges = edges_from_centers(X_vals)
        Y_edges = edges_from_centers(Y_vals)
        
        mesh = ax.pcolormesh(
            X_edges,
            Y_edges,
            Z,
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            rasterized=True
        )
        ax.set_facecolor('black')  # so empty region is black
        last_mesh = mesh
        
        # 3) Set log-scale on both axes
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        
        # 4a) X-axis ticks at powers of 10 only if they exist in X_vals
        x_tick_positions = []
        x_tick_labels = []
        for val in desired_x_values:
            idx_closest = np.argmin(np.abs(X_vals - val))
            x_tick_positions.append(X_vals[idx_closest])
            exponent = int(round(np.log10(val)))
            x_tick_labels.append(fr"$10^{{{exponent}}}$")
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, rotation=45)
        
        # 4b) Y-axis ticks at powers of 10 only if they exist in Y_vals
        y_tick_positions = []
        y_tick_labels = []
        for val in desired_y_values:
            if val in Y_vals:  # Only if exactly in the dataset
                y_tick_positions.append(val)
                exponent = int(round(np.log10(val)))
                y_tick_labels.append(val)
        print(y_tick_labels)
        print(y_tick_positions)
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        # Turn off scientific notation on default ticks
        # formatter = mticker.ScalarFormatter()
        # formatter.set_scientific(False)
        # formatter.set_useOffset(False)
        # ax.yaxis.set_major_formatter(formatter)
        
        ax.set_title(method_name)
        ax.set_ylabel("Split Size")
        ax.set_xlabel("Amount of Items (N)")

    # 5) Adjust subplot spacing
    fig.subplots_adjust(
        bottom=0.08,
        top=0.92,
        wspace=0.3,
        hspace=0.6
    )

    # 6) One colorbar for all subplots
    fig.colorbar(last_mesh, ax=axes.ravel().tolist(), label='Billion packs / Second')

    # 7) Save + Show
    plt.savefig('psa_inline_splitpack_splitSize_heatmap.pdf', format='pdf')
    plt.show()

# --------------------------------------
# Call the function to generate & show
plot_all_methods(sp1, sp2, sp4, sp8, sp16, sp32)
