import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "wrs_benchmark.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

def plot_total_latency_for_N(N, df):
    # Find the nearest available N
    unique_N = df['N'].unique()
    nearest_N = unique_N[np.argmin(np.abs(unique_N - N))]

    print(f"Using nearest N = {nearest_N} (requested N = {N})")

    # Filter data for the selected N
    selected_df = df[df['N'] == nearest_N]

    # Get unique methods
    methods = selected_df['method'].unique()

    plt.figure(figsize=(10, 6))

    for method in methods:
        method_df = selected_df[selected_df['method'] == method]
        
        # Sort by S for smooth plotting
        method_df = method_df.sort_values(by="S")
        
        # Plot results
        plt.plot(method_df['S'], method_df['total_latency'], label=f"{method}")

    plt.xlabel("Samples Drawn (S)")
    plt.ylabel("Total Latency")
    plt.xscale("log");
    plt.title(f"Total Latency vs Samples Drawn for Nearest N={nearest_N}")
    plt.legend()
    plt.grid(True)

# Example usage
N_target = (1 << 24)  # Change this to your desired number of weights
plot_total_latency_for_N(N_target, df)

N_target = (1 << 21)  # Change this to your desired number of weights
plot_total_latency_for_N(N_target, df)

N_target = (1 << 16)  # Change this to your desired number of weights
plot_total_latency_for_N(N_target, df)

plt.show()
