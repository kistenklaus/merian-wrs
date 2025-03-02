import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV
data = pd.read_csv("block_scan_scoreboard.csv")

# Sort data by 'throughput' in ascending order
data = data.sort_values('throughput', ascending=False)

# Create a horizontal bar plot
plt.figure(figsize=(10, 5))
plt.barh(data['method'], data['throughput'], color='skyblue')
plt.xlabel('Throughput')
plt.xlim((250,500))

# Show the plot
plt.show()
