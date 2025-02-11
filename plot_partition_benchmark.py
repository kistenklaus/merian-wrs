import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

rmse_curves = pd.read_csv("./partition_benchmark.csv")


for col in rmse_curves.columns:
    if col != "element_count":
        plt.plot(rmse_curves["element_count"], rmse_curves[col],'-', linewidth=1,label=col)
        
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Amount of elements");
plt.ylabel("Latency [ms]");
plt.title("Partition benchmark");
plt.legend()
plt.grid()
plt.show()

