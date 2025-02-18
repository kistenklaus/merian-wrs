import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

rmse_curves = pd.read_csv("./prefix_sum_benchmark.csv")

memcpy_limit = pd.read_csv("./memcpy_benchmark.csv")

n = rmse_curves["element_count"];
for col in rmse_curves.columns:
    if col != "element_count":
        plt.plot(n, rmse_curves[col],'-', linewidth=1,label=col)


plt.fill_between(memcpy_limit["element_count"], memcpy_limit["memcpy"], alpha=0.2,label="vkCmdCopyBuffer");
        
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Amount of elements");
plt.ylabel("Latency [ms]");
plt.title("Prefix Sum benchmark");
plt.legend()
plt.grid()
plt.show()

