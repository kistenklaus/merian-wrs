import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

rmse_curves = pd.read_csv("./its_benchmark.csv")


for col in rmse_curves.columns:
    if col != "sample_size":
        plt.plot(rmse_curves["sample_size"], rmse_curves[col],'-', linewidth=1,label=col)
        
plt.xscale("log")
plt.xlabel("Sample size");
plt.ylabel("Latency [ms]");
plt.title("Inverse Transform Sampling Latency");
plt.legend()
plt.grid()
plt.show()

