import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

rmse_curves = pd.read_csv("./psa_rmse_curve.csv")


sample_sizes = rmse_curves["sample_size"]
theoretical_rmse = 1 / np.sqrt(sample_sizes) 
plt.plot(sample_sizes, theoretical_rmse, label="1/sqrt(S)");

for col in rmse_curves.columns:
    if col != "sample_size":
        plt.plot(rmse_curves["sample_size"], rmse_curves[col],'--', linewidth=1,label=col)
        


plt.xscale("log")
plt.yscale("log")
plt.xlabel("Sample size");
plt.ylabel("RMSE");
plt.title("RMSE of PSA");
plt.legend()
plt.grid()
plt.show()

