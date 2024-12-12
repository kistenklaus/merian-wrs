import pandas as pd;
import matplotlib.pyplot as plt;

rmse_curves = pd.read_csv("./psa_reference_rmse.csv")


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
