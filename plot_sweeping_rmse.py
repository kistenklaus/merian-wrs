import pandas as pd;
import matplotlib.pyplot as plt;

rmse_curves = pd.read_csv("./sweeping_reference_rmse.csv")


for col in rmse_curves.columns:
    if col != "sample_size":
        plt.plot(rmse_curves["sample_size"], rmse_curves[col],'--', linewidth=1,label=col)
        

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Sample size");
plt.ylabel("RMSE");
plt.legend()
plt.title("RMSE of Sweeping construction by Vose et.al.")
plt.grid()
plt.show()
