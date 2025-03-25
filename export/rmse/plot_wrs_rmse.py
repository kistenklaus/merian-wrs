import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


dirname = os.path.dirname(__file__)
bench = pd.read_csv(os.path.join(dirname, "./curve.csv"))

print(bench["group"].unique())

print("N =", bench["N"].unique())

print(bench)

plt.rcParams.update({'font.size': 12})

def plotMe(method, label, color):
    group = bench[bench["group"] == method]

    plt.plot(group["S"], group["rmse"], "-", label=label, color=color)

plt.figure(figsize=(8, 3))

plotMe("ITS-0", "its-baseline", "tab:blue")
plotMe("ITS-128", "its-coop", "tab:orange")
plotMe("Cutpoint", "cutpoint", "tab:green")
plotMe("PSA2-0", "psa-baseline", "tab:red")
plotMe("PSA2-128", "psa-sectioned", "tab:brown")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Number of Samples")
plt.ylabel("RMSE")
plt.ylim((1e-9,1e-5))
plt.xscale("log")
plt.yscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig("wrs_rmse_psa_e8.pdf", format="pdf")


plt.show()



