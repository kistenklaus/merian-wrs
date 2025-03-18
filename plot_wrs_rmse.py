import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


bench1 = pd.read_csv("./wrs_rmse_curve.csv")

bench = pd.concat([bench1])

print(bench["group"].unique())

print("N =", bench["N"].unique())

print(bench)

plt.rcParams.update({'font.size': 12})

def plotMe(method, label, color):
    group = bench[bench["group"] == method]

    plt.plot(group["S"], group["rmse"], label=label, color=color)

plt.figure(figsize=(8, 3))

# plotMe("ITS-0", "its-baseline", "tab:blue")
# plotMe("ITS-128", "its-coop", "tab:orange")
# plotMe("Cutpoint", "cutpoint", "tab:green")
plotMe("PSA2-0", "psa-baseline", "tab:blue")
plotMe("PSA2-128", "psa-sectioned", "tab:orange")

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
# plt.ylim(top=60)
plt.xscale("log")
plt.yscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig("wrs_rmse_psa.pdf", format="pdf")


plt.show()



