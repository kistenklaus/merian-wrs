from functools import singledispatch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from numpy._core.numerictypes import single
import pandas as pd
import os

dirname = os.path.dirname(__file__)
error = pd.read_csv(os.path.join(dirname, "./error.csv"))

property = "rel_error"

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 3))

# for method in error["method"].unique():
#     df = error[error["method"] == method]
#     plt.plot(df["N"], df[property], "-", label=method);

seq = error[error["method"] == "sequential-scan"]
kahan = error[error["method"] == "sequential-scan-kahan"]
gpu = error[error["method"] == "single-dispatch"]

plt.plot(seq["N"], seq[property], "-", label="sequential")
plt.plot(kahan["N"], kahan[property], "-", label="kahan")
plt.plot(gpu["N"], gpu[property], "-", label="single-dispatch")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# If you want to remove the bottom and left spines as well:
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.xlabel("Amount of Elements")
plt.ylabel("Relative error")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)


plt.tight_layout()
plt.savefig("scan_error.pdf", format="pdf")


plt.show()




