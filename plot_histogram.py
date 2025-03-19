import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

hist = pd.read_csv("./histogram_PSA2-0.csv")

hist["diff"] = hist["observed"] - hist["expected"]

print(hist)

print(len(hist[hist["observed"] == 0]))
print(len(hist[hist["expected"] < 1]))
print(np.median(hist["expected"]))

# plt.plot(hist["X"], hist["observed"], ".", alpha=0.01)

# plt.plot(hist["X"], hist["expected"], "x")

# plt.plot(hist["X"], hist["diff"], "x")
# plt.grid();
#
# plt.show()



