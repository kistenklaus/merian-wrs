import pandas as pd;
import matplotlib.pyplot as plt;

psa = pd.read_csv("./psa_reference_rmse.csv")
std = pd.read_csv("./std_rmse.csv")
swep = pd.read_csv("./sweeping_reference_rmse.csv")


plt.plot(swep["sample_size"], swep["float-1024-uniform"], '--', label="Vose<float> of 1024 weights");
# plt.plot(swep["sample_size"], swep["double-1024-uniform"], ':', label="Vose<double> of 1024 weights");
plt.plot(psa["sample_size"], psa["float-1024-uniform-32"], '--', label="PSA<float> of 1024 weights");
# plt.plot(psa["sample_size"], psa["double-1024-uniform-32"], ':', label="PSA<double> of 1024 weights");
plt.plot(std["sample_size"], std["float-1024-uniform"], '--' ,label="STL<float> of 1024 weights");
# plt.plot(std["sample_size"], std["double-1024-uniform"], ':' ,label="STL<double> of 1024 weights");


# plt.plot(swep["sample_size"], swep["float-2048-uniform"], '--', label="Vose<float> of 2048 weights");
# plt.plot(swep["sample_size"], swep["double-2048-uniform"], ':', label="Vose<double> of 2048 weights");
# plt.plot(psa["sample_size"], psa["float-2048-uniform-64"], '--', label="PSA<float> of 2048 weights");
# plt.plot(psa["sample_size"], psa["double-2048-uniform-64"], ':', label="PSA<double> of 2048 weights");
# plt.plot(std["sample_size"], std["float-2048-uniform"], '--' ,label="STL<float> of 2048 weights");
# plt.plot(std["sample_size"], std["double-2048-uniform"], ':' ,label="STL<double> of 2048 weights");
        

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Sample size");
plt.ylabel("RMSE");
plt.legend()
plt.grid()
plt.show()
