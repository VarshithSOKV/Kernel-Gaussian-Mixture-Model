import pickle
import numpy as np

with open("shkgmm", 'rb') as f:
    s_kgmm = pickle.load(f)

with open("shgmm", 'rb') as f:
    s_gmm = pickle.load(f)

s_gmm = s_gmm.reshape(-1)
s_kgmm = s_kgmm.reshape(-1)

for i in range(s_gmm.shape[0]):
    if np.isnan(s_gmm[i]):
        s_gmm[i] = 0

print(np.mean(s_kgmm), np.std(s_kgmm))
print(np.mean(s_gmm), np.std(s_gmm))

print(np.max(s_gmm), np.min(s_gmm))
print(np.max(s_kgmm), np.min(s_kgmm))