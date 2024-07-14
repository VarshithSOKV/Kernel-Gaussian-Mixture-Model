import numpy as np
import pickle
import h5py

file_path1 = "data1.pkl"
file_path2 = "data2.pkl"

with open(file_path1, 'rb') as f:
    cluster_labels_normal = pickle.load(f)

with open(file_path2, 'rb') as f:
    cluster_labels_kernel = pickle.load(f)

data_key = 'data'
target_key = 'target'
with h5py.File('./usps.h5','r') as hf:
    train = hf.get('train')
    X_tr = train.get(data_key)[:]
    y_tr = train.get(target_key)[:]
    test = hf.get('test')
    X_te = test.get(data_key)[:]
    y_te = test.get(target_key)[:]

one_gmm = np.arange(300)[cluster_labels_normal == 0]
two_gmm = np.arange(300)[cluster_labels_normal == 1]
three_gmm = np.arange(300)[cluster_labels_normal == 2]

one_kgmm = np.arange(300)[cluster_labels_kernel == 0]
two_kgmm = np.arange(300)[cluster_labels_kernel == 1]
three_kgmm = np.arange(300)[cluster_labels_kernel == 2]

wrong1 = np.sum([two_gmm >= 100]) + np.sum([three_gmm < 200]) + one_gmm.shape[0] - np.sum(np.array([one_gmm >= 100])*np.array([one_gmm < 200]))
wrong2 = np.sum([two_kgmm >= 100]) + np.sum([three_kgmm < 200]) + one_gmm.shape[0] - np.sum(np.array([one_kgmm >= 100])*np.array([one_kgmm < 200]))

print(one_gmm.shape, two_gmm.shape, three_gmm.shape)
print(np.sum([two_gmm >= 100]), np.sum([three_gmm < 200]), one_gmm.shape[0] - np.sum(np.array([one_gmm >= 100])*np.array([one_gmm < 200])))
print(one_kgmm.shape, two_kgmm.shape, three_kgmm.shape)
print(np.sum([two_kgmm >= 100]), np.sum([three_kgmm < 200]), one_gmm.shape[0] - np.sum(np.array([one_kgmm >= 100])*np.array([one_kgmm < 200])))

# counts1 = np.unique(cluster_labels_normal, return_counts=True)[1]
print("miss rate of gmm : ", wrong1/3)
print("miss rate of kgmm : ", wrong2/3)