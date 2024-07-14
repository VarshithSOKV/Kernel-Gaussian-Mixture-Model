import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import random
import pickle
import h5py

# Define the file path
file_pathP = "Prob1.pkl"
file_path1 = "data1.pkl"
file_path2 = "data2.pkl"

class KernelGMM:

    def __init__(self, n_components=3, kernelFunction = "polynomial", initial_membership=np.eye(3), d_phi = 3):
        self.numclasses = n_components
        self.kernelFunction = kernelFunction
        self.membership = initial_membership        # size = (n_class, n_datapoints)
        self.class_weights = np.mean(initial_membership, axis = 1, keepdims=True)   # size = (n_class,1)
        self.W = np.zeros((n_components, n_components))
        self.d_phi = d_phi

    def gaussian(self, X, gamma):
        n1, n2 = X.shape
        x1 = X.reshape(1, n1, n2)
        x2 = X.reshape(n1, 1, n2)
        temp = (x1-x2)**2
        return np.exp(-gamma * np.sum(temp, axis = 2))
    
    def polynomial(self, X, degree):
        n = len(X)
        matrix = np.zeros((n,n))
        matrix = 1 + (np.matmul(X, X.T))

        return matrix**degree
    
    def sigmoid(self, X, sigma):
        n = len(X)
        matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = np.exp(- (1/ (2*sigma**2)) * np.linalg.norm(X[i]-X[j]))

        return matrix

    def computeclass_weights(self, P):
        return np.mean(P, axis = 1)
    
    def computeMembership(self, P):
        return np.sqrt(P / (np.sum(P, axis = 0)).reshape(1,-1))

    def computeW(self, membership):
        W = []
        n1, _ = membership.shape
        for i in range(n1):
            W.append(membership[i].reshape(-1,1) @ membership[i].reshape(1,-1))
        return np.array(W)

    def computeW_prime(self, membership):
        W_prime = []
        n1,n2 = membership.shape
        for i in range(n1):
            W_prime.append(np.matmul(np.ones((n2,1)), membership[i].reshape(1,-1)))
        return np.array(W_prime)

    def computeKernelMatrix(self, X):
        if self.kernelFunction == "gaussian":
            KernelMatrix = self.gaussian(X, 0.0005)

        if self.kernelFunction == "polynomial":
            KernelMatrix = self.polynomial(X, 2)

        if self.kernelFunction == "sigmoid":
            KernelMatrix = self.sigmoid(X, 1)

        return KernelMatrix

    def computeK_Tilda(self, KernelMatrix, membership):
        W = self.computeW(membership)
        K = KernelMatrix*W
        K_tilda = np.zeros_like(K)
        n1,n2,_ = K.shape
        for i in range(n1):
            temp = np.matmul(K[i],W[i])
            K_tilda[i] = K[i] - np.matmul(W[i],K[i]) - temp + np.matmul(W[i],temp)
        return K_tilda

    def computeK_Prime_Tilda(self, KernelMatrix, membership):
        W_Prime = self.computeW_prime(membership)
        W = self.computeW(membership)
        K = KernelMatrix*W
        K_Prime = KernelMatrix*W_Prime
        K_Prime_tilda = np.zeros_like(K)
        n1,n2,_ = K.shape
        for i in range(n1):
            temp = np.matmul(K[i],W[i])
            K_Prime_tilda[i] = K_Prime[i] - np.matmul(W_Prime[i],K[i]) - np.matmul(K_Prime[i],W[i]) + np.matmul(W_Prime[i],temp)
        return K_Prime_tilda
    
    def computeG(self, k_eigenvalues, k_eigenvectors, K_Prime_Tilda, other_eigenvectors, Rho):
        num_samples = k_eigenvectors.shape[0]
        G = np.zeros(num_samples)
        y = np.zeros(d_phi)

        for i in range(num_samples):
            y = np.zeros(d_phi)
            for j in range(self.d_phi):
                y[j] = np.dot(k_eigenvectors[:, j], K_Prime_Tilda[i])
            y_other = np.zeros(num_samples-d_phi)
            for j in range(num_samples-d_phi):
                y_other[j] = np.dot(other_eigenvectors[:,j], K_Prime_Tilda[i])

            epsilon_sq = np.sum(y_other**2)

            A = - (0.01) * (0.5) * (np.sum(((y.reshape(-1, 1))**2 / (0.001 + k_eigenvalues.reshape(-1, 1)))))
            B = (0.01) * (-0.5 * np.sum(np.log(k_eigenvalues)) - (self.d_phi/2) * np.log(2 * np.pi))
            # C = - ((num_samples - self.d_phi)/2) * (np.log(2 * np.pi * Rho))
            # D =  - (epsilon_sq/(2*Rho))

            print("A : ", A)
            print("B : ", B)
            # print("C : ", C)
            # print("D : ", D)

            G[i] = np.exp(A + B)

        return G


    def fit(self, X):
        num_samples = len(X)
        threshold = 0.01
        max_iter = 50
        iteration = 0

        KernelMatrix = self.computeKernelMatrix(X)
        # print(KernelMatrix)
        # exit()

        while True:
            iteration += 1

            K_Tilda = self.computeK_Tilda(KernelMatrix, self.membership)
            K_Prime_Tilda = self.computeK_Prime_Tilda(KernelMatrix, self.membership)

            G = np.zeros_like(self.membership)
            flag = 0
            for i in range(self.numclasses):

                eigenvalues, eigenvectors = np.linalg.eigh(K_Tilda[i])
                sorted_indices = np.argsort(eigenvalues)[::-1]
                sorted_eigenvectors = eigenvectors[:, sorted_indices]
                sorted_eigenvalues = eigenvalues[sorted_indices]

                # Take the first k eigenvectors
                k_eigenvectors = sorted_eigenvectors[:, :self.d_phi]
                k_eigenvalues = sorted_eigenvalues[:self.d_phi]

                if np.sum(k_eigenvalues == 0) != 0:
                    flag = 1
                    break

                k_eigenvectors = k_eigenvectors/(np.sum(k_eigenvectors**2, axis = 0, keepdims=True))
                k_eigenvectors = k_eigenvectors/(np.sqrt(k_eigenvalues).reshape(1, -1))
                k_eigenvectors = k_eigenvectors*(np.sqrt(k_eigenvectors.shape[0]))

                other_eigenvectors = eigenvectors[:, sorted_indices][:, self.d_phi:]
                Rho = (np.trace(K_Tilda[i]) - np.sum(k_eigenvalues)) / (num_samples - self.d_phi)

                G[i] = self.computeG(k_eigenvalues, k_eigenvectors, K_Prime_Tilda[i], other_eigenvectors, Rho)

            if flag:
                print("eigenvalue zero")
                break

            for i in range(num_classes):
                G[i] = G[i] / (np.sum(G[i]))

            # update class weights
            self.class_weights = self.computeclass_weights(self.membership)
            temp = self.class_weights.reshape(-1,1) * G
            old_mem = self.membership
            self.membership = self.computeMembership(temp)
            if (np.sum((self.membership - old_mem)**2) < threshold or iteration > max_iter):
                # print(self.membership**2)
                # print(old_mem**2)
                print(np.sum((self.membership - old_mem)**2))
                break

        # pass

    def predict(self):
        num_samples = self.membership.shape[1]
        cluster_labels_kernel = np.zeros(num_samples, dtype=int)
        for i in range(num_samples):
            cluster_labels_kernel[i] = np.argmax(self.membership[:, i]**2)
        
        return cluster_labels_kernel


data_key = 'data'
target_key = 'target'
with h5py.File('./usps.h5','r') as hf:
    train = hf.get('train')
    X_tr = train.get(data_key)[:]
    y_tr = train.get(target_key)[:]
    test = hf.get('test')
    X_te = test.get(data_key)[:]
    y_te = test.get(target_key)[:]

X_zero = X_tr[y_tr == 2][:100]
X_one = X_tr[y_tr == 7][:100]
X_two = X_tr[y_tr == 8][:100]
X_train = np.concatenate((X_zero, X_one), axis=0)
X_train = np.concatenate((X_train, X_two), axis=0)

min_intensity = np.min(X_train, axis = 1)
max_intensity = np.max(X_train, axis = 1)

# X_train = 255 * X_train/((max_intensity - min_intensity).reshape(-1,1))

X = X_train


num_samples = len(X)

num_classes = 3
d_phi = 200

# Fit normal GMM
gmm_normal = GaussianMixture(n_components=num_classes)
gmm_normal.fit(X)
cluster_labels_normal = gmm_normal.predict(X)

P = np.zeros((num_classes, len(X)))
# for i in range(num_classes):
#     P[i][i] = 1
# for i in range(num_samples-num_classes):
#     temp = np.random.randint(0, num_classes)
#     P[temp][i+num_classes] = 1
# for i in range(100):
#     for j in range(num_classes):
#         P[j][j*100 + i] = 1
    # P[0][i] = 1
    # P[1][i+100] = 1
    # P[2][i+200] = 1

# initialization for membership for kgmm
for i in range(num_samples):
    t = np.random.random(3)
    t = t/np.sum(t)
    P[0][i] = t[0]
    P[1][i] = t[1]
    P[2][i] = t[2]


with open(file_pathP, 'rb') as f:
    P = pickle.load(f)

kgmm_polynomial = KernelGMM(n_components=num_classes, kernelFunction = "gaussian", initial_membership= P, d_phi=d_phi)
kgmm_polynomial.fit(X)

cluster_labels_kernel = kgmm_polynomial.predict()


with open(file_path1, 'wb') as f:
    pickle.dump(cluster_labels_normal, f)

with open(file_path2, 'wb') as f:
    pickle.dump(cluster_labels_kernel, f)

print(np.arange(X_train.shape[0])[cluster_labels_normal == 0])
print("\n")
print(np.arange(X_train.shape[0])[cluster_labels_normal == 1])
print("\n")
print(np.arange(X_train.shape[0])[cluster_labels_normal == 2])
print("yes")
print(np.arange(X_train.shape[0])[cluster_labels_kernel == 0])
print("\n")
print(np.arange(X_train.shape[0])[cluster_labels_kernel == 1])
print("\n")
print(np.arange(X_train.shape[0])[cluster_labels_kernel == 2])

print(np.unique(cluster_labels_normal, return_counts=True)[1])
print(np.unique(cluster_labels_kernel, return_counts=True)[1])