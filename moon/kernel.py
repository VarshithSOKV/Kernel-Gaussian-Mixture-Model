import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
import math
import pickle

# Define the file path
file_path = "data.pkl"
file_pathP = "Prob.pkl"

class KernelGMM:

    def __init__(self, n_components=3, kernelFunction = "polynomial", initial_membership=np.eye(3), d_phi = 3):
        self.numclasses = n_components
        self.kernelFunction = kernelFunction
        self.membership = initial_membership        # size = (n_class, n_datapoints)
        self.class_weights = np.mean(initial_membership, axis = 1, keepdims=True)   # size = (n_class,1)
        self.W = np.zeros((n_components, n_components))
        self.d_phi = d_phi
    
    def radial(self,X):
        x, y = X[:, 0], X[:, 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return np.matmul(r,r.T) + np.matmul(theta,theta.T)
        
        # Compute the custom radial kernel
        # K = r1[:, np.newaxis] * r2[np.newaxis, :] + theta1[:, np.newaxis] * theta2[np.newaxis, :]
        
        # return K
    
    def gaussian(self, X, gamma):
        n1, n2 = X.shape
        x1 = X.reshape(1, n1, n2)
        x2 = X.reshape(n1, 1, n2)
        temp = (x1-x2)**2
        return np.exp(-gamma * np.sum(temp, axis = 2))

    
    def polynomial(self, X, degree):
        n = len(X)
        matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = (1 + np.dot(X[i], X[j]))

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
        # Doubt : can replace "np.matmul(np.ones((n2,1))" with "np.matmul(np.ones((n2,1)) / n2"??
        return np.array(W_prime)

    def computeKernelMatrix(self, X):
        if self.kernelFunction == "gaussian":
            KernelMatrix = self.gaussian(X, 7)

        if self.kernelFunction == "polynomial":
            KernelMatrix = self.polynomial(X, 2)

        if self.kernelFunction == "sigmoid":
            KernelMatrix = self.sigmoid(X, 1)

        if self.kernelFunction == "radial":
            KernelMatrix = self.radial(X)

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

            A = - (6e5) * (0.5) * (np.sum(((y.reshape(-1, 1))**2 / (k_eigenvalues.reshape(-1, 1)))))
            B = (-0.5 * np.sum(np.log(k_eigenvalues)) - (self.d_phi/2) * np.log(2 * np.pi))
            # C = - ((num_samples - self.d_phi)/2) * (np.log(2 * np.pi * Rho))
            D =  - (epsilon_sq/(2*Rho))

            # print("A : ", A)
            # print("B : ", B)
            # # # print("C : ", C)
            # print("D : ", D)

            G[i] = np.exp(A + B + D)

            # print(" G[0] : ", G[0])

            # G[np.isposinf(G) | np.isnan(G)] = 1e60
            # G[np.isneginf(G)] = 0

        return G


    def fit(self, X):
        num_samples = len(X)
        threshold = 0.01
        max_iter = 50
        iteration = 0

        KernelMatrix = self.computeKernelMatrix(X)

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
                k_eigenvectors = k_eigenvectors/(np.sqrt(k_eigenvectors.shape[0]))

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
        cluster_labels_kernel = np.zeros(num_samples)
        for i in range(num_samples):
            cluster_labels_kernel[i] = np.argmax(self.membership[:, i]**2)
        
        return cluster_labels_kernel
    
    def dis(self, X):
        temp = self.computeKernelMatrix(X)
        temp2 = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                temp2[i][j] = np.sqrt(temp[i][i] + temp[j][j] -2*temp[i][j])
        print("kgmm dist : ", temp2)
        return temp2
    
    def sh(self, X, cluster_labels):
        class1 = np.arange(X.shape[0])[cluster_labels == 0]
        class2 = np.arange(X.shape[0])[cluster_labels == 1]
        print(class1, class2)
        a = np.zeros((X.shape[0], 1))
        b = np.zeros((X.shape[0], 1))
        dist = self.dis(X)
        for i, index in enumerate(class1):
            a[index] = 0
            for j, index2 in enumerate(class1):
                a[index] += dist[index][index2]
            a[index] /= class1.shape[0]
        for i, index in enumerate(class2):
            a[index] = 0
            for j, index2 in enumerate(class2):
                a[index] += dist[index][index2]
            a[index] /= class2.shape[0]

        for i, index in enumerate(class1):
            b[index] = 0
            for j, index2 in enumerate(class2):
                b[index] += dist[index][index2]
            b[index] /= class2.shape[0]
        for i, index in enumerate(class2):
            b[index] = 0
            for j, index2 in enumerate(class1):
                b[index] += dist[index][index2]
            b[index] /= class1.shape[0]

        temp3 = np.concatenate((a, b), axis = 1)
        s = (b - a)/(np.max(temp3, axis = 1).reshape(-1,1))
        return s

def gaussian_sh(X, cluster_labels):
    class1 = np.arange(X.shape[0])[cluster_labels == 0]
    class2 = np.arange(X.shape[0])[cluster_labels == 1]
    a = np.zeros((X.shape[0], 1))
    b = np.zeros((X.shape[0], 1))
    dist = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            dist[i][j] = np.sqrt(np.sum((X[i] - X[j])**2))
    print("gmm dist : ", dist)
    for i, index in enumerate(class1):
        a[index] = 0
        for j, index2 in enumerate(class1):
            a[index] += dist[index][index2]
        a[index] /= class1.shape[0]
    for i, index in enumerate(class2):
        a[index] = 0
        for j, index2 in enumerate(class2):
            a[index] += dist[index][index2]
        a[index] /= class2.shape[0]

    for i, index in enumerate(class1):
        b[index] = 0
        for j, index2 in enumerate(class2):
            b[index] += dist[index][index2]
        b[index] /= class2.shape[0]
    for i, index in enumerate(class2):
        b[index] = 0
        for j, index2 in enumerate(class1):
            b[index] += dist[index][index2]
        b[index] /= class1.shape[0]

    temp3 = np.concatenate((a, b), axis = 1)
    s = (b - a)/(np.max(temp3, axis = 1).reshape(-1,1))
    return s

X, y = make_moons(n_samples=300, noise=0.02, random_state=417)
num_samples = len(X)

with open(file_path, 'rb') as f:
    X = pickle.load(f)

num_classes = 2
d_phi = 5

# Fit normal GMM
gmm_normal = GaussianMixture(n_components=num_classes)
gmm_normal.fit(X)
cluster_labels_normal = gmm_normal.predict(X)

# Pickle the array into a file

P = np.zeros((num_classes, len(X)))
for i in range(num_classes):
    P[i][i] = 1
for i in range(num_samples-num_classes):
    temp = np.random.randint(0, num_classes)
    P[temp][i+num_classes] = 1

for i in range(num_samples):
    t = np.random.random(1)
    P[0][i] = t[0]
    P[1][i] = 1 - t[0]

with open(file_pathP, 'rb') as f:
    P = pickle.load(f)

small_P = np.zeros(num_samples)
for i in range(num_samples):
    small_P[i] = np.argmax(P[:,i])

kgmm_polynomial = KernelGMM(n_components=num_classes, kernelFunction = "gaussian", initial_membership= P, d_phi=d_phi)
kgmm_polynomial.fit(X)

cluster_labels_kernel = kgmm_polynomial.predict()
# Segregating poimts based on class
class1 = X[cluster_labels_kernel == 0]
class2 = X[cluster_labels_kernel == 1]

# For a given data point in class1 we need to find sihoutte distance
s_kgmm = kgmm_polynomial.sh(X, cluster_labels_kernel)
s_gmm = gaussian_sh(X, cluster_labels_normal)

with open("shkgmm", 'wb') as f:
    pickle.dump(s_kgmm, f)

with open("shgmm", 'wb') as f:
    pickle.dump(s_gmm, f)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels_normal, cmap='viridis', s=50, alpha=0.7)
plt.title('Normal Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=small_P, cmap='viridis', s=50, alpha=0.7)
plt.title('Original')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')

plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels_kernel, cmap='viridis', s=50, alpha=0.7)
plt.title('Kernel Gaussian Mixture Model (Polynomial Kernel)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.savefig("myImage")