import numpy as np
import matplotlib.pyplot as plt

# Load diabetes dataset
data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)

# Extract features (exclude label column)
X = data[:, 0:-1]

# PCA implementation from scratch
# Step 1: Center the data
mean = np.mean(X, axis=0)
X_centered = X - mean

# Step 2: Compute covariance matrix
cov_matrix = (X_centered.T @ X_centered) / n

# Step 3: Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 4: Sort in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 5: Select top 2 components
components = eigenvectors[:, 0:2]

# Step 6: Project data onto 2D
X_pca = X_centered @ components

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], 
           c=data[:, -1], cmap='viridis', 
           alpha=0.6, edgecolors='k', linewidth=0.5)
plt.colorbar(label='Diabetes Progression')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Visualization of Diabetes Data Distribution')
plt.grid(True, alpha=0.3)
plt.show()