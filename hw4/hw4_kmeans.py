import numpy as np
import matplotlib.pyplot as plt

# Load diabetes dataset
data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)

# Extract features (exclude label column)
X = data[:, 0:-1]

# K-means implementation from scratch
def kmeans(X, k, max_iter=300):
    """
    K-means clustering algorithm
    X: data matrix (n_samples, n_features)
    k: number of clusters
    max_iter: maximum iterations
    Returns: cluster assignments, centroids
    """
    n_samples = X.shape[0]
    
    # Initialize centroids: randomly select k data points
    np.random.seed(42)
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx].copy()
    
    # Iterate until convergence
    for iteration in range(max_iter):
        # Assignment step: assign each point to nearest centroid
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.sum((X - centroids[j])**2, axis=1)
        
        labels = np.argmin(distances, axis=1)
        
        # Update step: recompute centroids
        new_centroids = np.zeros((k, X.shape[1]))
        for j in range(k):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = np.mean(points_in_cluster, axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Run K-means with k=3
k = 2
labels, centroids = kmeans(X, k)

# Apply PCA for 2D visualization
mean = np.mean(X, axis=0)
X_centered = X - mean
cov_matrix = (X_centered.T @ X_centered) / n
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]
components = eigenvectors[:, 0:2]
X_pca = X_centered @ components

# Visualization
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(k):
    cluster_points = X_pca[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               c=colors[i], label=f'Cluster {i+1}',
               alpha=0.6, edgecolors='k', linewidth=0.5)

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Visualization of Clustering Result (k = 2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()