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
    """
    n_samples = X.shape[0]
    
    # Initialize centroids: randomly select k data points
    np.random.seed(42)
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx].copy()
    
    # Iterate until convergence
    for iteration in range(max_iter):
        # Assignment step
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.sum((X - centroids[j])**2, axis=1)
        
        labels = np.argmin(distances, axis=1)
        
        # Update step
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

# Davies-Bouldin Index implementation
def davies_bouldin_index(X, labels, centroids):
    """
    Compute Davies-Bouldin index for clustering quality
    Lower values indicate better clustering
    """
    k = len(centroids)
    
    # Compute within-cluster scatter for each cluster
    scatter = np.zeros(k)
    for i in range(k):
        points_in_cluster = X[labels == i]
        if len(points_in_cluster) > 0:
            distances = np.sqrt(np.sum((points_in_cluster - centroids[i])**2, axis=1))
            scatter[i] = np.mean(distances)
    
    # Compute Davies-Bouldin index
    db_values = np.zeros(k)
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j:
                # Distance between centroids
                separation = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
                if separation > 0:
                    ratio = (scatter[i] + scatter[j]) / separation
                    max_ratio = max(max_ratio, ratio)
        db_values[i] = max_ratio
    
    db_index = np.mean(db_values)
    return db_index

# Test different values of k
k_values = [2, 3, 5, 7, 9]
db_indices = []

for k in k_values:
    labels, centroids = kmeans(X, k)
    db_index = davies_bouldin_index(X, labels, centroids)
    db_indices.append(db_index)
    print(f'k={k}, DB Index={db_index:.4f}')

# Visualization
plt.figure(figsize=(10, 8))
plt.plot(k_values, db_indices, marker='o', markersize=12, 
         fillstyle='none', linewidth=2, linestyle='-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index versus k')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.show()