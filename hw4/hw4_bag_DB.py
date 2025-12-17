import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load diabetes dataset
data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)

# Convert continuous target to binary classification (above/below median)
median_val = np.median(data[:, -1])
data[:, -1] = (data[:, -1] > median_val).astype(int)

# Split data: first 25% for training, rest for testing
num_train = int(0.25 * n)
num_test = n - num_train

sample_train = data[0:num_train, 0:-1]
label_train = data[0:num_train, -1]
sample_test = data[num_train:, 0:-1]
label_test = data[num_train:, -1]

# K-means implementation
def kmeans(X, k, max_iter=300):
    n_samples = X.shape[0]
    np.random.seed(42)
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx].copy()
    
    for iteration in range(max_iter):
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.sum((X - centroids[j])**2, axis=1)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros((k, X.shape[1]))
        for j in range(k):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                new_centroids[j] = np.mean(points_in_cluster, axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels

# BagK: generate m subsets using k-means clustering
def bagk_ensemble(X_train, y_train, X_test, m):
    # Run k-means with k=m clusters
    labels = kmeans(X_train, m)
    
    # Get subset sizes
    subset_sizes = []
    models = []
    
    for i in range(m):
        cluster_indices = np.where(labels == i)[0]
        subset_sizes.append(len(cluster_indices))
        
        # Train model on this cluster
        X_subset = X_train[cluster_indices]
        y_subset = y_train[cluster_indices]
        
        model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
        model.fit(X_subset, y_subset)
        models.append(model)
    
    # Ensemble prediction: majority vote
    predictions = np.zeros((len(X_test), m))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_test)
    
    # Majority vote
    final_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 
                                      axis=1, arr=predictions)
    
    return final_pred, subset_sizes

# Bagging: generate m bootstrap samples with specified sizes
def bagging_ensemble(X_train, y_train, X_test, m, subset_sizes):
    models = []
    np.random.seed(123)
    
    for i in range(m):
        # Bootstrap sample of size subset_sizes[i]
        indices = np.random.choice(len(X_train), size=subset_sizes[i], replace=True)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        
        # Train model on bootstrap sample
        model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
        model.fit(X_subset, y_subset)
        models.append(model)
    
    # Ensemble prediction: majority vote
    predictions = np.zeros((len(X_test), m))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_test)
    
    # Majority vote
    final_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 
                                      axis=1, arr=predictions)
    
    return final_pred

# Test different values of m
m_values = [2, 3, 5, 7, 10]
er_bagk = []
er_bagging = []

for m in m_values:
    # Run BagK first to get subset sizes
    pred_bagk, subset_sizes = bagk_ensemble(sample_train, label_train, sample_test, m)
    error_bagk = 1 - accuracy_score(label_test, pred_bagk)
    er_bagk.append(error_bagk)
    
    # Run Bagging with same subset sizes
    pred_bagging = bagging_ensemble(sample_train, label_train, sample_test, m, subset_sizes)
    error_bagging = 1 - accuracy_score(label_test, pred_bagging)
    er_bagging.append(error_bagging)
    
    print(f'm={m}, BagK Error={error_bagk:.4f}, Bagging Error={error_bagging:.4f}')

# Visualization
plt.figure(figsize=(10, 8))
plt.plot(m_values, er_bagk, marker='o', markersize=12, fillstyle='none',
         linewidth=2, linestyle='-', label='BagK Error')
plt.plot(m_values, er_bagging, marker='v', markersize=12, fillstyle='none',
         linewidth=2, linestyle='-', label='Bagging Error')
plt.xlabel('Ensemble Size (m)')
plt.ylabel('Classification Error')
plt.title('Ensemble Error versus m')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(m_values)
plt.show()