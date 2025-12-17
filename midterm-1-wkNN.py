import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# prepare data 
data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)
num_train = int(0.75 * n)
num_test = int(0.25 * n)
sample_train = data[:num_train, :-1]
sample_test = data[n - num_test:, :-1]
label_train = data[:num_train, -1]
label_test = data[n - num_test:, -1]

# hyper-parameter for k-NN
k_values = [1,3,5,9,15]
task4_k_values = [3, 5]     # for Task 4 evaluation

# evaluate and plot for multiple k
er_train_weighted = []
er_test_weighted = []
er_train_baseline = []
er_test_baseline = []
weights_per_k = []  # store learned weights for each k

for k in k_values:
    
    # Y_n: num_train x k matrix, where each row i contains labels of k nearest neighbors of x_i
    Y_n = np.zeros((num_train, k))
    for i in range(num_train):
        # Compute distances from training point x_i to all training points
        distances = np.linalg.norm(sample_train - sample_train[i], axis=1)
        distances[i] = np.inf  # deleteing this makes k=1 have no training error, but nothing else makes sense.
        nn_idx = np.argsort(distances)[:k]
        Y_n[i, :] = label_train[nn_idx]  # y_i = [y(i,1), ..., y(i,k)]
    
    # w = (Y_n^T Y_n)^{-1} Y_n^T Y
    w = np.linalg.inv(Y_n.T @ Y_n) @ (Y_n.T @ label_train)  # w = [w1, ..., wk]
    weights_per_k.append(w)  # store for later reporting
    
    # weighted = Y_n @ w
    f_train_weighted = Y_n @ w  # weighted kNN prediction for training
    # uniform weights = 1/k sum_j y_i,j
    f_train_baseline = np.mean(Y_n, axis=1)
    
    f_test_weighted = np.zeros(num_test)   # f(z_i) for weighted kNN
    f_test_baseline = np.zeros(num_test)   # f(z_i) for baseline
    for i in range(num_test):
        distances = np.linalg.norm(sample_train - sample_test[i], axis=1)
        nn_idx = np.argsort(distances)[:k]
        Y_z = label_train[nn_idx]  # Y_z = [y(z_i,1), ..., y(z_i,k)]
        
        # f(z_i) = sum_j w_j * y(z_i,j)
        f_test_weighted[i] = Y_z @ w
        # uniform average
        f_test_baseline[i] = np.mean(Y_z)
    
    er_train_weighted.append(mean_squared_error(label_train, f_train_weighted))
    er_test_weighted.append(mean_squared_error(label_test, f_test_weighted))
    er_train_baseline.append(mean_squared_error(label_train, f_train_baseline))
    er_test_baseline.append(mean_squared_error(label_test, f_test_baseline))

# Task 4
print("==== Task 4: Train/Test MSE for k=3 and k=5 ====")
for idx, k in enumerate(task4_k_values):
    k_idx = k_values.index(k)
    w_current = weights_per_k[k_idx]  # get correct w for current k
    print(f"k = {k}")
    print(f"Weighted kNN: Train MSE = {er_train_weighted[k_idx]:.4f}, Test MSE = {er_test_weighted[k_idx]:.4f}")
    print(f"Baseline kNN: Train MSE = {er_train_baseline[k_idx]:.4f}, Test MSE = {er_test_baseline[k_idx]:.4f}")
    print(f"Learned weights: {w_current}")
    print("------------------------------------")

plt.figure()
plt.plot(k_values, er_train_weighted, marker='o', label='Weighted kNN Train MSE')
plt.plot(k_values, er_test_weighted, marker='d', label='Weighted kNN Test MSE')
plt.plot(k_values, er_train_baseline, marker='o', linestyle='--', label='Baseline Train MSE')
plt.plot(k_values, er_test_baseline, marker='d', linestyle='--', label='Baseline Test MSE')
plt.xlabel('Value of k')
plt.ylabel('Mean Squared Error')
plt.title('Weighted kNN vs Baseline kNN: Train/Test MSE vs k')
plt.legend()
plt.grid(True)
plt.show()
