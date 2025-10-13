import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = data[0:num_train,0:-1]
sample_test = data[n-num_test:,0:-1]
label_train = data[0:num_train,-1]
label_test = data[n-num_test:,-1]

# pick lamda for KRR/AKRR
lamda = .001
# pick gamma for RBF kernel (in both KRR/AKRR)
gamma = .1

# baseline KRR from scikit
model = KernelRidge(alpha=lamda, kernel = 'rbf', gamma = gamma)
model.fit(sample_train,label_train)
label_test_pred = model.predict(sample_test)
er_krr = mean_squared_error(label_test, label_test_pred)

# now pick at least five values for m
m_values = [10, 20, 30, 40, 50]
er_base = []
er_yours = []

for m in m_values:
    # baseline performance, fixed across m
    er_base.append(er_krr)
    
    # AKRR Implementation
    # Step 1: Select first m training samples as inducing points
    inducing_points = sample_train[:m, :]  # first m samples
    inducing_labels = label_train[:m]
    
    # Step 2: Compute kernel matrices
    # K_mm: kernel between inducing points (m x m)
    K_mm = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            diff = inducing_points[i] - inducing_points[j]
            K_mm[i, j] = np.exp(-gamma * np.sum(diff**2))
    
    # K_nm: kernel between all training points and inducing points (n x m)
    K_nm = np.zeros((num_train, m))
    for i in range(num_train):
        for j in range(m):
            diff = sample_train[i] - inducing_points[j]
            K_nm[i, j] = np.exp(-gamma * np.sum(diff**2))
    
    # K_tm: kernel between test points and inducing points (num_test x m)
    K_tm = np.zeros((num_test, m))
    for i in range(num_test):
        for j in range(m):
            diff = sample_test[i] - inducing_points[j]
            K_tm[i, j] = np.exp(-gamma * np.sum(diff**2))
    
    # Step 3: Solve AKRR optimization
    # B= (K_nm^T K_nm + λmI)^(-1) K_nm^T y_train
    A = K_nm.T @ K_nm + lamda * m * np.eye(m)
    b = K_nm.T @ label_train
    beta_tilde = np.linalg.solve(A, b)
    
    # Step 4: Make predictions
    # y_pred = K_tm B
    label_test_pred_akrr = K_tm @ beta_tilde
    
    # Step 5: Calculate and store MSE
    er_yours.append(mean_squared_error(label_test, label_test_pred_akrr))

# the following code will plot Figure 1.
plt.figure()
plt.plot(m_values, er_base, '--o', label='Scikit Error')
plt.plot(m_values, er_yours, '-s', label='My Error')
plt.xlabel('m')
plt.ylabel('MSE')
plt.legend()
plt.title('KRR vs AKRR Performance')
plt.grid(True, alpha=0.3)
plt.show()

print(f"KRR MSE: {er_krr:.6f}")
print("AKRR MSE values:")
for i, (m, mse) in enumerate(zip(m_values, er_yours)):
    print(f"m = {m}: MSE = {mse:.6f}")