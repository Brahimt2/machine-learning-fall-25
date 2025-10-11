import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

num_train = int(0.75*n)
num_test = int(0.25*n)

sample_train = data[0:num_train,0:-1]
sample_test = data[n-num_test:,0:-1]
label_train = data[0:num_train,-1]
label_test = data[n-num_test:,-1]

# baseline: Logistic Regression from scikit 
model = LogisticRegression(penalty=None, C=10)
model.fit(sample_train,label_train)
label_test_pred = model.predict(sample_test)
er0 = 1 - accuracy_score(label_test, label_test_pred)

# number of model updates
# you can decide the length
num_iter = 20

# initialize beta
beta_grad = np.zeros(sample_train.shape[1])
beta_newt = np.zeros(sample_train.shape[1])

# start model update 
er_base = []
er_yours_grad = []
er_yours_newt = []
for i in range(num_iter):
  
    # baseline performance
    er_base.append(er0) 
    
    # now, implement your LR with gradient descend
    # p = [p1, ..., pn]^T
    p_vec = np.exp(-sample_train @ beta_grad) / (1 + np.exp(-sample_train @ beta_grad))  
    # X^T (p - y)
    grad_vec = sample_train.T @ (p_vec - label_train) 
    # Update beta
    beta_grad = beta_grad - 0.3 * grad_vec
    # Predict on test set using updated beta
    p_test = np.exp(-sample_test @ beta_grad) / (1 + np.exp(-sample_test @ beta_grad))
    label_test_pred_grad = (p_test >= 0.5).astype(int)
    er_yours_grad.append(1 - accuracy_score(label_test, label_test_pred_grad))
    
    # now, implement your LR with Newton's method
    p_vec_newt = np.exp(-sample_train @ beta_newt) / (1 + np.exp(-sample_train @ beta_newt))
    # X^T (p - y)
    grad_vec_newt = sample_train.T @ (p_vec_newt - label_train)
    # W = diag(p * (1-p))
    W = np.diag(p_vec_newt * (1 - p_vec_newt)) 
    # - X^T W X
    H = -sample_train.T @ W @ sample_train
    # Update beta using Newton's method
    beta_newt = beta_newt - np.linalg.inv(H) @ grad_vec_newt

    # Predict on test set using updated beta
    p_test_newt = np.exp(-sample_test @ beta_newt) / (1 + np.exp(-sample_test @ beta_newt))
    label_test_pred_newt = (p_test_newt >= 0.5).astype(int)
    er_yours_newt.append(1 - accuracy_score(label_test, label_test_pred_newt))


# Plot Figure 1
plt.figure()
plt.plot(er_base, marker='o', markersize = 12, fillstyle = 'none', linewidth=2, linestyle='-', label='Scikit Error')
plt.plot(er_yours_grad, marker='o', markersize = 12, fillstyle = 'none', linewidth=2, linestyle='-', label='My Error (Gradient)')
plt.plot(er_yours_newt, marker='o', markersize = 12, fillstyle = 'none', linewidth=2, linestyle='-', label='My Error (Newton)')
plt.xlabel('Number of Model Updates')
plt.ylabel('Classification Error')
plt.legend()
plt.show()
