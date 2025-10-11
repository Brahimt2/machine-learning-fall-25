import numpy as np
from sklearn import linear_model
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


# pick 5 candidate values that can 
# more comprehensively reflect the 
# the impact of lambda on performance 
lamda_list = [0.01, .1, 1, 1.2, 1.6, 2]  

er_base = []
er_yours = []


# the following arrays store model sparsity 
sp_base = []
sp_yours = []


for lamda in lamda_list:
    
    # baseline implementation using scikit-learn 
    model = linear_model.Lasso(lamda)
    model.fit(sample_train, label_train)
    label_train_pred = model.predict(sample_train)
    label_test_pred = model.predict(sample_test)
    er_base.append(mean_squared_error(label_test, label_test_pred))
    # in addition to measure error, also measure sparsity of your model 
    # sparsity is defined as # zero elements / # elements in the model 
    sp = sum(model.coef_==0)/len(model.coef_)
    sp_base.append(sp)
    
    
    # now, implement ridge regression 
    m, d = sample_train.shape
    
    # Copy the data
    X = sample_train.copy()
    y = label_train.copy()
    
    # Center y
    y_mean = np.mean(y)
    y = y - y_mean
    
    # Standardize X
    X_means = np.mean(X, axis=0)
    X_stds = np.std(X, axis=0)
    X_stds[X_stds == 0] = 1
    X = (X - X_means) / X_stds
    
    # Initialize beta
    beta = np.zeros(d)
    
    # Precompute X squared norms for each feature
    X_squared_norms = (X ** 2).sum(axis=0)
    
    # Coordinate descent
    max_iter = 1000
    tol = 1e-4
    
    # The key: sklearn uses alpha = lambda / n_samples in their objective
    # So the threshold for soft thresholding is n_samples * alpha = lambda
    threshold = m * lamda
    
    for _ in range(max_iter):
        beta_old = beta.copy()
        
        for j in range(d):
            # Skip if feature has no variance
            if X_squared_norms[j] == 0:
                beta[j] = 0
                continue
                
            # Compute residual
            residual = y - X.dot(beta) + X[:, j] * beta[j]
            
            # Compute rho
            rho = X[:, j].dot(residual)
            
            # Soft thresholding
            if rho < -threshold:
                beta[j] = (rho + threshold) / X_squared_norms[j]
            elif rho > threshold:
                beta[j] = (rho - threshold) / X_squared_norms[j]
            else:
                beta[j] = 0
        
        # Check convergence
        if np.linalg.norm(beta - beta_old, ord=np.inf) < tol:
            break
    
    # Scale beta back to original feature scale
    beta = beta / X_stds
    
    # Compute intercept
    intercept = y_mean - np.dot(X_means, beta)
    
    # Make predictions
    label_test_pred = sample_test.dot(beta) + intercept
    
    # store your MSE
    er_yours.append(mean_squared_error(label_test, label_test_pred))
    
    # also compute and store your model sparsity 
    sp = np.sum(np.abs(beta) < 1e-10) / len(beta)
    sp_yours.append(sp)


# the following code will plot two figures. 
# one is error versus lambda (two curves) 
# another is sparsity versus lambda (two curves)
plt.figure()
plt.plot(lamda_list,er_base, '--o', label='Scikit Error')
plt.plot(lamda_list,er_yours, label='My Error')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.legend()
plt.show()


plt.figure()
plt.plot(lamda_list,sp_base, '--o', label='Scikit Sparsity')
plt.plot(lamda_list,sp_yours, label='My Sparsity')
plt.xlabel('Lambda')
plt.ylabel('Sparsity')
plt.legend()
plt.show()