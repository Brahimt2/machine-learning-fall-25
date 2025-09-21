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
lamda = 0

# pick gamma for RBF kernel (in both KRR/AKRR) 
gamma = 0

# baseline KRR from scikit 
model = KernelRidge(alpha=lamda, kernel = 'rbf', gamma = gamma)
model.fit(sample_train,label_train)
label_test_pred = model.predict(sample_test)
er_krr = mean_squared_error(label_test, label_test_pred)

# now pick at least five values for m
m_values = [10, 20]

er_base = []
er_yours = []

for m in m_values:
    
    # baseline performance, fixed across m
    er_base.append(er_krr) 
    
    # now, implement AKRR by yourself 
    # ......
    # ......
    # ......
    # store your MSE
    #er_yours.append(mean_squared_error(label_test, label_test_pred))
    er_yours.append(er_krr)

# the following code will plot Figure 1. 
plt.figure()
plt.plot(m_values,er_base, '--o', label='Scikit Error')
plt.plot(m_values,er_yours, label='My Error')
plt.xlabel('m')
plt.ylabel('MSE')
plt.legend()
plt.show()

