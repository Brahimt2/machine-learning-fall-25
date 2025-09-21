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
lamda_list = [] 

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
    
    
    # now, implement ridge regression by yourself 
    # ......
    # ......
    # ......
    # store your MSE
    er_yours.append(mean_squared_error(label_test, label_test_pred))
    # also compute and store your model sparsity 
    # sp = .....
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
