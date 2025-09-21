import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

num_train = int(0.1*n)
num_test = int(0.25*n)

sample_train = data[0:num_train,0:-1]
sample_test = data[n-num_test:,0:-1]
label_train = data[0:num_train,-1]
label_test = data[n-num_test:,-1]

# pick 5 candidate values that can 
# more comprehensively reflect the 
# the impact of lambda on performance 
lamda_list = [] 

# loop over candidate values in lamda_list
er_base = []
er_yours = []

for lamda in lamda_list:
    
    # baseline implementation using scikit-learn 
    # use it as a reference for your own implementation
    model = linear_model.Ridge(lamda)
    model.fit(sample_train, label_train)
    label_train_pred = model.predict(sample_train)
    label_test_pred = model.predict(sample_test)
    er_base.append(mean_squared_error(label_test, label_test_pred))
    
    # now, implement ridge regression by yourself 
    # ......
    # ......
    # ......
    # store your MSE, assuming you use the same variable names
    er_yours.append(mean_squared_error(label_test, label_test_pred))

# the following code will plot two error curves
# one is your implementation, another is scikit-learn
plt.plot(lamda_list,er_base, '--o', label='Scikit Error')
plt.plot(lamda_list,er_yours, label='My Error')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.legend()
plt.show()