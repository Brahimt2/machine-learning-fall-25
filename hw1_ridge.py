import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# load crimerate csv, skip cells with commas, skip the first row or maybe by 1 row
data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data) # prints out the m and n of the matrix 

num_train = int(0.1*n) # 10% of the data is used to train
num_test = int(0.25*n) # 25% is used to test

sample_train = data[0:num_train,0:-1] # create a 0 to numtrain by 0 to -1 matrix. continue
sample_test = data[n-num_test:,0:-1]
label_train = data[0:num_train,-1]
label_test = data[n-num_test:,-1]

# pick 5 candidate values that can 
# more comprehensively reflect the 
# the impact of lambda on performance 
lamda_list = [0.5, 1, 5, 10, 25, 50]  # what is the lambda ?

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
    I = np.eye(sample_train.shape[1])     # identity matrix 
    XT_X = sample_train.T @ sample_train  # X^T times X      
    LambdaI = lamda * I                   # Lambda times identity matrix    
    XT_Y = sample_train.T @ label_train   # X^T times Y

    beta = np.linalg.inv(XT_X + LambdaI) @ (XT_Y) # beta = (X^{T}X+lambdaI)^{-1}(X^{T}X)
    
    label_test_pred = sample_test @ beta

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
