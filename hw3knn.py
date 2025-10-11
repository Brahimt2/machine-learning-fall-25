import numpy as np
import matplotlib.pyplot as plt

# prepare data 
data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = data[0:num_train,0:-1]
sample_test = data[n-num_test:,0:-1]
label_train = data[0:num_train,-1]
label_test = data[n-num_test:,-1]

# hyper-parameter for k-NN
k_values = [1, 3, 5, 7, 9]

er_train_k = []
er_test_k = []

for k in k_values:
    
    # train
    train_preds = []
    for i in range(sample_train.shape[0]):
        distances = np.linalg.norm(sample_train - sample_train[i], axis=1)
        
        # get indices of k nearest neighbors
        neighbors_idx = np.argsort(distances)[1:k+1]
        neighbor_labels = label_train[neighbors_idx]
        
        # vote
        pred = np.round(np.mean(neighbor_labels)) 
        train_preds.append(pred)
    
    train_preds = np.array(train_preds)
    er_train_k.append(np.mean(train_preds != label_train))
    
    # test
    test_preds = []
    for i in range(sample_test.shape[0]):
        distances = np.linalg.norm(sample_train - sample_test[i], axis=1)
        neighbors_idx = np.argsort(distances)[:k]
        neighbor_labels = label_train[neighbors_idx]
        pred = np.round(np.mean(neighbor_labels))
        test_preds.append(pred)
    
    test_preds = np.array(test_preds)
    er_test_k.append(np.mean(test_preds != label_test))

# Plot errors
plt.figure()    
plt.plot(k_values, er_train_k, marker='o', label='Training Error')
plt.plot(k_values, er_test_k, marker='d', label='Testing Error')
plt.xlabel('Value of k')
plt.ylabel('Classification Error')
plt.title('k-NN Training and Testing Error')
plt.legend()
plt.grid(True)
plt.show()
